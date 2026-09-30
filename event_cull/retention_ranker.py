"""Experimental retention ranking from person evidence and burst comparisons.

No filenames, timestamps, frame numbers, group identifiers, delivery pixels, or
labels enter inference features. Group membership is only used to compare the
observed image measurements. This module never saves/enables a user profile.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from src.core.image_features.human_analysis import FEATURE_DIMENSION, PIPELINE_ID

TECHNICAL_FIELDS = ("sharpness", "subject_sharpness", "highlight_clip", "shadow_clip",
                    "median_luminance", "subject_luminance", "face_coverage",
                    "frontal_face_count", "smiling_face_count", "open_eye_face_count")


def observed_features(results, visual):
    """Build absolute and relative features without looking at human labels."""
    if len(results)!=len(visual) or not np.isfinite(visual).all():
        raise ValueError("Incomplete visual feature matrix")
    persons=[]
    technical=[]
    for item in results:
        person=item.person_evidence
        if (person.get("status")!="ok" or person.get("pipeline")!=PIPELINE_ID
                or len(person.get("features",()))!=FEATURE_DIMENSION):
            raise ValueError("Person evidence incomplete; no silent subset evaluation")
        persons.append(person["features"])
        row=[]
        for key in TECHNICAL_FIELDS:
            value=getattr(item,key)
            # Both the measurement and its availability enter the classifier.
            # In particular, an unobserved subject is not a zero-sharpness face.
            row += [0. if value is None else float(value),float(value is not None)]
        technical.append(row)
    person_features=np.asarray(persons,dtype=np.float32)
    base=np.concatenate((person_features,np.asarray(technical)),axis=1)
    groups=defaultdict(list)
    for index,item in enumerate(results):
        groups[(item.path.parent,item.group_id)].append(index)
    differences=np.zeros_like(base)
    relative_visual=np.zeros_like(visual)
    sizes=np.zeros((len(results),1))
    for indexes in groups.values():
        # Median and range are calculated from images only. No target-derived
        # scene quotas and no training/held-out labels are mixed here.
        values=base[indexes]
        spread=np.maximum(values.max(axis=0)-values.min(axis=0),.05)
        differences[indexes]=(values-np.median(values,axis=0))/spread
        relative_visual[indexes]=visual[indexes]-np.median(visual[indexes],axis=0)
        sizes[indexes]=math.log1p(len(indexes))
    features=np.concatenate((visual,base,differences,relative_visual,sizes),axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite person/relative features")
    return features.astype(np.float32)


class RetentionRanker:
    """Small regularized nonlinear model; no hidden random frame holdout."""
    def __init__(self, *, checkpoint=lambda:None, strategy="boosted"):
        if strategy not in {"boosted", "linear", "rbf", "neural"}:
            raise ValueError("Unknown experimental retention strategy")
        self._checkpoint=checkpoint
        self.strategy=strategy
        self._model=None
        self.positive_fraction=None

    def fit(self, features, labels):
        # Training-only dependency, deliberately not imported by the GUI.
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        labels=np.asarray(labels,dtype=np.int8)
        if len(features)!=len(labels) or labels.sum()<12 or (1-labels).sum()<12:
            raise ValueError("Insufficient retained/non-retained examples")
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite training features")
        self.positive_fraction=float(labels.mean())
        if self.strategy != "boosted":
            # Fit preprocessing ONLY on training frames. A margin classifier
            # tests whether oblique visual directions are lost by axis-aligned
            # trees; its sigmoid is a ranking score, not a calibrated chance.
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import LinearSVC, SVC
            from sklearn.exceptions import ConvergenceWarning
            import warnings
            if self.strategy == "neural":
                from sklearn.neural_network import MLPClassifier
                estimator=MLPClassifier(hidden_layer_sizes=(128,32),alpha=.1,batch_size=128,
                                        learning_rate_init=.0005,early_stopping=False,random_state=0)
            elif self.strategy == "linear":
                estimator=LinearSVC(C=.02,class_weight="balanced",max_iter=3000,random_state=0)
            else:
                estimator=SVC(C=2.,kernel="rbf",gamma="scale",class_weight="balanced",
                              max_iter=10000,cache_size=256)
            model=make_pipeline(StandardScaler(), estimator)
            self._checkpoint()
            with threadpool_limits(limits=4), warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                if self.strategy == "neural":
                    transformed=model.steps[0][1].fit_transform(features)
                    # Fixed training-only epochs, with checkpoints between each
                    # pass. Do not use shuffled neighbour frames as early-stop
                    # validation or choose an epoch from held-out photo labels.
                    for _ in range(120):
                        self._checkpoint()
                        estimator.partial_fit(transformed,labels,classes=np.asarray([0,1]))
                else:
                    model.fit(features,labels)
            self._checkpoint()
            self._model=model
            return self
        model=HistGradientBoostingClassifier(
            learning_rate=.05,max_leaf_nodes=15,max_depth=5,min_samples_leaf=30,
            l2_regularization=5.,max_bins=128,early_stopping=False,
            warm_start=True,random_state=0,max_iter=20)
        # Fixed design, not chosen by testing the user's held-out filenames.
        # Small chunks let cancellation leave a previously working model intact.
        with threadpool_limits(limits=4):
            for count in range(20,161,20):
                self._checkpoint()
                model.set_params(max_iter=count)
                model.fit(features,labels)
        self._checkpoint()
        self._model=model
        return self

    def predict(self, features):
        from threadpoolctl import threadpool_limits
        if self._model is None:
            raise ValueError("Retention ranker has not been trained")
        self._checkpoint()
        with threadpool_limits(limits=4):
            if self.strategy in {"boosted","neural"}:
                values=self._model.predict_proba(features)[:,1]
            else:
                margin=self._model.decision_function(features)
                values=1/(1+np.exp(-np.clip(margin,-30,30)))
        if not np.isfinite(values).all():
            raise ValueError("Invalid retention prediction")
        self._checkpoint()
        return values

    def close(self):
        """Release a caller-owned experiment without saving or activating it."""
        self._model = None
        self.positive_fraction = None


def combine_pairwise_scores(absolute, relative):
    """Fixed blending shared by same-case and whole-case validation."""
    absolute, relative = np.asarray(absolute), np.asarray(relative)
    if (absolute.shape != relative.shape or absolute.ndim != 1
            or not np.isfinite(absolute).all() or not np.isfinite(relative).all()
            or ((absolute < 0) | (absolute > 1)).any()
            or ((relative < 0) | (relative > 1)).any()):
        raise ValueError("Invalid absolute/relative ranking scores")
    absolute = np.clip(absolute, 1e-6, 1-1e-6)
    relative = np.clip(relative, 1e-6, 1-1e-6)
    logits = np.log(absolute/(1-absolute)) + .5*np.log(relative/(1-relative))
    return 1/(1+np.exp(-np.clip(logits, -30, 30)))


def same_group_pairs(groups, *, maximum_opponents=8):
    """At most eight relative comparisons per frame, without quadratic work."""
    members=defaultdict(list)
    for i,group in enumerate(groups):
        members[group].append(i)
    left,right=[],[]
    for indexes in members.values():
        if len(indexes)<2:
            continue
        for position,index in enumerate(indexes):
            # Small bursts compare every other frame; longer bursts use evenly
            # spread opponents. Position picks comparison partners only, never
            # enters learned features or decides which photograph is retained.
            others=[i for i in indexes if i!=index]
            if len(others)>maximum_opponents:
                others=[others[i] for i in np.linspace(0,len(others)-1,maximum_opponents,dtype=int)]
            left.extend([index]*len(others))
            right.extend(others)
    return np.asarray(left,dtype=int),np.asarray(right,dtype=int)


class PairwiseRetentionRanker(RetentionRanker):
    """Learn relative decisions from observed within-burst differences."""
    def fit_pairs(self, features, labels, groups):
        left,right=same_group_pairs(groups)
        labels=np.asarray(labels,dtype=np.int8)
        different=labels[left]!=labels[right]
        if different.sum()<24:
            raise ValueError("Not enough different-label burst comparisons")
        self.fit(features[left[different]]-features[right[different]],labels[left[different]])
        return self

    def compare(self, features, groups):
        left,right=same_group_pairs(groups)
        scores=np.full(len(features),.5)
        if not len(left):
            return scores
        totals=np.zeros(len(features))
        counts=np.zeros(len(features))
        # Bounded feature memory and cooperative cancellation even on a large
        # wedding. Singleton bursts stay neutral, not falsely "best".
        for start in range(0,len(left),512):
            a,b=left[start:start+512],right[start:start+512]
            values=self.predict(features[a]-features[b])
            np.add.at(totals,a,values)
            np.add.at(counts,a,1)
        available=counts>0
        scores[available]=totals[available]/counts[available]
        return scores


class PairedRetentionRanker:
    """One transactional owner for existing absolute and pair classifiers.

    Parameters and blending match the existing boosted-pairwise experiment;
    this is not a new visual model. Test-case labels never enter this owner.
    """
    def __init__(self, *, checkpoint=lambda: None, absolute_factory=RetentionRanker,
                 pair_factory=PairwiseRetentionRanker):
        self._checkpoint = checkpoint
        self._absolute_factory, self._pair_factory = absolute_factory, pair_factory
        self._absolute = self._pair = None
        self.positive_fraction = None

    def fit(self, features, labels, groups):
        if len(features) != len(groups):
            raise ValueError("Incomplete training group identities")
        absolute = pair = None
        try:
            absolute = self._absolute_factory(checkpoint=self._checkpoint, strategy="boosted")
            pair = self._pair_factory(checkpoint=self._checkpoint, strategy="boosted")
            absolute.fit(features, labels)
            pair.fit_pairs(features, labels, groups)
            self._checkpoint()
        except BaseException:
            for owner in (absolute, pair):
                if owner is not None:
                    owner.close()
            raise
        self.close()
        self._absolute, self._pair = absolute, pair
        self.positive_fraction = absolute.positive_fraction
        return self

    def predict(self, features, groups):
        if self._absolute is None or self._pair is None:
            raise ValueError("Paired retention ranker has not been trained")
        if len(features) != len(groups):
            raise ValueError("Incomplete prediction group identities")
        self._checkpoint()
        return combine_pairwise_scores(self._absolute.predict(features), self._pair.compare(features, groups))

    def close(self):
        for owner in (self._absolute, self._pair):
            if owner is not None:
                owner.close()
        self._absolute = self._pair = None
        self.positive_fraction = None
