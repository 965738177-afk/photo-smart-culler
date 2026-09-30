"""Local cross-wedding identities, independent from training or AI advice.

Caller supplies the shared scan results. RAW+JPEG use the existing grouping
owner, never count as two captures. The explicit case ID is retained as a
split/identity boundary, never as a learned inference feature.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .engine import _representatives


def digest_file(path,checkpoint=lambda:None):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            checkpoint()
            digest.update(chunk)
    return digest.hexdigest()


def inventory_case(paths,references,*,case_id,confirmed_complete=False,preculled_source=False):
    """Freeze local identities; missing membership is unknown until complete.

    No model, decoder, training, file write or preference activation. File
    stat signatures detect stale reference/source copies across later runs.
    """
    if preculled_source and confirmed_complete:
        raise ValueError("Preculled positives cannot be a complete unfiltered selection benchmark")
    if not isinstance(case_id,str) or not case_id.strip():
        raise ValueError("Explicit unique wedding case ID required")
    photos=_representatives(list(paths))
    references=list(references)
    if not photos or not references or any(not Path(p).is_file() for p in references):
        raise ValueError("Nonempty available originals/reference required")
    by_stem=defaultdict(list)
    for primary,pairs in photos:
        by_stem[primary.stem.casefold()].append((primary,pairs))
    selected=defaultdict(list)
    for path in references:
        selected[Path(path).stem.casefold()].append(Path(path))
    ambiguous={stem for stem,rows in by_stem.items() if len(rows)>1}
    ambiguous|={stem for stem,rows in selected.items() if len({p.parent.resolve() for p in rows})>1}
    unmatched=set(selected)-set(by_stem)
    if ambiguous or unmatched:
        raise ValueError(f"Ambiguous or unmatched wedding identities: {sorted(ambiguous|unmatched)[:10]}")
    def signature(path):
        stat=path.stat()
        return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}
    records=[]
    for primary,pairs in photos:
        stem=primary.stem.casefold()
        records.append({"identity":f"{case_id}:{stem}","primary":signature(primary),
            "companions":[signature(p) for p in pairs],
            "human_retained":True if preculled_source or stem in selected else (False if confirmed_complete else None),
            "human_label_basis":"user_confirmed_preculled_source" if preculled_source else "reference_membership",
            "references":[signature(p) for p in selected.get(stem,[])],"ai_rating":None})
    report={"case_id":case_id,"physical_source_files":sum(len(p) for _,p in photos),
        "logical_photos":len(photos),"source_formats":dict(Counter(p.suffix.lower() for _,pairs in photos for p in pairs)),
        "reference_files":len(references),"reference_unique_captures":len(selected),
        "reference_formats":dict(Counter(Path(p).suffix.lower() for p in references)),
        "reference_confirmed_complete":bool(confirmed_complete),"ambiguous_identities":[],"unmatched_reference":[],
        "source_preculled":bool(preculled_source),
        "full_selection_acceptance_eligible":bool(confirmed_complete) and not preculled_source,
        "role":"curated_positive_control_not_full_selection_validation" if preculled_source else "held_out_wedding_not_used_for_training_or_model_selection",
        "training_performed":False,"model_inference_performed":False,"photos_uploaded":False,
        "active_profile_changed":False,"records":records}
    report["identity_manifest_sha256"]=hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return report


def verify_reference_copies(report,*,checkpoint=lambda:None,progress=None):
    """Compare same-format references to existing companions, not RAW vs JPEG.

    A rendered/edited JPEG may differ legitimately. Byte differences are
    explicit, not evidence of a bad picture or an automatic filename mismatch.
    Two readers bound external-disk load. No image pixels are decoded.
    """
    jobs=[]
    for record in report["records"]:
        for reference in record["references"]:
            candidates=[entry for entry in record["companions"]
                        if Path(entry["path"]).suffix.casefold()==Path(reference["path"]).suffix.casefold()]
            if len(candidates)>1:
                raise ValueError("Ambiguous same-format verification companion")
            if candidates:
                jobs.append((record["identity"],reference,candidates[0]))
    def verify(job):
        identity,reference,source=job
        checkpoint()
        for entry in (reference,source):
            stat=Path(entry["path"]).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(entry["size"],entry["mtime_ns"]):
                raise ValueError("Wedding file changed after inventory")
        left=digest_file(reference["path"],checkpoint)
        right=left if Path(reference["path"]).samefile(source["path"]) else digest_file(source["path"],checkpoint)
        for entry in (reference,source):
            stat=Path(entry["path"]).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(entry["size"],entry["mtime_ns"]):
                raise ValueError("Wedding file changed during identity hashing")
        return {"identity":identity,"reference":reference["path"],"source_companion":source["path"],
                "reference_sha256":left,"source_sha256":right,"same_bytes":left==right}
    verified=[]
    # Futures carry metadata only; at most two workers read 1 MiB byte chunks.
    # No per-file model or decoded image escapes this owner.
    with ThreadPoolExecutor(max_workers=2) as workers:
        for row in workers.map(verify,jobs):
            verified.append(row)
            if progress:
                progress(len(verified),len(jobs))
    return {"verified_files":len(verified),"same_bytes":sum(r["same_bytes"] for r in verified),
            "different_bytes":sum(not r["same_bytes"] for r in verified),
            "not_same_format_comparable":report["reference_files"]-len(jobs),"records":verified,
            "images_decoded":0,"training_performed":False}


def verify_capture_metadata(report,*,reader=None,checkpoint=lambda:None,progress=None,batch_size=64):
    """Check rendered references against original capture metadata, read only.

    Metadata agreement is evidence of pairing, NOT a cryptographic pixel
    identity proof. Missing metadata stays unknown. It never becomes a waste,
    retention or retouch label, and never enters learned ranking features.
    The supplied reader accepts a bounded list of paths and returns ExifTool's
    JSON rows, enabling tests without launching a competing metadata service.
    """
    if type(batch_size) is not int or not 1<=batch_size<=64:
        raise ValueError("Invalid bounded metadata batch")
    jobs=[]
    entries={}
    for record in report["records"]:
        for reference in record["references"]:
            jobs.append((record["identity"],record["primary"],reference))
            for entry in (record["primary"],reference):
                path=str(Path(entry["path"]).resolve())
                if path in entries and entries[path]!=entry:
                    raise ValueError("Conflicting capture signatures")
                entries[path]=entry
    if reader is None:
        from .xmp import _run_bundled_exiftool
        def reader(paths):
            # No assignment option (=), recursion, write or model invocation.
            return json.loads(_run_bundled_exiftool("-j","-DateTimeOriginal","-SubSecTimeOriginal",
                "-Make","-Model","-ImageUniqueID","-RawFileName",*(str(p) for p in paths)))
    observed={}
    paths=list(entries)
    def unchanged(path):
        entry=entries[path]
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(entry["size"],entry["mtime_ns"]):
            raise ValueError("Wedding file changed during capture verification")
    for start in range(0,len(paths),batch_size):
        checkpoint()
        batch=paths[start:start+batch_size]
        for path in batch:
            unchanged(path)
        rows=reader([Path(p) for p in batch])
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError("Malformed capture metadata response")
        found={}
        for row in rows:
            if not isinstance(row.get("SourceFile"),str):
                raise ValueError("Capture response identity missing")
            path=str(Path(row["SourceFile"]).resolve())
            if path not in batch or path in found or row.get("Error"):
                raise ValueError("Unexpected, duplicate or failed capture metadata identity")
            found[path]=row
        if set(found)!=set(batch):
            raise ValueError("Incomplete capture metadata response; no silent subset")
        for path in batch:
            unchanged(path)
        observed.update(found)
        if progress:
            progress(len(observed),len(paths))
    verified=[]
    for identity,source,reference in jobs:
        checkpoint()
        original=observed[str(Path(source["path"]).resolve())]
        rendered=observed[str(Path(reference["path"]).resolve())]
        checks={}
        for key in ("DateTimeOriginal","SubSecTimeOriginal","Make","Model","ImageUniqueID"):
            a,b=original.get(key),rendered.get(key)
            # A zero subsecond value is meaningful, not missing.
            checks[key]=None if a is None or b is None or str(a)=="" or str(b)=="" else str(a).strip()==str(b).strip()
        raw=rendered.get("RawFileName")
        if raw is None or str(raw).strip()=="":
            checks["RawFileName"]=None
        else:
            name=str(raw).strip().replace("\\","/").rsplit("/",1)[-1]
            checks["RawFileName"]=Path(name).stem.casefold()==Path(source["path"]).stem.casefold()
        contradicted=any(value is False for value in checks.values())
        fine=(checks["DateTimeOriginal"] is True and checks["Model"] is True
              and (checks["SubSecTimeOriginal"] is True or checks["RawFileName"] is True))
        status=("contradicted" if contradicted else "consistent_fine_capture_metadata" if fine else
                "consistent_second_only" if checks["DateTimeOriginal"] is True else "unknown")
        fields=("DateTimeOriginal","SubSecTimeOriginal","Make","Model","ImageUniqueID","RawFileName")
        verified.append({"identity":identity,"source":source["path"],"reference":reference["path"],
            "status":status,"checks":checks,"source_metadata":{k:original[k] for k in fields if k in original},
            "reference_metadata":{k:rendered[k] for k in fields if k in rendered}})
    return {"verified_pairs":len(verified),"files_read":len(observed),
        "status_counts":dict(Counter(row["status"] for row in verified)),"records":verified,
        "images_decoded":0,"training_performed":False,"metadata_uploaded":False,
        "pixel_identity_proven":False,"human_labels_modified":False}
