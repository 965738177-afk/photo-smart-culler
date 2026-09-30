"""Resolve managed, pinned retention model weights for local evaluation only."""
from __future__ import annotations
import sys
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--semantic-regions",action="store_true")
    mode.add_argument("--semantic-concepts",action="store_true")
    mode.add_argument("--tiny-vision",action="store_true")
    args=parser.parse_args()
    # Existing Qt engine owns provisioning; no new ad-hoc model download path.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from core.model_provisioning import (EMBEDDING_MODEL, SEMANTIC_RETENTION_MODEL,
                                        SEMANTIC_RETENTION_TEXT_MODEL, TINY_RETENTION_MODEL, resolve_snapshot)
    model=(TINY_RETENTION_MODEL if args.tiny_vision else
           SEMANTIC_RETENTION_TEXT_MODEL if args.semantic_concepts else
           SEMANTIC_RETENTION_MODEL if args.semantic_regions else EMBEDDING_MODEL)
    print(resolve_snapshot(model,allow_download=True),flush=True)


if __name__=="__main__":
    main()
