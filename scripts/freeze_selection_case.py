"""Read-only wedding-case inventory and optional same-format identity hashing.

Output is a new private directory, never an AI baseline, profile or source edit.
The complete flag requires explicit user confirmation; positive-only references
remain unknown outside the provided selection.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from event_cull.case_inventory import inventory_case,verify_reference_copies
from event_cull.source_scan import scan_source_images
from event_cull.calibration import exported_selection_files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("originals","selected","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--case-id",required=True)
    role=parser.add_mutually_exclusive_group()
    role.add_argument("--confirmed-complete",action="store_true")
    role.add_argument("--preculled-source",action="store_true",help="user confirms all remaining originals were already selected; never a waste-culling accuracy benchmark")
    parser.add_argument("--verify-copies",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if args.output.exists() or any(output.is_relative_to(folder.resolve()) for folder in (args.originals,args.selected)):
        parser.error("New isolated output outside all photo folders required")
    started,last=time.monotonic(),0
    inventory=inventory_case(scan_source_images([args.originals],recursive=False),
        exported_selection_files(args.selected,recursive=False),case_id=args.case_id,confirmed_complete=args.confirmed_complete,
        preculled_source=args.preculled_source)
    args.output.mkdir(parents=True)
    try:
        with (args.output/"inventory.json").open("x",encoding="utf-8") as stream:
            json.dump(inventory,stream,ensure_ascii=False,indent=2)
        def progress(current,total):
            nonlocal last
            now=time.monotonic()
            if now-last>=20 or current==total:
                print(f"reference byte identities {current}/{total}; {now-started:.1f}s",flush=True)
                last=now
        copies=verify_reference_copies(inventory,progress=progress) if args.verify_copies else None
        if copies is not None:
            with (args.output/"reference-copies.json").open("x",encoding="utf-8") as stream:
                json.dump(copies,stream,ensure_ascii=False,indent=2)
        report={k:v for k,v in inventory.items() if k!="records"}
        report.update(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,
            reference_copy_verification=None if copies is None else {k:v for k,v in copies.items() if k!="records"},
            ai_accuracy=None,release_approved=False)
        with (args.output/"summary.json").open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        with (args.output/"failed.json").open("x",encoding="utf-8") as stream:
            json.dump({"complete":False,"error":str(exc)},stream,ensure_ascii=False,indent=2)
        raise


if __name__=="__main__":
    main()
