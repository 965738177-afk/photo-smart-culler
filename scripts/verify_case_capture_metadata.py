"""Read-only capture pairing audit; metadata is never a learned feature."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

from event_cull.calibration import exported_selection_files
from event_cull.case_inventory import inventory_case,verify_capture_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","selected","output"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--case-id",required=True)
    args=parser.parse_args()
    raw=args.baseline.read_bytes()
    payload=json.loads(raw)
    paths=[Path(row["folder"])/row["file"] for row in payload["photos"]]
    if args.output.exists() or any(args.output.resolve().is_relative_to(folder.resolve())
            for folder in {p.parent for p in paths}|{args.selected}):
        parser.error("new isolated audit output required")
    # This audit records identities only. It does not need complete-negative
    # reference labels and must not invent them from an incomplete selection.
    inventory=inventory_case(paths,exported_selection_files(args.selected,recursive=False),case_id=args.case_id)
    args.output.mkdir(parents=True)
    started,last=time.monotonic(),0
    try:
        def progress(current,total):
            nonlocal last
            now=time.monotonic()
            if now-last>=20 or current==total:
                print(f"capture metadata {current}/{total}; {now-started:.1f}s",flush=True)
                last=now
        report=verify_capture_metadata(inventory,progress=progress)
        report.update(complete=True,case_id=args.case_id,baseline_sha256=hashlib.sha256(raw).hexdigest(),
                      created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started)
        for name,content in (("inventory.json",inventory),("capture-identity.json",report)):
            with (args.output/name).open("x",encoding="utf-8") as stream:
                json.dump(content,stream,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!="records"},ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        with (args.output/"failed.json").open("x",encoding="utf-8") as stream:
            json.dump({"complete":False,"error":str(exc)},stream,ensure_ascii=False,indent=2)
        raise


if __name__=="__main__":
    main()
