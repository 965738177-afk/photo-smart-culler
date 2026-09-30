"""Private bounded stdin/stdout person inference protocol; no UI or file IO."""
from __future__ import annotations
import json
import struct
import sys
import numpy as np


MAX_IMAGE_BYTES=1600*1600*3


def read_exact(stream,length):
    chunks=[]
    while length:
        chunk=stream.read(length)
        if not chunk:
            raise EOFError("person worker pipe closed")
        chunks.append(chunk)
        length-=len(chunk)
    return b"".join(chunks)


def main():
    from .human_analysis import HumanAnalysisService
    service=None
    recovery_mode=None
    try:
        while True:
            prefix=sys.stdin.buffer.read(4)
            if not prefix:
                return
            if len(prefix)!=4:
                raise ValueError("truncated person request")
            length=struct.unpack("!I",prefix)[0]
            if not 0<length<=16384:
                raise ValueError("invalid person request size")
            header=json.loads(read_exact(sys.stdin.buffer,length))
            h,w=header["shape"]
            if type(h) is not int or type(w) is not int or not 1<=h<=1600 or not 1<=w<=1600:
                raise ValueError("unbounded person image")
            requested_mode=header.get("recover_faces",False)
            if type(requested_mode) is not bool:
                raise ValueError("invalid face recovery mode")
            if service is None:
                recovery_mode=requested_mode
                service=HumanAnalysisService(recover_faces=recovery_mode)
            elif requested_mode != recovery_mode:
                raise ValueError("person worker mode changed during owned run")
            rgb=np.frombuffer(read_exact(sys.stdin.buffer,h*w*3),np.uint8).reshape(h,w,3)
            try:
                result=service.inspect(rgb,face_regions=header.get("regions",()))
            except Exception as exc:
                result={"status":"error","error":str(exc)[:160]}
            print(json.dumps(result,allow_nan=False,separators=(",",":")),flush=True)
    finally:
        if service is not None:
            service.close()


if __name__=="__main__":
    main()
