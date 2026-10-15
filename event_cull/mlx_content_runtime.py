"""Optional research-only MLX owner shared by single-frame and A/B probes.

Lazy dependency imports keep the production app independent of MLX. One owner
loads verified local model data once, serializes generation, and checkpoints
between tokens. No downloads, remote code, HTTP service or photo mutations.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import time


def validate_model_files(root,manifest):
    root=Path(root).resolve()
    for record in manifest['files']:
        path=root/record['file']
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size!=record['bytes']:
            raise ValueError('Pinned model inventory mismatch')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(chunk)
        if digest.hexdigest()!=record['sha256']:
            raise ValueError('Pinned model checksum mismatch')


class MlxContentRuntime:
    def __init__(self,root,manifest, *, checkpoint=lambda:None):
        self.checkpoint=checkpoint
        self.lock=threading.Lock()
        self.closed=False
        validate_model_files(root,manifest)
        self.checkpoint()
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        started=time.monotonic()
        self.model,self.processor=load(str(root),trust_remote_code=False,local_files_only=True)
        self.config=load_config(root)
        self.load_seconds=time.monotonic()-started
        self.last_reply=None

    def chat(self,encoded_images,prompt,schema, *, multiscale=False):
        while not self.lock.acquire(timeout=.1):
            self.checkpoint()
        try:
            self.checkpoint()
            self.last_reply=None
            if self.closed:
                raise RuntimeError('Content runtime closed')
            from PIL import Image
            from mlx_vlm import stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.structured import build_json_schema_logits_processor
            pictures=[Image.open(io.BytesIO(base64.b64decode(s))).convert('RGB') for s in encoded_images]
            if multiscale:
                if len(pictures)!=1:
                    raise ValueError('Multiscale views require one original photo')
                picture=pictures[0];w,h=picture.size
                pictures=[picture.copy()]+[picture.crop((x*w//2,y*h//2,(x+1)*w//2,(y+1)*h//2))
                                          for y in range(2) for x in range(2)]
                for p in pictures:
                    p.thumbnail((896,896))
                prompt='图片1为完整照片，图片2至5为同一张照片的四个放大局部，并非不同时间或人物组合。结合全图和局部观察。\n'+prompt
            formatted=apply_chat_template(self.processor,self.config,prompt,num_images=len(pictures))
            grammar=build_json_schema_logits_processor(self.processor.tokenizer,schema)
            text=''
            for part in stream_generate(self.model,self.processor,formatted,image=pictures,max_tokens=360,
                                        temperature=0.0,logits_processors=[grammar]):
                self.checkpoint()
                if not getattr(part,'is_draft',False):
                    text+=part.text
            self.last_reply=text
            text=text.strip()
            if text.startswith('```json') and text.endswith('```'):
                text=text[7:-3].strip()
            return json.loads(text)
        finally:
            self.lock.release()

    def clear_cache(self):
        import mlx.core as mx
        mx.clear_cache()

    def close(self):
        with self.lock:
            self.closed=True
            self.model=None
            self.processor=None
            self.clear_cache()
