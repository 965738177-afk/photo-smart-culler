"""A focused native review screen using the app's injected preview service."""
import queue
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from event_cull.preview_warmup import PreviewWarmup


class DuplicateReviewDialog(tk.Toplevel):
    def __init__(self, parent, session, *, load_preview, preview_source, on_change):
        super().__init__(parent)
        self.session, self.preview_source, self.on_change = session, preview_source, on_change
        self.title("秒选 · A/B 人工复核")
        self.configure(bg="#17191b")
        self.geometry(f"{min(1120, self.winfo_screenwidth() - 60)}x{min(780, self.winfo_screenheight() - 70)}")
        self.minsize(800, 620)
        self.transient(parent)
        self.grab_set()
        self._generation = 0
        self._images = {}
        self._display_images = []
        self._events = queue.Queue()
        self._closed = False
        self._loaded_paths = set()
        # Reuse generation-aware scheduling; pixels cross the UI queue, never
        # instantiate Tk images on a worker or decode again on the UI thread.
        self._warmup = PreviewWarmup(
            load_preview, lambda _generation, _path: None,
            result_ready=lambda generation, path, image: self._events.put((generation, path, image)),
            failed=lambda generation, path, error: self._events.put((generation, path, error)))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.bind("<Escape>", lambda _e: self.close())
        header = tk.Frame(self, bg="#202326", padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="重复照片 · A/B 人工复核", fg="#fffaf0", bg="#202326",
                 font=("Helvetica Neue", 18, "bold")).pack(anchor="w")
        self.progress_text = tk.Label(header, fg="#c3c3bf", bg="#202326")
        self.progress_text.pack(anchor="w", pady=(7, 0))
        tk.Label(header, text="相似不等于重复：不同人物或合影组合可以都留。未展示的照片不会被批量淘汰。",
                 fg="#c3c3bf", bg="#202326").pack(anchor="w", pady=(4, 0))
        body = tk.Frame(self, bg="#17191b", padx=16, pady=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure((0, 1), weight=1, uniform="pair")
        body.rowconfigure(1, weight=1)
        self.names, self.canvases, self.details = [], [], []
        for column, label in enumerate(("A", "B")):
            name = tk.Label(body, fg="#fffaf0", bg="#17191b", anchor="w",
                            font=("Helvetica Neue", 13, "bold"))
            name.grid(row=0, column=column, sticky="ew", padx=8, pady=(0, 10))
            canvas = tk.Canvas(body, bg="#101214", highlightbackground="#494d50", highlightthickness=1)
            canvas.grid(row=1, column=column, sticky="nsew", padx=8)
            canvas.bind("<Configure>", lambda _e: self._render())
            canvas.bind("<Double-Button-1>", lambda _e: self._toggle_zoom())
            canvas.bind("<Button-1>", self._start_pan)
            canvas.bind("<B1-Motion>", self._pan)
            detail = tk.Label(body, fg="#c3c3bf", bg="#17191b", anchor="w", justify="left", wraplength=400)
            detail.grid(row=2, column=column, sticky="ew", padx=8, pady=10)
            self.names.append(name)
            self.canvases.append(canvas)
            self.details.append(detail)
        footer_shell = tk.Frame(self, bg="#202326")
        footer_shell.pack(fill="x")
        footer = tk.Frame(footer_shell, bg="#202326", padx=20, pady=10)
        footer.pack()
        self.buttons = []
        for label, action in (("← 选 A", "a"), ("选 B →", "b"), ("↑ 都留", "both"),
                              ("↓ 都不留", "neither"), ("S 稍后", "skip")):
            button = ttk.Button(footer, text=label, command=lambda value=action: self.act(value))
            button.pack(side="left", padx=4)
            self.buttons.append(button)
        tools = tk.Frame(footer_shell, bg="#202326", pady=6)
        tools.pack()
        self.undo_button = ttk.Button(tools, text="撤销 · Ctrl / ⌘ Z", command=self.undo)
        self.undo_button.pack(side="left", padx=4)
        ttk.Button(tools, text="同步放大 · Z / 双击；拖动查看细节", command=self._toggle_zoom).pack(side="left", padx=4)
        self.note = tk.Label(self, text="仅更新人工评级：保留为 2★；不删除照片，不自动写入 XMP。",
                             fg="#c3c3bf", bg="#17191b", pady=10)
        self.note.pack(fill="x")
        self.revisit_button = ttk.Button(self, text="继续稍后处理的组", command=self.revisit)
        self._pressed_keys = set()
        self.bind("<KeyRelease>", lambda event: self._pressed_keys.discard(event.keysym), add="+")
        for key, action in (("<Left>", "a"), ("<Right>", "b"), ("<Up>", "both"),
                            ("<Down>", "neither"), ("s", "skip")):
            self.bind(key, lambda event, value=action: self._key_action(event, value))
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Command-z>", lambda _e: self.undo())
        self.bind("z", lambda _e: self._toggle_zoom())
        self._zoom = False
        self._pan_offset = (0, 0)
        self._poll_id = self.after(50, self._poll)
        self._show_pair()
        self.focus_set()

    def _show_pair(self):
        self._generation += 1
        self._images.clear()
        self._loaded_paths.clear()
        pair = self.session.pair
        finished = self.session.total_groups - len(self.session.pending) - len(self.session.deferred)
        self.progress_text.configure(text=f"已处理 {finished}/{self.session.total_groups} 组 · "
                                     f"待处理 {len(self.session.pending)} · 稍后 {len(self.session.deferred)}")
        self._zoom = False
        self._pan_offset = (0, 0)
        self.revisit_button.pack_forget()
        if not pair:
            self._warmup.cancel(self._generation)
            self.note.configure(text="当前队列已结束；单独剩余的照片保持原评级，未自动确认。")
            if self.session.deferred:
                self.revisit_button.pack(pady=(0, 12))
        else:
            self.note.configure(text="仅更新人工评级：保留为 2★；不删除照片，不自动写入 XMP。")
            paths = [self.preview_source(p.path, p.pair_paths) for p in pair]
            self._warmup.submit(paths, self._generation)
        for index, name in enumerate(self.names):
            item = pair[index] if pair else None
            name.configure(text=f"{'AB'[index]}  {item.path.name}" if item else "复核完成")
            self.details[index].configure(text=(
                f"组 {item.group_id} · AI 评分 {item.score:.1f}（非置信度）\n"
                f"当前 {item.suggested_rating}★ · 正面人脸 {item.frontal_face_count} · "
                f"闭眼提示 {'有' if item.eye_penalty else '无'}" if item else ""))
        self.undo_button.configure(state="normal" if self.session.can_undo else "disabled")
        self._update_actions()
        self._render()

    def _update_actions(self):
        pair = self.session.pair
        ready = bool(pair) and all(self.preview_source(p.path, p.pair_paths) in self._loaded_paths for p in pair)
        for index, button in enumerate(self.buttons):
            button.configure(state="normal" if (ready or (index == 4 and pair)) else "disabled")

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                generation, path, value = self._events.get_nowait()
                if generation != self._generation:
                    continue
                if isinstance(value, Image.Image):
                    self._images[path] = value
                    self._loaded_paths.add(path)
                elif isinstance(value, str):
                    self._images[path] = "预览不可用，请稍后处理\n" + value[:120]
                self._render()
                self._update_actions()
        except queue.Empty:
            pass
        self._poll_id = self.after(50, self._poll)

    def _render(self):
        self._display_images.clear()
        for index, canvas in enumerate(self.canvases):
            canvas.delete("all")
            w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            pair = self.session.pair
            image = self._images.get(self.preview_source(pair[index].path, pair[index].pair_paths)) if pair else None
            if isinstance(image, Image.Image):
                scale = min(w / image.width, h / image.height) * (2 if self._zoom else 1)
                shown = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
                display = ImageTk.PhotoImage(shown, master=self)
                self._display_images.append(display)
                dx = self._pan_offset[0] * max(0, (shown.width - w) / 2)
                dy = self._pan_offset[1] * max(0, (shown.height - h) / 2)
                canvas.create_image(w / 2 + dx, h / 2 + dy, image=display)
            else:
                canvas.create_text(w / 2, h / 2, text=image or ("正在准备预览…" if pair else "当前队列已完成"),
                                   fill="#c3c3bf", width=max(100, w - 40))

    def _toggle_zoom(self):
        self._zoom = not self._zoom
        self._pan_offset = (0, 0)
        self._render()

    def _start_pan(self, event):
        self._pan_origin = (event.x, event.y, self._pan_offset)

    def _pan(self, event):
        if not self._zoom or not hasattr(self, "_pan_origin"):
            return
        x, y, old = self._pan_origin
        self._pan_offset = (max(-1, min(1, old[0] + (event.x - x) / max(1, event.widget.winfo_width() / 2))),
                            max(-1, min(1, old[1] + (event.y - y) / max(1, event.widget.winfo_height() / 2))))
        self._render()

    def act(self, action):
        if self._closed or not self.session.pair:
            return "break"
        if action != "skip" and not all(self.preview_source(p.path, p.pair_paths) in self._loaded_paths
                                         for p in self.session.pair):
            return "break"  # Never confirm unseen or missing previews via hotkeys.
        if self.session.decide(action):
            self.on_change()
            self._show_pair()
        return "break"

    def _key_action(self, event, action):
        if event.keysym not in self._pressed_keys:
            self._pressed_keys.add(event.keysym)
            self.act(action)
        return "break"  # Holding an arrow must not confirm multiple pairs.

    def undo(self):
        if self.session.undo():
            self.on_change()
            self._show_pair()
        return "break"

    def revisit(self):
        self.session.revisit_deferred()
        self._show_pair()

    def close(self):
        self._shutdown()
        self.destroy()

    def _on_destroy(self, event):
        if event.widget is self:
            self._shutdown()

    def _shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._warmup.close()
        self.after_cancel(self._poll_id)
