"""Private, best-effort interaction video using the existing camera context."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rlebench.core.media import Media, Recorder, enabled
from .config import CONTROL_HZ, MAX_STEPS
from .render import render

FPS = 10
STRIDE = CONTROL_HZ // FPS
SIZE = 512
HEADER = 64
NAME = "interaction.mp4"


class InteractionVideo:
    def __init__(self, root, trial=1, total=1):
        self.trial, self.total = trial, total
        self.media = Media(root, on=enabled())
        self.writer = self.media.video(NAME, fps=FPS,
                                       max_frames=MAX_STEPS // STRIDE + 1 + FPS)
        self.recorder = Recorder(self.writer, every=STRIDE) if self.writer else None
        self.closed = False

    def render(self, env, steps, answer=None):
        frame = Image.new("RGB", (2*SIZE, SIZE+HEADER), "#edf0f4")
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default(size=16)
        status = f"{steps / CONTROL_HZ:.2f} s | step {steps}"
        if answer is not None:
            status += f" | submitted {answer}"
        for i, (camera, label) in enumerate((("workspace", "Workspace"), ("top", "Top view"))):
            frame.paste(render(env, camera, SIZE), (i*SIZE, HEADER))
            draw.text((i*SIZE+12, 8), f"Trial {self.trial}/{self.total} | {label}", fill="#17212d", font=font)
            draw.text((i*SIZE+12, 34), status, fill="#17212d", font=font)
        return np.asarray(frame)

    def capture(self, env, steps):
        if not self.closed and self.recorder:
            self.recorder.capture(lambda: self.render(env, steps))

    def finish(self, env, steps, answer):
        if self.closed:
            return
        self.closed = True
        if self.recorder and self.recorder.active:
            try:
                frame = self.render(env, steps, answer)
                for _ in range(FPS):
                    self.recorder.add(frame)
            except Exception as exc:
                self.media.skip("final frame", exc)
        self.media.finish(self.writer)
        self.media.close()


def export(destination):
    from .config import CASES, STATE
    media = Media(destination, on=True)
    for i in range(1, len(CASES)+1):
        folder = f"trial-{i:02d}"
        files = Media.export(Path(STATE).parent / "media" / folder, Path(destination) / folder)
        media.files.extend(f"{folder}/{name}" for name in files)
    media.close()
