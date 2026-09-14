"""Render public-seed motion previews: raw RGB beside simulator truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from harness import episodes, render, spec
from rlebench.core.media import Media
from .audit_motion import motion_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23])
    args = parser.parse_args()
    media = Media(args.output, on=True)
    reports = []
    for shape in spec.BLOCK_SHAPES:
        for seed in args.seeds:
            ep = episodes.push_episode(seed, shapes=(shape,), max_frames=60)
            name = f"{shape}-seed{seed}.mp4"
            writer = media.video(name, size=(2*spec.IMG_W, spec.IMG_H+86), fps=spec.FRAME_HZ)
            if writer is None:
                raise RuntimeError("video writer unavailable")
            yaw = np.unwrap(np.asarray(ep.gts)[:, 2])
            hidden_turn = 0.0
            for i, frame in enumerate(ep.frames):
                if i and frame.occluded and ep.frames[i-1].occluded:
                    hidden_turn += abs(float(np.degrees(yaw[i]-yaw[i-1])))
                raw = Image.fromarray(frame.obs["rgb"])
                truth = raw.copy()
                draw = ImageDraw.Draw(truth)
                for stroke in render._pose_strokes(frame.gt, shape):
                    draw.line([tuple(p) for p in stroke], fill=render.TRUTH, width=3)
                panels = []
                for observation, label in zip((raw, truth), ("RAW OBSERVATION", "SIMULATOR TRUTH (not prediction)")):
                    panel = Image.new("RGB", (spec.IMG_W, spec.IMG_H+86))
                    panel.paste(observation, (0,86))
                    draw = ImageDraw.Draw(panel)
                    lines = [f"{shape} seed={seed}  {label}",
                             f"t={frame.t:.1f}s yaw={np.degrees(frame.gt[2]):.1f} deg",
                             f"{'OCCLUDED' if frame.occluded else 'VISIBLE'} | turn during occluded intervals: {hidden_turn:.1f} deg"]
                    for row, line in enumerate(lines):
                        draw.text((8,5+row*25),line,font=render._font(15),
                                  fill=render.WARN if frame.occluded else render.TRUTH)
                    panels.append(panel)
                writer.add(np.concatenate([np.asarray(p) for p in panels],axis=1))
            media.finish(writer)
            report = dict(video=name,seed=seed,shape=shape,**motion_metrics(ep),
                          max_tilt_deg=ep.max_tilt_deg,max_height_m=ep.max_height_m)
            reports.append(report)
            print(json.dumps(report),flush=True)
    index = media.report()
    if index.get("skipped"):
        raise RuntimeError(index["skipped"])
    (args.output/"motion.json").write_text(json.dumps(reports,indent=2))


if __name__ == "__main__":
    main()
