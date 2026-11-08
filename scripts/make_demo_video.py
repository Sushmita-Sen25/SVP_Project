"""Render the synthetic demo video used by the video-analysis page.

Eight procedurally drawn scenes (static, panning, bouncing objects, walking
figures, traffic, a fast colour pattern, a night scene) joined by hard cuts, a
cross-dissolve and a fade through black. Ground-truth scene boundaries are
written next to the video so the page can score its own detection.

Usage:  pip install opencv-python numpy imageio-ffmpeg
        python scripts/make_demo_video.py
"""
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:  # fall back to ffmpeg on PATH
    FFMPEG = "ffmpeg"

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 640, 360, 25
rng = np.random.default_rng(7)


def gradient(top, bottom):
    t = np.linspace(0, 1, H)[:, None, None]
    col = (1 - t) * np.array(top, float) + t * np.array(bottom, float)
    return np.repeat(col, W, axis=1).astype(np.uint8)


def label(img, text):
    cv2.putText(img, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def walker(img, x, ground, phase, color, scale=1.0, facing=1):
    """Stick figure with swinging limbs; x = hip position."""
    s = scale
    hip = np.array([x, ground - 62 * s])
    neck = hip + [0, -48 * s]
    head = neck + [0, -16 * s]
    th = max(2, int(9 * s))
    swing = math.sin(phase)
    for sgn in (1, -1):
        a = 0.55 * swing * sgn
        knee = hip + [math.sin(a) * 32 * s, math.cos(a) * 32 * s]
        b = a - 0.35 * max(0.0, math.sin(phase + (0 if sgn > 0 else math.pi)))
        foot = knee + [math.sin(b) * 32 * s, math.cos(b) * 32 * s]
        cv2.line(img, tuple(hip.astype(int)), tuple(knee.astype(int)), color, th, cv2.LINE_AA)
        cv2.line(img, tuple(knee.astype(int)), tuple(foot.astype(int)), color, th, cv2.LINE_AA)
        arm_a = -0.6 * swing * sgn
        elbow = neck + [math.sin(arm_a) * 24 * s, 6 * s + math.cos(arm_a) * 22 * s]
        hand = elbow + [math.sin(arm_a + 0.3 * facing) * 22 * s, math.cos(arm_a) * 20 * s]
        cv2.line(img, tuple((neck + [0, 4 * s]).astype(int)), tuple(elbow.astype(int)), color, max(2, th - 2), cv2.LINE_AA)
        cv2.line(img, tuple(elbow.astype(int)), tuple(hand.astype(int)), color, max(2, th - 2), cv2.LINE_AA)
    cv2.line(img, tuple(hip.astype(int)), tuple(neck.astype(int)), color, th + 6, cv2.LINE_AA)
    cv2.circle(img, tuple(head.astype(int)), int(13 * s), (90, 160, 225), -1, cv2.LINE_AA)


# ------------------------------------------------------------------ scenes
def scene_sunrise(n):
    for i in range(n):
        t = i / n
        img = gradient((90 + 60 * t, 40 + 50 * t, 30 + 20 * t), (60 + 40 * t, 140 + 70 * t, 230))
        sun = (W // 2 + 60, int(300 - 160 * t))
        cv2.circle(img, sun, 42, (120, 220, 255), -1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 290), (W, H), (40, 70, 40), -1)
        yield label(img, "Scene 1 - sunrise")


def scene_balls(n):
    balls = [
        dict(p=np.array([100.0, 100.0]), v=np.array([170.0, 120.0]), r=30, c=(60, 80, 230)),
        dict(p=np.array([400.0, 200.0]), v=np.array([-140.0, 160.0]), r=24, c=(80, 200, 90)),
        dict(p=np.array([300.0, 80.0]), v=np.array([120.0, -150.0]), r=34, c=(230, 160, 50)),
    ]
    base = np.full((H, W, 3), (38, 30, 28), np.uint8)
    for x in range(0, W, 40):
        cv2.line(base, (x, 0), (x, H), (52, 44, 40), 1)
    for y in range(0, H, 40):
        cv2.line(base, (0, y), (W, y), (52, 44, 40), 1)
    for _ in range(n):
        img = base.copy()
        for b in balls:
            b["p"] += b["v"] / FPS
            for k, lim in ((0, W), (1, H)):
                if b["p"][k] < b["r"] or b["p"][k] > lim - b["r"]:
                    b["v"][k] *= -1
                    b["p"][k] = min(max(b["p"][k], b["r"]), lim - b["r"])
            cv2.circle(img, tuple(b["p"].astype(int)), b["r"], b["c"], -1, cv2.LINE_AA)
        yield label(img, "Scene 2 - bouncing balls")


def scene_pan(n):
    width = W + 60 * n // FPS + 10
    canvas = gradient((235, 200, 150), (250, 230, 200))
    canvas = cv2.resize(canvas, (width, H))
    for layer, (col, amp, base_y, freq) in enumerate([((150, 140, 110), 40, 190, 0.006), ((90, 120, 70), 50, 240, 0.011), ((50, 90, 40), 35, 290, 0.019)]):
        xs = np.arange(width)
        ys = (base_y + amp * np.sin(xs * freq + layer) + 0.4 * amp * np.sin(xs * freq * 2.7 + 2 * layer)).astype(int)
        for x in range(width):
            canvas[ys[x]:, x] = col
    for x in range(0, width, 140):  # trees
        cv2.rectangle(canvas, (x + 60, 250), (x + 68, 300), (30, 50, 80), -1)
        cv2.circle(canvas, (x + 64, 240), 22, (40, 110, 50), -1, cv2.LINE_AA)
    for i in range(n):
        off = int(60 * i / FPS)
        yield label(canvas[:, off:off + W].copy(), "Scene 3 - camera pan")


def street_bg():
    img = gradient((220, 190, 160), (235, 215, 195))
    xs = 0
    while xs < W:
        w = int(rng.integers(60, 120))
        h = int(rng.integers(120, 220))
        shade = int(rng.integers(90, 150))
        cv2.rectangle(img, (xs, 280 - h), (xs + w, 280), (shade, shade - 10, shade - 25), -1)
        for wy in range(280 - h + 12, 270, 26):
            for wx in range(xs + 8, xs + w - 14, 20):
                cv2.rectangle(img, (wx, wy), (wx + 9, wy + 12), (170, 200, 220), -1)
        xs += w + 6
    cv2.rectangle(img, (0, 280), (W, H), (120, 120, 125), -1)
    cv2.line(img, (0, 282), (W, 282), (200, 200, 200), 2)
    return img


def scene_walkers(n):
    bg = street_bg()
    for i in range(n):
        t = i / FPS
        img = bg.copy()
        walker(img, -40 + 52 * t, 345, t * 6.0, (40, 40, 170), 1.0, 1)
        walker(img, W + 40 - 38 * t, 335, t * 5.0 + 1, (150, 60, 30), 0.85, -1)
        yield label(img, "Scene 4 - pedestrians")


def scene_title(n):
    for i in range(n):
        img = gradient((70, 30, 20), (40, 20, 10))
        cv2.putText(img, "Interlude", (190, 190), cv2.FONT_HERSHEY_DUPLEX, 1.8, (230, 230, 240), 2, cv2.LINE_AA)
        cv2.putText(img, "static title card", (232, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 190), 1, cv2.LINE_AA)
        yield label(img, "Scene 5 - title")


def car(img, x, y, color, length=110):
    x = int(x)
    cv2.rectangle(img, (x, y - 26), (x + length, y), color, -1)
    cv2.rectangle(img, (x + 22, y - 46), (x + length - 26, y - 24), color, -1)
    cv2.rectangle(img, (x + 30, y - 42), (x + length - 34, y - 28), (220, 220, 200), -1)
    for wx in (x + 22, x + length - 22):
        cv2.circle(img, (wx, y), 12, (30, 30, 30), -1, cv2.LINE_AA)


def scene_traffic(n):
    bg = gradient((215, 185, 140), (230, 210, 180))
    cv2.rectangle(bg, (0, 200), (W, H), (80, 80, 85), -1)
    for x in range(0, W, 60):
        cv2.rectangle(bg, (x, 276), (x + 30, 282), (220, 220, 220), -1)
    for i in range(n):
        t = i / FPS
        img = bg.copy()
        car(img, -120 + 70 * t, 262, (40, 60, 200))
        car(img, W + 20 - 95 * t, 335, (180, 120, 40), 130)
        car(img, -400 + 110 * t, 262, (60, 160, 60), 100)
        yield label(img, "Scene 6 - traffic")


def scene_pattern(n):
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.hypot(yy - cy, xx - cx)
    for i in range(n):
        t = i / FPS
        v = np.sin(6 * ang + 2.2 * t) * np.cos(rad / 18 - 3 * t)
        hue = ((ang / np.pi * 90 + 40 * t + rad / 4) % 180).astype(np.uint8)
        hsv = np.dstack([hue, np.full_like(hue, 200), (150 + 100 * v).clip(0, 255).astype(np.uint8)])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        yield label(img, "Scene 7 - kaleidoscope")


def scene_night(n):
    stars = rng.integers(0, [W, 250], size=(80, 2))
    for i in range(n):
        t = i / FPS
        img = gradient((40, 18, 10), (70, 35, 20))
        for k, (x, y) in enumerate(stars):
            b = int(150 + 100 * math.sin(t * 3 + k))
            img[y, x] = (b, b, b)
        cv2.circle(img, (520, 70), 28, (200, 230, 240), -1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 300), (W, H), (30, 40, 30), -1)
        walker(img, -40 + 50 * t, 345, t * 6.0, (60, 160, 200), 0.9, 1)
        yield label(img, "Scene 8 - night walk")


SCENES = [
    ("sunrise", scene_sunrise, 14), ("balls", scene_balls, 16), ("pan", scene_pan, 14),
    ("walkers", scene_walkers, 18), ("title", scene_title, 8), ("traffic", scene_traffic, 16),
    ("pattern", scene_pattern, 14), ("night", scene_night, 14),
]
# transition INTO scene i: "cut", "dissolve" (1 s cross-fade) or "fade" (through black)
TRANSITIONS = {4: "dissolve", 5: "fade"}


def main():
    out = ROOT / "media" / "demo.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-preset", "medium", "-g", str(FPS), "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    frames = []
    for _, fn, sec in SCENES:
        frames.append(list(fn(sec * FPS)))
    written = 0
    truth = []
    for i, scene in enumerate(frames):
        kind = TRANSITIONS.get(i, "cut")
        if i > 0 and kind == "dissolve":
            prev_tail = frames[i - 1][-FPS:]
            for k in range(FPS):
                a = (k + 1) / (FPS + 1)
                proc.stdin.write(cv2.addWeighted(prev_tail[k], 1 - a, scene[k], a, 0).tobytes())
            start = written + FPS / 2  # boundary in the middle of the dissolve
            body = scene[FPS:]
        elif i > 0 and kind == "fade":
            start = written
            body = [(f * min(1.0, (k + 1) / 12)).astype(np.uint8) if k < 12 else f for k, f in enumerate(scene)]
        else:
            start = written
            body = scene
        if i > 0:
            truth.append({"time": round(start / FPS, 2), "type": kind, "into": SCENES[i][0]})
        if i + 1 < len(frames) and TRANSITIONS.get(i + 1) == "dissolve":
            body = body[:-FPS]
        if i + 1 < len(frames) and TRANSITIONS.get(i + 1) == "fade":
            body = [(f * min(1.0, (len(body) - k) / 12)).astype(np.uint8) if len(body) - k <= 12 else f for k, f in enumerate(body)]
        for f in body:
            proc.stdin.write(f.tobytes())
        written += len(body) + (FPS if kind == "dissolve" and i > 0 else 0)
    proc.stdin.close()
    proc.wait()
    (ROOT / "media" / "demo_truth.json").write_text(json.dumps({"duration": round(written / FPS, 2), "fps": FPS, "boundaries": truth}, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({written / FPS:.1f} s, {out.stat().st_size / 1e6:.1f} MB) and media/demo_truth.json")


if __name__ == "__main__":
    main()
