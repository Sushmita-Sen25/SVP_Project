"""Generate the demo audio corpus with neural TTS (edge-tts).

Creates
  audio/words/<word>/<voice>__<form>.mp3            pronunciation-assessment samples
  audio/gita/train/<verse>/<voice>__<config>.mp3    HMM training recitations
  audio/gita/test/<verse>/<voice>.mp3               held-out recitations

Usage:  pip install edge-tts
        python scripts/generate_audio.py
Existing files are skipped, so the script can be re-run safely.
"""
import asyncio
import json
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parent.parent
CONCURRENCY = 4


def vid(verse_id: str) -> str:
    return "v" + verse_id.replace(".", "_")


def tts_text(lines):
    text = " ".join(lines)
    # The avagraha is silent; the double danda is read like a full stop.
    return text.replace("ऽ", "").replace("॥", "।")


def build_jobs():
    words = json.loads((ROOT / "data/words.json").read_text(encoding="utf-8"))
    verses = json.loads((ROOT / "data/verses.json").read_text(encoding="utf-8"))
    jobs = []
    for w in words["words"]:
        for v in words["voices"]:
            for f in w["forms"]:
                out = ROOT / "audio/words" / w["id"] / f"{v['id']}__{f['id']}.mp3"
                jobs.append((f["deva"] + "।", v["id"], "-10%", "+0Hz", out))
    for verse in verses["verses"]:
        text = tts_text(verse["deva"])
        for voice in verses["train_voices"]:
            for cfg in verses["train_configs"]:
                out = ROOT / "audio/gita/train" / vid(verse["id"]) / f"{voice}__{cfg['id']}.mp3"
                jobs.append((text, voice, cfg["rate"], cfg["pitch"], out))
        for ts in verses["test_sets"]:
            cfg = ts["config"]
            out = ROOT / "audio/gita/test" / vid(verse["id"]) / f"{ts['voice']}.mp3"
            jobs.append((text, ts["voice"], cfg["rate"], cfg["pitch"], out))
    return jobs


async def synth(sem, job, stats):
    text, voice, rate, pitch, out = job
    if out.exists() and out.stat().st_size > 1000:
        stats["skipped"] += 1
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for attempt in range(4):
            try:
                await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(out))
                stats["made"] += 1
                return
            except Exception as exc:  # network hiccups / throttling
                await asyncio.sleep(2 + attempt * 3)
                last = exc
        stats["failed"] += 1
        print(f"FAILED {out.relative_to(ROOT)}: {last}", file=sys.stderr)


async def main():
    jobs = build_jobs()
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"made": 0, "skipped": 0, "failed": 0}
    await asyncio.gather(*(synth(sem, j, stats) for j in jobs))
    print(f"{len(jobs)} clips: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
