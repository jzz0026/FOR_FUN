#!/usr/bin/env python3
"""Generate all 6 Song-Yuan ink-and-wash dynamic clips (1080p, 8s) in parallel via Gemini Omni 1.2 Flash."""
import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from google import genai

BASE = "/project/small_grains/Jinlie.Zhou/for_fun/Yong_lu"
ADIR = os.path.join(BASE, "youtube_video", "assets")
ODIR = os.path.join(BASE, "youtube_video", "omni_clips_1080p")
os.makedirs(ODIR, exist_ok=True)

ENV_PATH = os.path.expanduser("~/.config/video-gen/.env")
LEDGER_PATH = os.path.expanduser("~/.config/video-gen/spend_ledger.json")
VIDEO_OUT_PRICE_PER_M = 12.00
TEXT_IN_PRICE_PER_M = 0.50

SCENES = [
    (
        "scene01_watertown_mist",
        "scene01_watertown_mist.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash painting of a misty Jiangnan watertown. Keep the exact composition, art style, arched stone bridge, houses, and warm parchment tones. Gentle morning mist drifts slowly across the distant hills, soft breeze sways the willow branches, delicate ripples shimmer on the canal water, and the small wooden boatman under the bridge rows slowly forward. Very slow poetic camera push-in, peaceful atmosphere, no text.",
    ),
    (
        "scene02_ancient_books",
        "scene02_ancient_books.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash still life of classical thread-bound books and scholar's desk. Keep the exact composition, warm parchment paper texture, ink painting style, and objects unchanged. Thin wisps of fragrant incense smoke curl gracefully upward in slow motion, warm candlelight flickers softly across the open pages, and a gentle breeze barely stirs the corner of a scroll. Very slow subtle camera drift, serene and scholarly, no text.",
    ),
    (
        "scene03_caoyuan_garden",
        "scene03_caoyuan_garden.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash painting of a grand Jiangnan classical garden with pavilions, bridges, and ponds. Keep the exact composition, architecture, and muted mineral ink tones. Soft mist drifts slowly among the distant trees and tiled roofs, gentle breeze sways bamboo and willow leaves, and calm ripples shimmer across the garden pond. Slow sweeping poetic camera glide, timeless and tranquil, no text.",
    ),
    (
        "scene04_suishengting_gathering",
        "scene04_suishengting_gathering.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash painting of eight scholars gathered around a warm brazier inside a lakeside pavilion on a snowy winter night. Keep the exact composition, faces, robes, pavilion, and ink style locked. Fine snowflakes drift slowly down across the dark winter sky and pine trees outside, warm orange firelight from the charcoal brazier and hanging silk lanterns flickers gently on the scholars, who subtly nod and converse peacefully. Very slow cinematic push-in, warm and atmospheric, no text.",
    ),
    (
        "scene05_lantern_night_visit",
        "scene05_lantern_night_visit.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash painting of an evening Lantern Festival visit in a Jiangnan courtyard. Keep the exact composition, figures, architecture, and ink-wash aesthetic. Warm glowing paper lanterns sway gently in the night breeze, soft golden light shimmers on the stone path and water, and light winter mist drifts slowly through the courtyard trees. Slow gentle camera movement, warm and intimate, no text.",
    ),
    (
        "scene06_farewell_jinling",
        "scene06_farewell_jinling.jpg",
        "Animate this exact Song-Yuan Chinese ink-and-wash painting of a quiet riverside farewell in Jiangnan mist. Keep the exact composition, figures, solitary boat, pine trees, and ink-wash tones. Soft river mist drifts slowly across the distant water, gentle ripples rock the waiting wooden boat, and a light breeze softly stirs the pine needles and the edges of the scholars' robes. Slow contemplative camera pull-back, poignant and serene, no text.",
    ),
]


def load_api_key():
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"GEMINI_API_KEY not found in {ENV_PATH}")


def generate_one(api_key, name, img_file, prompt):
    out_mp4 = os.path.join(ODIR, f"{name}_1080p.mp4")
    if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 500_000:
        print(f"[SKIP] {name}: already exists ({os.path.getsize(out_mp4)/(1024*1024):.2f} MB)", flush=True)
        return {"name": name, "out_file": out_mp4, "skipped": True, "cost_usd": 0.0}

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})
    img_path = os.path.join(ADIR, img_file)
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    print(f"[START] {name} (1080p, 8s)...", flush=True)
    inter = client.interactions.create(
        model="gemini-omni-1.2-flash",
        input=[
            {"type": "image", "data": img_b64, "mime_type": "image/jpeg"},
            {"type": "text", "text": prompt},
        ],
        response_modalities=["video"],
        response_format={
            "type": "video",
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "duration": "8s",
        },
        background=True,
    )
    t0 = time.time()
    while inter.status in ("in_progress", "queued", "pending", None):
        time.sleep(6)
        inter = client.interactions.get(inter.id)
        if time.time() - t0 > 600:
            raise TimeoutError(f"{name} timed out")

    if inter.status != "completed":
        raise RuntimeError(f"{name} failed with status={inter.status}")

    ov = getattr(inter, "output_video", None)
    if ov is None:
        raise RuntimeError(f"{name} missing output_video")
    if getattr(ov, "data", None):
        raw = base64.b64decode(ov.data) if isinstance(ov.data, str) else ov.data
        with open(out_mp4, "wb") as f:
            f.write(raw)
    elif getattr(ov, "uri", None):
        vuri = ov.uri
        dl = vuri if "key=" in vuri else (f"{vuri}&key={api_key}" if "?" in vuri else f"{vuri}?key={api_key}")
        urllib.request.urlretrieve(dl, out_mp4)
    else:
        raise RuntimeError(f"{name} output_video has no data or uri")

    usage = inter.usage
    in_tok = getattr(usage, "total_input_tokens", 0) or 0
    out_tok = getattr(usage, "total_output_tokens", 0) or 0
    thought_tok = getattr(usage, "total_thought_tokens", 0) or 0
    cost_usd = round((out_tok / 1_000_000.0) * VIDEO_OUT_PRICE_PER_M + (in_tok / 1_000_000.0) * TEXT_IN_PRICE_PER_M, 4)

    mb = os.path.getsize(out_mp4) / (1024 * 1024)
    print(f"[DONE] {name}: {mb:.2f} MB | {int(time.time()-t0)}s | out_tok={out_tok} | cost=${cost_usd:.4f}", flush=True)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_id": inter.id,
        "name": name,
        "model": "gemini-omni-1.2-flash",
        "resolution": "1080p",
        "duration": "8s",
        "out_file": out_mp4,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thought_tokens": thought_tok,
        "cost_usd": cost_usd,
        "skipped": False,
    }


def main():
    api_key = load_api_key()
    results = []
    # 3 concurrent workers to stay gentle on API rate limits while finishing in ~1-2 min
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(generate_one, api_key, n, f, p): n for n, f, p in SCENES}
        for fut in as_completed(futs):
            results.append(fut.result())

    # Update ledger
    data = {"entries": [], "total_usd": 0.0}
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, encoding="utf-8") as f:
            data = json.load(f)
    batch_cost = 0.0
    for r in results:
        if not r.get("skipped"):
            data["entries"].append(r)
            batch_cost += r["cost_usd"]
    data["total_usd"] = round(sum(e.get("cost_usd", 0.0) for e in data["entries"]), 4)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("=" * 60, flush=True)
    print(f"BATCH_COST_USD: ${batch_cost:.4f}", flush=True)
    print(f"CUMULATIVE_SPEND_USD: ${data['total_usd']:.4f}", flush=True)
    print("ALL 6 OMNI 1080p CLIPS READY!", flush=True)


if __name__ == "__main__":
    main()
