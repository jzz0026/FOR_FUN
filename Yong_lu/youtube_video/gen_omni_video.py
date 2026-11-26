#!/usr/bin/env python3
"""Generate dynamic video clips from still illustrations via Gemini Omni 1.2 Flash (Interactions API)."""
import argparse
import base64
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from google import genai

ENV_PATH = os.path.expanduser("~/.config/video-gen/.env")
LEDGER_PATH = os.path.expanduser("~/.config/video-gen/spend_ledger.json")

# Official Gemini Omni 1.2 Flash pricing: $12.00 / 1M video output tokens
# (~$0.023/s at 360p, ~$0.070/s at 720p, ~$0.104/s at 1080p)
VIDEO_OUT_PRICE_PER_M = 12.00
TEXT_IN_PRICE_PER_M = 0.50


def load_api_key():
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"GEMINI_API_KEY not found in {ENV_PATH}")


def record_ledger(entry):
    data = {"entries": [], "total_usd": 0.0}
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, encoding="utf-8") as f:
            data = json.load(f)
    data["entries"].append(entry)
    data["total_usd"] = round(sum(e.get("cost_usd", 0.0) for e in data["entries"]), 6)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data["total_usd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to reference image")
    ap.add_argument("--prompt", required=True, help="Motion/scene prompt")
    ap.add_argument("--out", required=True, help="Output MP4 path")
    ap.add_argument("--resolution", default="360p", choices=["360p", "720p", "1080p", "4k"])
    ap.add_argument("--duration", default="6s", help="Duration like 5s, 6s, 8s, 10s")
    ap.add_argument("--aspect", default="16:9", choices=["16:9", "9:16"])
    ap.add_argument("--model", default="gemini-omni-1.2-flash")
    args = ap.parse_args()

    api_key = load_api_key()
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    ext = os.path.splitext(args.image)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    print(f"Submitting {args.model} ({args.resolution}, {args.duration}, {args.aspect})...", flush=True)
    inter = client.interactions.create(
        model=args.model,
        input=[
            {"type": "image", "data": img_b64, "mime_type": mime},
            {"type": "text", "text": args.prompt},
        ],
        response_modalities=["video"],
        response_format={
            "type": "video",
            "resolution": args.resolution,
            "aspect_ratio": args.aspect,
            "duration": args.duration,
        },
        background=True,
    )
    print(f"Interaction ID: {inter.id}, initial status: {inter.status}", flush=True)

    t0 = time.time()
    while inter.status in ("in_progress", "queued", "pending", None):
        time.sleep(5)
        inter = client.interactions.get(inter.id)
        print(f"  [{int(time.time()-t0)}s] status={inter.status}", flush=True)
        if time.time() - t0 > 600:
            raise TimeoutError("Timed out waiting for video generation")

    if inter.status != "completed":
        print("FULL RESPONSE:", inter.model_dump_json(indent=2)[:2000], flush=True)
        raise RuntimeError(f"Generation did not complete: status={inter.status}")

    # Extract video from inter.output_video
    saved = False
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ov = getattr(inter, "output_video", None)
    if ov is not None:
        vdata = getattr(ov, "data", None)
        vuri = getattr(ov, "uri", None)
        if vdata:
            raw = base64.b64decode(vdata) if isinstance(vdata, str) else vdata
            with open(args.out, "wb") as f:
                f.write(raw)
            saved = True
        elif vuri:
            dl_url = vuri if "key=" in vuri else (f"{vuri}&key={api_key}" if "?" in vuri else f"{vuri}?key={api_key}")
            urllib.request.urlretrieve(dl_url, args.out)
            saved = True

    if not saved:
        print("DUMP:", inter.model_dump_json(indent=2)[:2000], flush=True)
        raise RuntimeError("No output_video found in completed interaction")

    # Compute cost from usage
    usage = inter.usage
    in_tok = getattr(usage, "total_input_tokens", 0) or 0
    out_tok = getattr(usage, "total_output_tokens", 0) or 0
    cost_usd = round((out_tok / 1_000_000.0) * VIDEO_OUT_PRICE_PER_M + (in_tok / 1_000_000.0) * TEXT_IN_PRICE_PER_M, 4)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_id": inter.id,
        "model": args.model,
        "resolution": args.resolution,
        "duration": args.duration,
        "out_file": args.out,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost_usd,
    }
    cum_usd = record_ledger(entry)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"SAVED: {args.out} ({size_mb:.2f} MB)")
    print(f"USAGE: input_tokens={in_tok}, output_tokens={out_tok}")
    print(f"COST_THIS_CALL_USD: ${cost_usd:.4f}")
    print(f"CUMULATIVE_SPEND_USD: ${cum_usd:.4f}")


if __name__ == "__main__":
    main()
