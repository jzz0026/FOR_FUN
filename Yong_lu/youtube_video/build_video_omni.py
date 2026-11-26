#!/usr/bin/env python3
"""Render full 1080p dynamic documentary using Gemini Omni 1.2 Flash 1080p clips + maps + narration."""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import imageio_ffmpeg

BASE = "/project/small_grains/Jinlie.Zhou/for_fun/Yong_lu"
VDIR = os.path.join(BASE, "youtube_video")
ADIR = os.path.join(VDIR, "assets")
AUDDIR = os.path.join(VDIR, "audio")
ODIR = os.path.join(VDIR, "omni_clips_1080p")
CLIPDIR = os.path.join(VDIR, "clips_omni")
os.makedirs(CLIPDIR, exist_ok=True)

sys.path.insert(0, VDIR)
import build_video as bv

W, H, FPS = bv.W, bv.H, bv.FPS


def precompute_overlay_roi(ov_rgba):
    ys, xs = np.where(ov_rgba[:, :, 3] > 0)
    if len(ys) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub_rgba = ov_rgba[y0:y1, x0:x1]
    a = (sub_rgba[:, :, 3:4].astype(np.float32) / 255.0)
    inv_a = 1.0 - a
    bgr_pre = sub_rgba[:, :, [2, 1, 0]].astype(np.float32) * a
    return (y0, y1, x0, x1, inv_a, bgr_pre)


def apply_roi(frame, roi):
    if roi is None:
        return frame
    y0, y1, x0, x1, inv_a, bgr_pre = roi
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (patch * inv_a + bgr_pre).astype(np.uint8)
    return frame


def load_omni_frames(mp4_path):
    """Load all frames from an 8s 1080p Omni clip into memory (192 frames ~1.1GB max -> resize/verify 1920x1080)."""
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to read frames from {mp4_path}")
    return frames


def get_seamless_loop_frame(frames, fi, xfade=18):
    """Seamless crossfade loop through an 8s video clip (avoids awkward reverse motion while eliminating hard cuts)."""
    n = len(frames)
    if n <= xfade * 2:
        return frames[fi % n].copy()
    period = n - xfade
    pos = fi % period
    if fi >= period and pos < xfade:
        # Crossfade tail [period + pos] into head [pos]
        alpha = pos / float(xfade)
        f_head = frames[pos]
        f_tail = frames[period + pos]
        return cv2.addWeighted(f_tail, 1.0 - alpha, f_head, alpha, 0)
    return frames[pos].copy()


def render_one_scene(args):
    sid, chapter, seg_dur, subs, src_path, src_type, kb_mode, card_lines = args
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clip_mp4 = os.path.join(CLIPDIR, f"{sid}.mp4")

    if src_type == "omni_video":
        vid_frames = load_omni_frames(src_path)
        base_img = None
    else:
        vid_frames = None
        base_img = bv.load_cover(src_path) if src_type == "cover" else bv.load_contain_parchment(src_path)

    n_frames = max(1, int(round(seg_dur * FPS)))
    fade_frames = int(0.45 * FPS)

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-threads", "2",
        clip_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    top_roi = precompute_overlay_roi(bv.render_overlay_layer(chapter, card_lines, ""))
    sub_cache = {}

    for fi in range(n_frames):
        local_t = fi / FPS
        prog = fi / max(1, n_frames - 1)
        if vid_frames is not None:
            frame = get_seamless_loop_frame(vid_frames, fi, xfade=18)
        else:
            frame = bv.ken_burns_crop(base_img, prog, mode=kb_mode)

        apply_roi(frame, top_roi)

        cur_sub = ""
        for st, et, txt in subs:
            if st <= local_t < et:
                cur_sub = txt
                break
        if cur_sub:
            if cur_sub not in sub_cache:
                sub_cache[cur_sub] = precompute_overlay_roi(bv.render_overlay_layer("", None, cur_sub))
            apply_roi(frame, sub_cache[cur_sub])

        if fi < fade_frames:
            alpha = fi / fade_frames
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        elif fi >= n_frames - fade_frames:
            alpha = (n_frames - 1 - fi) / fade_frames
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{sid} ffmpeg exited with code {ret}")
    return sid, clip_mp4, n_frames


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with open(os.path.join(AUDDIR, "tts_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    def probe_dur(p):
        r = subprocess.run([ffmpeg, "-i", p], capture_output=True, text=True)
        mm = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        return (int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + float(mm.group(3))) if mm else 10.0

    def omni(name):
        return os.path.join(ODIR, f"{name}_1080p.mp4")

    scene_cfg = {
        "s01": (omni("scene01_watertown_mist"), "omni_video", None,
                ["从一本家谱出发", "青浦西南 · 古称“蒸里”（大蒸 / 小蒸）", "汉濮阳王蒸土传说 · 旁有良渚金山坟遗址"]),
        "s02": (omni("scene02_ancient_books"), "omni_video", None,
                ["《田家五行》署名悬案", "明嘉靖九年（1530）顾应祥刻本", "卷端题“田舍子娄元礼鹤天述”——史志无载"]),
        "s03": (omni("scene02_ancient_books"), "omni_video", None,
                ["早期明代方志铁证", "《正德松江府志》（1512）：陆泳字伯翔，隐居大蒸", "钱惟善《田家五行序》（1365）：华亭之士陆伯翔氏"]),
        "s04": (omni("scene02_ancient_books"), "omni_video", None,
                ["2023年版本学破案（董晨）", "陆泳（元末）原著《田家五行》", "→ 娄元礼（明初）校定并合编《纪历撮要》", "→ 1530年顾应祥刻本，后世误署娄氏"]),
        "s05": (omni("scene03_caoyuan_garden"), "omni_video", None,
                ["岳父：曹知白（1272—1355）", "字又玄，号云西，私谥贞素先生，人称“多田翁”", "小蒸曹园“园池八十余所，广袤数里”"]),
        "s06": (omni("scene04_suishengting_gathering"), "omni_video", None,
                ["1336年冬 · 遂生亭八人围炉联句", "陆泳：百两辞重聘，千金戒垂堂", "陆伯弘：躬勤乐耕钓，志俭忘膏粱"]),
        "s07": (omni("scene05_lantern_night_visit"), "omni_video", None,
                ["1356年正月十六落灯夜", "连襟鲁得之（火鲁忽达）：西域康里贵族", "“弱冠为馆甥于小蒸曹氏”，官至利用监大卿"]),
        "s08": (omni("scene03_caoyuan_garden"), "omni_video", None,
                ["乱世义气与元夕祝寿", "1356年曹炳辟室“一枝安”馆避兵老友钱南金", "邵亨贞寿陆泳：祖砚传家终遗远，农书成帙更芟繁"]),
        "s09": (os.path.join(BASE, "images/xiaozheng_historical_map.png"), "contain", "map_dazheng",
                ["寻迹一：古大蒸镇与庄严寺", "大浜村灵观庙 = 古大蒸镇陆家宅 / 钓湾", "东厍村庄严寺 = 元大德濮阳王庙 / 旁金山坟"]),
        "s10": (os.path.join(BASE, "images/xiaozheng_historical_map.png"), "contain", "map_qiaowan",
                ["寻迹二：无尽庵以东数里", "乔湾里 = 邵湾里无尽庵（邵桂子/邵亨贞故居）", "向东：庄前港（清娄县一保49图）与芦花利涉桥"]),
        "s11": (os.path.join(BASE, "6554.jpg"), "contain", "zoom_in",
                ["清代三泖木刻古图对照", "北傍圆泖（明星/庄浜望泖塔、沈巷）", "东入大泖（小蒸街市河东出），南临大蒸塘"]),
        "s12": (omni("scene06_farewell_jinling"), "omni_video", None,
                ["1370年洪武庚戌 · 盛筵散尽", "曹园已“荡为丘陇”，遂生亭八人仅存陆、邵二人", "送陆伯翔赴金陵：回首故园如隔世，身闲不必问生涯"]),
        "s13": (omni("scene01_watertown_mist"), "omni_video", None,
                ["主要参考一手文献", "《田家五行》《正德松江府志》《蚁术诗选》", "《野处集》《玩斋集》《赵氏铁网珊瑚》《蒸里志略》"]),
    }

    # Verify all required files exist
    for sid, (src, stype, _, _) in scene_cfg.items():
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing {sid} source: {src}")

    jobs = []
    for i, m in enumerate(meta):
        fast_mp3 = os.path.join(AUDDIR, f"{m['id']}_fast.mp3")
        dur = probe_dur(fast_mp3)
        gap = 0.5 if i < len(meta) - 1 else 0.0
        subs = bv.split_subtitles(m["text"], dur)
        src_path, src_type, kb_mode, card_lines = scene_cfg[m["id"]]
        jobs.append((m["id"], m["chapter"], dur + gap, subs, src_path, src_type, kb_mode, card_lines))

    workers = min(13, max(4, (os.cpu_count() or 8) // 2))
    print(f"Launching 13 parallel dynamic scene renders ({workers} workers)...", flush=True)
    done_clips = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_one_scene, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            sid, cpath, nf = fut.result()
            done_clips[sid] = cpath
            print(f"  Done {sid} ({nf} frames)", flush=True)

    vconcat = os.path.join(CLIPDIR, "vconcat.txt")
    with open(vconcat, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(f"file '{done_clips[m['id']]}'\n")

    out_mp4 = os.path.join(VDIR, "从一本家谱出发_陆泳与田家五行_1080p_动态版.mp4")
    combined_mp3 = os.path.join(AUDDIR, "full_narration.mp3")
    subprocess.run([
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", vconcat,
        "-i", combined_mp3,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        out_mp4,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"ALL DONE! {out_mp4} ({os.path.getsize(out_mp4)/(1024*1024):.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
