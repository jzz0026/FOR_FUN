import os, sys, json, re, math, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

sys.path.insert(0, "/project/small_grains/Jinlie.Zhou/for_fun/youtube_video")
import build_video as bv

W, H, FPS = bv.W, bv.H, bv.FPS
BASE, VDIR, ADIR, AUDDIR = bv.BASE, bv.VDIR, bv.ADIR, bv.AUDDIR
CLIPDIR = os.path.join(VDIR, "clips")
os.makedirs(CLIPDIR, exist_ok=True)

def precompute_overlay_roi(ov_rgba):
    # Only blend pixels where alpha > 0 (saves ~85% of per-frame compute!)
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

def render_one_scene(args):
    sid, chapter, seg_dur, subs, img_path, loader, kb_mode, card_lines = args
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clip_mp4 = os.path.join(CLIPDIR, f"{sid}.mp4")

    base_img = bv.load_cover(img_path) if loader == "cover" else bv.load_contain_parchment(img_path)
    n_frames = max(1, int(round(seg_dur * FPS)))
    fade_frames = int(0.4 * FPS)

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-threads", "2",
        clip_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Precompute static top (chapter + card) and per-subtitle bottom ROIs separately!
    top_roi = precompute_overlay_roi(bv.render_overlay_layer(chapter, card_lines, ""))
    sub_cache = {}

    for fi in range(n_frames):
        local_t = fi / FPS
        prog = fi / max(1, n_frames - 1)
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
        return (int(mm.group(1))*3600 + int(mm.group(2))*60 + float(mm.group(3))) if mm else 10.0

    scene_cfg = {
        "s01": (os.path.join(ADIR, "scene01_watertown_mist.jpg"), "cover", "zoom_in",
                ["从一本家谱出发", "青浦西南 · 古称“蒸里”（大蒸 / 小蒸）", "汉濮阳王蒸土传说 · 旁有良渚金山坟遗址"]),
        "s02": (os.path.join(ADIR, "scene02_ancient_books.jpg"), "cover", "zoom_out",
                ["《田家五行》署名悬案", "明嘉靖九年（1530）顾应祥刻本", "卷端题“田舍子娄元礼鹤天述”——史志无载"]),
        "s03": (os.path.join(ADIR, "scene02_ancient_books.jpg"), "cover", "pan_right",
                ["早期明代方志铁证", "《正德松江府志》（1512）：陆泳字伯翔，隐居大蒸", "钱惟善《田家五行序》（1365）：华亭之士陆伯翔氏"]),
        "s04": (os.path.join(ADIR, "scene02_ancient_books.jpg"), "cover", "zoom_in",
                ["2023年版本学破案（董晨）", "陆泳（元末）原著《田家五行》", "→ 娄元礼（明初）校定并合编《纪历撮要》", "→ 1530年顾应祥刻本，后世误署娄氏"]),
        "s05": (os.path.join(ADIR, "scene03_caoyuan_garden.jpg"), "cover", "pan_right",
                ["岳父：曹知白（1272—1355）", "字又玄，号云西，私谥贞素先生，人称“多田翁”", "小蒸曹园“园池八十余所，广袤数里”"]),
        "s06": (os.path.join(ADIR, "scene04_suishengting_gathering.jpg"), "cover", "zoom_in",
                ["1336年冬 · 遂生亭八人围炉联句", "陆泳：百两辞重聘，千金戒垂堂", "陆伯弘：躬勤乐耕钓，志俭忘膏粱"]),
        "s07": (os.path.join(ADIR, "scene05_lantern_night_visit.jpg"), "cover", "zoom_in",
                ["1356年正月十六落灯夜", "连襟鲁得之（火鲁忽达）：西域康里贵族", "“弱冠为馆甥于小蒸曹氏”，官至利用监大卿"]),
        "s08": (os.path.join(ADIR, "scene03_caoyuan_garden.jpg"), "cover", "pan_left",
                ["乱世义气与元夕祝寿", "1356年曹炳辟室“一枝安”馆避兵老友钱南金", "邵亨贞寿陆泳：祖砚传家终遗远，农书成帙更芟繁"]),
        "s09": (os.path.join(BASE, "images/xiaozheng_historical_map.png"), "contain", "map_dazheng",
                ["寻迹一：古大蒸镇与庄严寺", "大浜村灵观庙 = 古大蒸镇陆家宅 / 钓湾", "东厍村庄严寺 = 元大德濮阳王庙 / 旁金山坟"]),
        "s10": (os.path.join(BASE, "images/xiaozheng_historical_map.png"), "contain", "map_qiaowan",
                ["寻迹二：无尽庵以东数里", "乔湾里 = 邵湾里无尽庵（邵桂子/邵亨贞故居）", "向东：庄前港（清娄县一保49图）与芦花利涉桥"]),
        "s11": (os.path.join(BASE, "6554.jpg"), "contain", "zoom_in",
                ["清代三泖木刻古图对照", "北傍圆泖（明星/庄浜望泖塔、沈巷）", "东入大泖（小蒸街市河东出），南临大蒸塘"]),
        "s12": (os.path.join(ADIR, "scene06_farewell_jinling.jpg"), "cover", "zoom_out",
                ["1370年洪武庚戌 · 盛筵散尽", "曹园已“荡为丘陇”，遂生亭八人仅存陆、邵二人", "送陆伯翔赴金陵：回首故园如隔世，身闲不必问生涯"]),
        "s13": (os.path.join(ADIR, "scene01_watertown_mist.jpg"), "cover", "zoom_out",
                ["主要参考一手文献", "《田家五行》《正德松江府志》《蚁术诗选》", "《野处集》《玩斋集》《赵氏铁网珊瑚》《蒸里志略》"]),
    }

    jobs = []
    for i, m in enumerate(meta):
        fast_mp3 = os.path.join(AUDDIR, f"{m['id']}_fast.mp3")
        dur = probe_dur(fast_mp3)
        gap = 0.5 if i < len(meta) - 1 else 0.0
        subs = bv.split_subtitles(m["text"], dur)
        img_path, loader, kb_mode, card_lines = scene_cfg[m["id"]]
        jobs.append((m["id"], m["chapter"], dur + gap, subs, img_path, loader, kb_mode, card_lines))

    print(f"Launching 13 parallel scene renders (8 workers)...", flush=True)
    done_clips = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(render_one_scene, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            sid, cpath, nf = fut.result()
            done_clips[sid] = cpath
            print(f"  Done {sid} ({nf} frames)", flush=True)

    # Concatenate video clips and mux with full_narration.mp3
    vconcat = os.path.join(CLIPDIR, "vconcat.txt")
    with open(vconcat, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(f"file '{done_clips[m['id']]}'\n")

    out_mp4 = os.path.join(VDIR, "从一本家谱出发_陆泳与田家五行_1080p.mp4")
    combined_mp3 = os.path.join(AUDDIR, "full_narration.mp3")
    subprocess.run([
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", vconcat,
        "-i", combined_mp3,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        out_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"ALL DONE! {out_mp4} ({os.path.getsize(out_mp4)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
