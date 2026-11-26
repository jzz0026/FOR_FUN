import os, sys, json, re, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

BASE = "/project/small_grains/Jinlie.Zhou/for_fun/Yong_lu"
VDIR = os.path.join(BASE, "youtube_video")
ADIR = os.path.join(VDIR, "assets")
AUDDIR = os.path.join(VDIR, "audio")

FONT_CJK = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
FONT_LAT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
FONT_LAT_BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"

W, H = 1920, 1080
FPS = 24

def is_cjk(ch):
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF) or (0xF900 <= o <= 0xFAFF) or (0xFF00 <= o <= 0xFFEF) or ch in "《》（）、。，；：！？"

def make_font(sz, bold=False):
    fl = ImageFont.truetype(FONT_LAT_BOLD if bold else FONT_LAT, sz)
    fc = ImageFont.truetype(FONT_CJK, sz)
    return (fl, fc)

def measure_mixed(text, fonts):
    fl, fc = fonts
    w = 0.0
    max_h = 0
    for ch in text:
        f = fc if is_cjk(ch) else fl
        w += f.getlength(ch)
        bb = f.getbbox(ch)
        if bb:
            max_h = max(max_h, bb[3] - bb[1])
    return int(math.ceil(w)), max_h

def draw_mixed(draw, x, y, text, fonts, fill, stroke_fill=None, stroke_w=0):
    fl, fc = fonts
    cx = float(x)
    for ch in text:
        f = fc if is_cjk(ch) else fl
        dy = 3 if f is fl else 0
        if stroke_fill and stroke_w > 0:
            for ox in range(-stroke_w, stroke_w + 1):
                for oy in range(-stroke_w, stroke_w + 1):
                    if ox != 0 or oy != 0:
                        draw.text((cx + ox, y + dy + oy), ch, font=f, fill=stroke_fill)
        draw.text((cx, y + dy), ch, font=f, fill=fill)
        cx += f.getlength(ch)

def generate_thumbnail():
    bg_path = os.path.join(ADIR, "thumbnail_bg.jpg")
    out_path = os.path.join(VDIR, "youtube_thumbnail.jpg")
    im = Image.open(bg_path).convert("RGBA").resize((1280, 720), Image.LANCZOS)
    ov = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    # Soft dark gradient on upper-left for crisp readability
    for y in range(440):
        alpha = int(175 * (1.0 - y / 440.0))
        od.line([(0, y), (1280, y)], fill=(12, 18, 26, alpha))
    # Crimson badge
    od.rounded_rectangle([48, 38, 378, 86], radius=8, fill=(176, 36, 32, 235))
    im = Image.alpha_composite(im, ov).convert("RGB")
    draw = ImageDraw.Draw(im)

    f_badge = make_font(24, bold=True)
    f_big = make_font(64, bold=True)
    f_gold = make_font(54, bold=True)
    f_foot = make_font(24, bold=False)

    draw_mixed(draw, 66, 48, "元末江南寻根 · 古籍破案", f_badge, (255, 250, 240))
    draw_mixed(draw, 48, 108, "从一本家谱出发", f_big, (255, 255, 255), stroke_fill=(15, 20, 28), stroke_w=3)
    draw_mixed(draw, 48, 194, "我在元末史书里找到", f_gold, (252, 211, 77), stroke_fill=(20, 15, 10), stroke_w=3)
    draw_mixed(draw, 48, 268, "670年前种地写书的先祖", f_gold, (252, 211, 77), stroke_fill=(20, 15, 10), stroke_w=3)

    # Bottom right tag
    tag = "陆泳 · 《田家五行》 · 小蒸曹园雅集"
    tw, th = measure_mixed(tag, f_foot)
    draw.rectangle([1280 - tw - 60, 650, 1280 - 24, 696], fill=(18, 24, 34))
    draw_mixed(draw, 1280 - tw - 42, 660, tag, f_foot, (240, 235, 220))

    im.save(out_path, quality=95)
    print("Generated thumbnail:", out_path)

def split_subtitles(text, total_dur):
    # Split by Chinese punctuation into readable chunks (10-22 chars)
    parts = re.split(r"([。！？；：，—]+)", text)
    raw_chunks = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if re.match(r"^[。！？；：，—]+$", p):
            buf += p
            if len(buf) >= 12 or p[0] in "。！？；":
                raw_chunks.append(buf.strip("，；： "))
                buf = ""
        else:
            if len(buf) + len(p) > 26 and buf:
                raw_chunks.append(buf.strip("，；： "))
                buf = p
            else:
                buf += p
    if buf.strip():
        raw_chunks.append(buf.strip("，；： "))

    # Further split any long chunk > 26 chars
    chunks = []
    for c in raw_chunks:
        c = c.strip()
        if not c:
            continue
        while len(c) > 26:
            chunks.append(c[:22])
            c = c[22:]
        if c:
            chunks.append(c)

    weights = [max(len(c), 4) for c in chunks]
    tot_w = sum(weights)
    timed = []
    t = 0.0
    for c, w in zip(chunks, weights):
        d = total_dur * (w / tot_w)
        timed.append((t, t + d, c))
        t += d
    return timed

def fmt_srt(sec):
    ms = int(round(sec * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def load_cover(path):
    im = Image.open(path).convert("RGB")
    # Ensure at least 2200x1238 for smooth Ken Burns without upscaling blur
    iw, ih = im.size
    target_ratio = W / H
    cur_ratio = iw / ih
    if cur_ratio > target_ratio:
        # wider than 16:9 -> fit height to 1240
        nh = 1240
        nw = int(iw * nh / ih)
    else:
        nw = 2200
        nh = int(ih * nw / iw)
    im = im.resize((nw, nh), Image.LANCZOS)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def load_contain_parchment(path):
    # For maps or vertical woodblock prints, place nicely on aged parchment canvas (2200x1240)
    canvas = Image.new("RGB", (2200, 1240), (242, 234, 216))
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    scale = min(2060 / iw, 1140 / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    ox, oy = (2200 - nw) // 2, (1240 - nh) // 2
    canvas.paste(im, (ox, oy))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def render_overlay_layer(chapter, card_lines=None, sub_text=""):
    # Create RGBA overlay (1920x1080) with PIL
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)

    f_chap = make_font(24, bold=True)
    f_card_t = make_font(30, bold=True)
    f_card_b = make_font(24, bold=False)
    f_sub = make_font(36, bold=True)

    # Top-left chapter badge
    if chapter:
        cw, ch = measure_mixed(chapter, f_chap)
        draw.rounded_rectangle([40, 32, 40 + cw + 36, 32 + 48], radius=8, fill=(18, 24, 34, 185), outline=(212, 175, 55, 200), width=2)
        draw_mixed(draw, 58, 43, chapter, f_chap, (248, 244, 230))

    # Optional side quote / evidence card (top-right)
    if card_lines:
        title_line = card_lines[0]
        body_lines = card_lines[1:]
        max_w = measure_mixed(title_line, f_card_t)[0]
        for bl in body_lines:
            max_w = max(max_w, measure_mixed(bl, f_card_b)[0])
        bw = min(max_w + 56, 780)
        bh = 58 + len(body_lines) * 38 + 20
        bx = W - bw - 44
        by = 36
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=10, fill=(248, 242, 226, 228), outline=(145, 42, 34, 230), width=3)
        draw_mixed(draw, bx + 26, by + 16, title_line, f_card_t, (150, 35, 30))
        cy = by + 62
        for bl in body_lines:
            draw_mixed(draw, bx + 26, cy, bl, f_card_b, (38, 32, 28))
            cy += 38

    # Bottom subtitle bar
    if sub_text:
        sw, sh = measure_mixed(sub_text, f_sub)
        bar_w = sw + 56
        bar_x = (W - bar_w) // 2
        bar_y = H - 106
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 64], radius=10, fill=(12, 16, 22, 195))
        draw_mixed(draw, bar_x + 28, bar_y + 12, sub_text, f_sub, (255, 255, 255), stroke_fill=(0, 0, 0), stroke_w=1)

    return np.array(ov) # RGBA

def blend_rgba_onto_bgr(frame_bgr, ov_rgba):
    alpha = ov_rgba[:, :, 3:4].astype(np.float32) / 255.0
    ov_bgr = ov_rgba[:, :, [2, 1, 0]].astype(np.float32)
    out = frame_bgr.astype(np.float32) * (1.0 - alpha) + ov_bgr * alpha
    return out.astype(np.uint8)

def ken_burns_crop(img_bgr, prog, mode="zoom_in", focus=(0.5, 0.5)):
    ih, iw = img_bgr.shape[:2]
    # Ease in-out
    t = prog
    if mode == "zoom_in":
        z = 1.0 + 0.12 * t
        fx, fy = focus
    elif mode == "zoom_out":
        z = 1.12 - 0.12 * t
        fx, fy = focus
    elif mode == "pan_right":
        z = 1.10
        fx = 0.44 + 0.12 * t
        fy = focus[1]
    elif mode == "pan_left":
        z = 1.10
        fx = 0.56 - 0.12 * t
        fy = focus[1]
    elif mode == "map_dazheng":
        # Zoom into bottom-left [1] 大浜村 & [2] 庄严寺
        z = 1.05 + 0.45 * t
        fx = 0.50 - 0.24 * t
        fy = 0.50 + 0.30 * t
    elif mode == "map_qiaowan":
        # Pan/zoom into center-right [3] 乔湾里 -> [4] 庄前港 -> [5] 芦花蒸南
        z = 1.25 + 0.30 * t
        fx = 0.46 + 0.18 * t
        fy = 0.65 - 0.04 * t
    else:
        z = 1.0 + 0.08 * t
        fx, fy = focus

    # Target aspect ratio 16:9 inside img_bgr
    base_h = min(ih, iw / (W / H))
    base_w = base_h * (W / H)
    cw = base_w / z
    ch = base_h / z
    cx = max(cw / 2, min(iw - cw / 2, fx * iw))
    cy = max(ch / 2, min(ih - ch / 2, fy * ih))
    x0 = int(round(cx - cw / 2))
    y0 = int(round(cy - ch / 2))
    x1 = min(iw, int(round(x0 + cw)))
    y1 = min(ih, int(round(y0 + ch)))
    crop = img_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

def main():
    generate_thumbnail()

    with open(os.path.join(AUDDIR, "tts_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # Scene visual config: (image_path, loader, kb_mode, card_lines)
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

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # 1. Speed up gTTS slightly (1.22x) for natural YouTube documentary pacing & concatenate with 0.5s gap
    concat_list = os.path.join(AUDDIR, "concat.txt")
    silence_mp3 = os.path.join(AUDDIR, "silence_05.mp3")
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.5", "-q:a", "4", silence_mp3],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def probe_dur(p):
        r = subprocess.run([ffmpeg, "-i", p], capture_output=True, text=True)
        mm = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r.stderr)
        return (int(mm.group(1))*3600 + int(mm.group(2))*60 + float(mm.group(3))) if mm else 10.0

    with open(concat_list, "w", encoding="utf-8") as f:
        for i, m in enumerate(meta):
            fast_mp3 = os.path.join(AUDDIR, f"{m['id']}_fast.mp3")
            subprocess.run([ffmpeg, "-y", "-i", m["mp3"], "-filter:a", "atempo=1.22", "-q:a", "3", fast_mp3],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            m["mp3_fast"] = fast_mp3
            m["duration"] = probe_dur(fast_mp3)
            f.write(f"file '{fast_mp3}'\n")
            if i < len(meta) - 1:
                f.write(f"file '{silence_mp3}'\n")

    combined_mp3 = os.path.join(AUDDIR, "full_narration.mp3")
    subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", combined_mp3],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Build SRT file & timeline
    srt_path = os.path.join(VDIR, "从一本家谱出发_陆泳与田家五行.srt")
    srt_entries = []
    timeline = []
    cur_t = 0.0
    for i, m in enumerate(meta):
        dur = m["duration"]
        gap = 0.5 if i < len(meta) - 1 else 0.0
        subs = split_subtitles(m["text"], dur)
        for st, et, txt in subs:
            srt_entries.append((cur_t + st, cur_t + et, txt))
        timeline.append((m["id"], m["chapter"], cur_t, cur_t + dur + gap, subs))
        cur_t += dur + gap

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (st, et, txt) in enumerate(srt_entries, 1):
            f.write(f"{idx}\n{fmt_srt(st)} --> {fmt_srt(et)}\n{txt}\n\n")
    print(f"Generated SRT ({len(srt_entries)} lines), total video duration: {cur_t:.1f}s")

    # 3. Render 1080p MP4 streaming directly into ffmpeg
    out_mp4 = os.path.join(VDIR, "从一本家谱出发_陆泳与田家五行_1080p.mp4")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", combined_mp3,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        out_mp4
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    img_cache = {}
    def get_img(path, loader):
        key = (path, loader)
        if key not in img_cache:
            img_cache[key] = load_cover(path) if loader == "cover" else load_contain_parchment(path)
        return img_cache[key]

    for sid, chapter, t0, t1, subs in timeline:
        img_path, loader, kb_mode, card_lines = scene_cfg[sid]
        base_img = get_img(img_path, loader)
        seg_dur = t1 - t0
        n_frames = max(1, int(round(seg_dur * FPS)))
        fade_frames = int(0.4 * FPS)

        # Pre-render overlay cache per unique subtitle string in this scene!
        ov_cache = {}
        print(f"Rendering {sid} ({seg_dur:.1f}s, {n_frames} frames)...", flush=True)
        for fi in range(n_frames):
            local_t = fi / FPS
            prog = fi / max(1, n_frames - 1)
            frame = ken_burns_crop(base_img, prog, mode=kb_mode)

            # Find active subtitle
            cur_sub = ""
            for st, et, txt in subs:
                if st <= local_t < et:
                    cur_sub = txt
                    break

            if cur_sub not in ov_cache:
                ov_cache[cur_sub] = render_overlay_layer(chapter, card_lines, cur_sub)
            frame = blend_rgba_onto_bgr(frame, ov_cache[cur_sub])

            # Smooth fade-in / fade-out
            if fi < fade_frames:
                alpha = fi / fade_frames
                frame = (frame.astype(np.float32) * alpha).astype(np.uint8)
            elif fi >= n_frames - fade_frames:
                alpha = (n_frames - 1 - fi) / fade_frames
                frame = (frame.astype(np.float32) * alpha).astype(np.uint8)

            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", errors="ignore")
    ret = proc.wait()
    if ret != 0:
        print("FFmpeg error:", err[-1500:])
        sys.exit(1)
    print(f"SUCCESS! Rendered {out_mp4} ({os.path.getsize(out_mp4)/(1024*1024):.1f} MB)")

if __name__ == "__main__":
    main()
