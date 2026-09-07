#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video-sheet-extract.py — 视频转乐谱核心提取引擎 v2.0
综合1.0版排版代码 + 1.1版画面优化代码

依赖: opencv-python numpy Pillow img2pdf, 系统需安装 ffmpeg
"""
import argparse
import os
import sys
import shutil
import subprocess
import time

import cv2
import img2pdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 零、进度打印工具（来自1.0版）
# ============================================================

class Progress:
    def __init__(self, total, label, every=1):
        self.total = max(1, total)
        self.label = label
        self.every = every
        self.start = time.time()
        self.count = 0

    def update(self, n=1, extra=""):
        self.count += n
        if self.count % self.every == 0 or self.count == self.total:
            elapsed = time.time() - self.start
            pct = self.count / self.total * 100
            rate = self.count / elapsed if elapsed > 0 else 0
            remain = (self.total - self.count) / rate if rate > 0 else 0
            print(f"  [{self.label}] {pct:5.1f}% "
                  f"({self.count}/{self.total}) 已用{elapsed:5.1f}s "
                  f"预计剩余{remain:5.1f}s {extra}")

    def done(self):
        elapsed = time.time() - self.start
        print(f"  [{self.label}] 完成，共耗时 {elapsed:.1f}s")


# ============================================================
# 一、画面优化工具（来自1.1版，画质最好）
# ============================================================

def min_channel_gray(bgr):
    """gray = min(B,G,R)：衡量像素离白色的距离。"""
    return bgr.min(axis=2).astype(np.uint8)


def normalize_frame_polarity(bgr):
    """自动识别并统一背景明暗极性。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if np.median(gray) < 127:
        return cv2.bitwise_not(bgr)
    return bgr


def highlight_bg_mask(bgr, bright_t=180, sat_t=20):
    """识别浅色高亮背景条。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    gray = min_channel_gray(bgr)
    return (gray > bright_t) & (s > sat_t)


def to_ink(bgr):
    """墨迹图（连续灰度，非二值）。"""
    ink = min_channel_gray(bgr).copy()
    ink[highlight_bg_mask(bgr)] = 255
    return ink


def combine(strips):
    """多帧中位数合成。"""
    h = min(s.shape[0] for s in strips)
    w = min(s.shape[1] for s in strips)
    stack = np.stack([s[:h, :w] for s in strips])
    return np.median(stack, axis=0).astype(np.uint8)


def flatten_bg(gray):
    """背景平整：高斯模糊估计背景 → 墨迹强度 → 线性拉伸。"""
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    ink = cv2.subtract(bg, gray).astype(np.float32)
    hi = np.percentile(ink, 99.5)
    if hi < 1:
        hi = 1
    out = 255 - np.clip(ink / hi * 255, 0, 255)
    return out.astype(np.uint8)


def enhance_strip(strip, target_w, p_lo=0.5, p_hi=99.5,
                  sharpen=1.2, sigma=1.0):
    """百分位对比度拉伸 + 轻微锐化 + Lanczos 放大（来自1.1版）。"""
    f = strip.astype(np.float32)
    lo, hi = np.percentile(f, p_lo), np.percentile(f, p_hi)
    if hi > lo:
        f = np.clip((f - lo) / (hi - lo) * 255, 0, 255)
    if sharpen > 0:
        blur = cv2.GaussianBlur(f, (0, 0), sigma)
        f = np.clip(f + sharpen * (f - blur), 0, 255)
    strip = f.astype(np.uint8)
    scale = target_w / strip.shape[1]
    return cv2.resize(strip, (target_w, max(1, int(strip.shape[0] * scale))),
                      interpolation=cv2.INTER_LANCZOS4)


# ============================================================
# 二、乐谱检测（来自1.0版，简单可靠）
# ============================================================

def white_stats(gray):
    p85 = float(np.percentile(gray, 85))
    white_t = max(160.0, p85 - 30.0)
    white_mask = gray >= white_t
    white_ratio = float(white_mask.mean())
    row_white = white_mask.mean(axis=1)
    row_cover = float((row_white > 0.60).mean())
    ink_ratio = float((gray < white_t - 70).mean())
    return white_ratio, row_cover, ink_ratio


def notation_ink_range(notation):
    if notation == 'tab':
        return 0.003, 0.32
    if notation == 'staff':
        return 0.002, 0.20
    if notation == 'both':
        return 0.003, 0.32
    return 0.002, 0.25


def has_sheet(gray, notation='auto', white_min=0.55, cover_min=0.60, debug=False):
    """判断该帧是否包含乐谱内容。"""
    ink_min, ink_max = notation_ink_range(notation)
    white_ratio, row_cover, ink_ratio = white_stats(gray)
    ok = (white_ratio >= white_min and row_cover >= cover_min
          and ink_min <= ink_ratio <= ink_max)
    if debug:
        print(f"    [debug] white={white_ratio:.3f} cover={row_cover:.3f} "
              f"ink={ink_ratio:.4f} -> {ok}")
    return ok


# ============================================================
# 三、相似度判定（来自1.0版）
# ============================================================

def ncc(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def diff_ratio(a, b):
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    _, ba = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bb = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((ba != bb).mean())


def is_duplicate(a, b, ncc_t, diff_max=0.06):
    return ncc(a, b) > ncc_t and diff_ratio(a, b) < diff_max


# ============================================================
# 四、Unicode路径写入（来自1.0版）
# ============================================================

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ret, buf = cv2.imencode(ext, img)
    if ret:
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
    return ret


# ============================================================
# 五、图像增强：对比度拉伸 + 裁边（来自1.0版）
# ============================================================

def contrast_stretch(gray, low_pct, high_pct):
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi <= lo:
        return gray
    out = (gray.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def unsharp_mask(gray, amount, sigma):
    if amount <= 0:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def dynamic_crop(gray, pad=24, ink_thresh=180):
    rows = np.where((gray < ink_thresh).any(axis=1))[0]
    if len(rows) < 10:
        return gray
    top = max(0, rows[0] - 8)
    bottom = min(gray.shape[0], rows[-1] + pad)
    return gray[top:bottom, :]


# ============================================================
# 六、三级去重流程（来自1.0版，快速可靠）
# ============================================================

def stage1_group_consecutive(candidates, group_t, diff_max):
    if not candidates:
        return []
    items = []
    cur_img = candidates[0][1]
    cur_rep = candidates[0][0]
    cur_members = [candidates[0][0]]
    n = len(candidates)
    for i in range(1, n):
        raw_idx, img = candidates[i]
        if not is_duplicate(cur_img, img, group_t, diff_max):
            items.append((cur_img, cur_rep, cur_members))
            cur_img, cur_rep, cur_members = img, raw_idx, [raw_idx]
        else:
            cur_members.append(raw_idx)
            cur_img, cur_rep = img, raw_idx
    items.append((cur_img, cur_rep, cur_members))
    return items


def stage2_merge_repeats(items, merge_t, diff_max):
    n = len(items)
    used = [False] * n
    result = []
    for i in range(n):
        if used[i]:
            continue
        img_i, rep_i, members_i = items[i]
        members_i = list(members_i)
        for j in range(i + 1, n):
            if used[j]:
                continue
            img_j, rep_j, members_j = items[j]
            if is_duplicate(img_i, img_j, merge_t, diff_max):
                used[j] = True
                members_i.extend(members_j)
        result.append((img_i, rep_i, members_i))
    return result


def stage3_final_dedup(items, dedup_t, diff_max):
    if not items:
        return []
    result = [items[0]]
    n = len(items)
    for i in range(1, n):
        img_i, rep_i, members_i = items[i]
        prev_img, prev_rep, prev_members = result[-1]
        if not is_duplicate(prev_img, img_i, dedup_t, diff_max):
            result.append((img_i, rep_i, list(members_i)))
        else:
            prev_members.extend(members_i)
            result[-1] = (prev_img, prev_rep, prev_members)
    return result


# ============================================================
# 七、自动裁剪区域检测（来自1.0版）
# ============================================================

def auto_detect_crop(cap, start_frame, end_frame, samples=30, pad_ratio=0.02):
    span = max(1, end_frame - start_frame)
    positions = [start_frame + int(span * i / max(1, samples - 1)) for i in range(samples)]
    freq_map = None
    h0 = w0 = None
    count = 0
    prog = Progress(len(positions), "自动裁剪检测", every=5)
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        prog.update()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if h0 is None:
            h0, w0 = gray.shape
            freq_map = np.zeros((h0, w0), dtype=np.float32)
        p85 = float(np.percentile(gray, 85))
        white_t = max(160.0, p85 - 30.0)
        freq_map += (gray >= white_t).astype(np.float32)
        count += 1
    prog.done()
    if count == 0 or freq_map is None:
        return None
    freq_map /= count
    mask = (freq_map > 0.5).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    if num <= 1:
        return None
    best_label, best_area = 0, 0
    for lbl in range(1, num):
        area = int((labels == lbl).sum())
        if area > best_area:
            best_area, best_label = area, lbl
    ys, xs = np.where(labels == best_label)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    region = freq_map[y0:y1+1, x0:x1+1]
    row_avg = region.mean(axis=1)
    col_avg = region.mean(axis=0)

    pos_rows = row_avg[row_avg > 0]
    pos_cols = col_avg[col_avg > 0]
    if len(pos_rows) > 0:
        edge_thresh = max(0.15, float(np.percentile(pos_rows, 20)))
    else:
        edge_thresh = 0.15

    orig_h = y1 - y0 + 1
    orig_w = x1 - x0 + 1

    valid_rows = np.where(row_avg > edge_thresh)[0]
    valid_cols = np.where(col_avg > edge_thresh)[0]
    if len(valid_rows) > 0:
        new_y0 = y0 + int(valid_rows[0])
        new_y1 = y0 + int(valid_rows[-1])
        if (new_y1 - new_y0 + 1) >= orig_h * 0.5:
            y0, y1 = new_y0, new_y1
    if len(valid_cols) > 0:
        new_x0 = x0 + int(valid_cols[0])
        new_x1 = x0 + int(valid_cols[-1])
        if (new_x1 - new_x0 + 1) >= orig_w * 0.5:
            x0, x1 = new_x0, new_x1

    pad_x = int((x1 - x0) * pad_ratio)
    pad_y = int((y1 - y0) * pad_ratio)

    if pad_y > 0 and y0 - pad_y >= 0:
        top_band = freq_map[y0 - pad_y:y0, x0:x1 + 1]
        if top_band.mean() < edge_thresh:
            pad_y = 0
    if pad_y > 0 and y1 + pad_y < h0:
        bot_band = freq_map[y1 + 1:y1 + 1 + pad_y, x0:x1 + 1]
        if bot_band.mean() < edge_thresh:
            pad_y = 0
    if pad_x > 0 and x0 - pad_x >= 0:
        left_band = freq_map[y0:y1 + 1, x0 - pad_x:x0]
        if left_band.mean() < edge_thresh:
            pad_x = 0
    if pad_x > 0 and x1 + pad_x < w0:
        right_band = freq_map[y0:y1 + 1, x1 + 1:x1 + 1 + pad_x]
        if right_band.mean() < edge_thresh:
            pad_x = 0

    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(w0 - 1, x1 + pad_x); y1 = min(h0 - 1, y1 + pad_y)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


# ============================================================
# 八、标题/副标题渲染（来自1.0版，自适应字号）
# ============================================================

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_font_cache = {}
_font_warned = [False]


def get_font(size):
    size = max(1, int(size))
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[size] = f
                return f
            except Exception:
                continue
    if not _font_warned[0]:
        print("  警告: 未找到中文字体，标题可能显示异常，建议安装微软雅黑等中文字体。")
        _font_warned[0] = True
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def fit_single_line(draw, text, max_width, font_start_px, font_min_px=6, step=1):
    if not text:
        return get_font(font_min_px), font_min_px
    size = max(int(font_start_px), int(font_min_px))
    font = get_font(size)
    w = text_width(draw, text, font)
    while w > max_width and size > font_min_px:
        size -= step
        font = get_font(size)
        w = text_width(draw, text, font)
    return font, size


LINE_SPACING = 1.35


def measure_title_block(page_w, page_h, content_w, title, subtitle):
    dummy = Image.new("L", (page_w, page_h), 255)
    draw = ImageDraw.Draw(dummy)

    title_start = int(round(page_h * 0.045))
    title_min = max(6, int(round(page_h * 0.012)))
    sub_start = int(round(page_h * 0.026))
    sub_min = max(6, int(round(page_h * 0.009)))
    gap_px = int(round(page_h * 0.016)) if (title or subtitle) else 0

    font_title, title_size = fit_single_line(draw, title, content_w, title_start, title_min)
    font_sub, sub_size = fit_single_line(draw, subtitle, content_w, sub_start, sub_min)

    reserve = 0
    if title:
        reserve += int(round(title_size * LINE_SPACING))
    if subtitle:
        reserve += int(round(sub_size * LINE_SPACING))
    reserve += gap_px

    return reserve, font_title, title_size, font_sub, sub_size, gap_px


def render_title_header(page_gray, m_top, m_left, content_w, title, subtitle,
                         font_title, title_size, font_sub, sub_size, gap_px):
    if not title and not subtitle:
        return page_gray
    pil_img = Image.fromarray(page_gray).convert("L")
    draw = ImageDraw.Draw(pil_img)
    y = m_top

    if title:
        tw = text_width(draw, title, font_title)
        x = m_left + max(0, (content_w - tw) // 2)
        draw.text((x, y), title, font=font_title, fill=0)
        y += int(round(title_size * LINE_SPACING))

    if subtitle:
        tw = text_width(draw, subtitle, font_sub)
        x = m_left + max(0, (content_w - tw) // 2)
        draw.text((x, y), subtitle, font=font_sub, fill=90)
        y += int(round(sub_size * LINE_SPACING))

    line_y = y + gap_px // 3
    line_w = max(1, title_size // 30) if title_size else 2
    draw.line([(m_left, line_y), (m_left + content_w, line_y)],
              fill=180, width=line_w)
    return np.array(pil_img)


# ============================================================
# 九、A4排版（来自1.0版，排版最好）
# ============================================================

MM_PER_INCH = 25.4


def mm_to_px(mm, dpi):
    return int(round(mm * dpi / MM_PER_INCH))


def layout_to_a4(strips, dpi, margins_mm, gap_ratio=0.02, justify=True,
                  bg=255, title_reserve_px=0):
    page_w = mm_to_px(210.0, dpi)
    page_h = mm_to_px(297.0, dpi)
    m_top, m_bot, m_left, m_right = [mm_to_px(m, dpi) for m in margins_mm]

    content_w = page_w - m_left - m_right
    content_h_normal = page_h - m_top - m_bot
    content_h_first = max(1, content_h_normal - title_reserve_px)
    gap = int(round(page_h * gap_ratio))

    scaled = []
    for s in strips:
        h, w = s.shape[:2]
        new_h = max(1, int(round(h * content_w / float(w))))
        img = cv2.resize(s, (content_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        if new_h > content_h_normal:
            scale = content_h_normal / float(new_h)
            img = cv2.resize(img, (max(1, int(round(content_w * scale))), content_h_normal),
                              interpolation=cv2.INTER_LANCZOS4)
        scaled.append(img)

    pages, cur, cur_h = [], [], 0
    page_index = 0
    for img in scaled:
        content_h = content_h_first if page_index == 0 else content_h_normal
        need = img.shape[0] + (gap if cur else 0)
        if cur and cur_h + need > content_h:
            pages.append(cur)
            page_index += 1
            cur, cur_h = [], 0
            need = img.shape[0]
        cur.append(img)
        cur_h += need
    if cur:
        pages.append(cur)

    page_arrays = []
    for pi, items in enumerate(pages):
        content_h = content_h_first if pi == 0 else content_h_normal
        top_offset = m_top + (title_reserve_px if pi == 0 else 0)
        page = np.full((page_h, page_w), bg, dtype=np.uint8)
        total_h = sum(im.shape[0] for im in items)
        if justify and len(items) > 1:
            free = content_h - total_h
            g = min(free // (len(items) - 1), gap * 3)
            g = max(g, gap)
        else:
            g = gap
        y = top_offset
        for im in items:
            h, w = im.shape[:2]
            x = m_left + (content_w - w) // 2
            page[y:y + h, x:x + w] = im
            y += h + g
        page_arrays.append(page)

    geometry = (page_w, page_h, m_top, m_left, content_w)
    return page_arrays, geometry


# ============================================================
# 十、辅助: 参数解析（来自1.0版）
# ============================================================

def parse_time(s):
    if s is None or str(s).strip() == "":
        return None
    parts = [float(p) for p in str(s).split(':')]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def fmt_time(sec):
    if sec is None:
        return "?"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_margins(s):
    vals = [float(v) for v in s.split(':')]
    if len(vals) != 4:
        raise ValueError("--margins 需要4个值: 上:下:左:右")
    return tuple(vals)


def parse_percentile(s):
    if s is None:
        return 0.5, 99.5
    vals = [float(v) for v in s.split(':')]
    if len(vals) != 2:
        raise ValueError("--percentile 需要2个值: 低:高")
    return vals[0], vals[1]


def parse_crop_ratio(s):
    if not s:
        return None
    r0, r1 = [float(v) for v in s.split(':')]
    return r0, r1


def parse_crop_y(s):
    if not s:
        return None
    y0, y1 = [int(v) for v in s.split(':')]
    return y0, y1


# ============================================================
# 十一、主流程（综合1.0版排版 + 1.1版画面优化）
# ============================================================

def main():
    t_total_start = time.time()

    ap = argparse.ArgumentParser(description="从视频提取乐谱截图并排入A4版面导出PDF (v2.0)")
    ap.add_argument("video", help="视频文件路径")
    ap.add_argument("-o", "--output", required=True, help="输出PDF路径")
    ap.add_argument("-t", "--title", type=str, default="",
                     help="标题(单行显示，自动缩字号适配页宽)")
    ap.add_argument("-s", "--subtitle", type=str, default="",
                     help="副标题(单行显示，自动缩字号适配页宽)")

    ap.add_argument("--crop", type=str, default=None,
                     help="比例裁剪 '起始:结束'，如 0.55:1.0")
    ap.add_argument("--crop-y", type=str, default=None,
                     help="像素裁剪 'y0:y1'，如 120:600")
    ap.add_argument("--auto-crop", action="store_true",
                     help="自动检测乐谱在画面中的固定显示区域")

    ap.add_argument("--notation", choices=["auto", "staff", "tab", "both"], default="auto")
    ap.add_argument("--start", type=str, default=None, help="起始时间 HH:MM:SS 或秒数")
    ap.add_argument("--end", type=str, default=None, help="结束时间 HH:MM:SS 或秒数")
    ap.add_argument("--fps", type=float, default=0.5, help="采样频率(次/秒)")

    ap.add_argument("--group-t", type=float, default=0.95)
    ap.add_argument("--merge-t", type=float, default=0.95)
    ap.add_argument("--dedup-t", type=float, default=0.65)
    ap.add_argument("--diff-max", type=float, default=0.08)

    ap.add_argument("--white-min", type=float, default=0.55)
    ap.add_argument("--cover-min", type=float, default=0.60)

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--margins", type=str, default="15:15:10:10")
    ap.add_argument("--gap-ratio", type=float, default=0.005)
    ap.add_argument("--no-justify", action="store_true")
    ap.add_argument("--percentile", type=str, default="0.5:99.5")
    ap.add_argument("--sharpen", type=float, default=1.2)
    ap.add_argument("--sharpen-sigma", type=float, default=1.0)

    temp_group = ap.add_mutually_exclusive_group()
    temp_group.add_argument("--keep-temp", action="store_true",
                             help="处理完成后保留临时文件夹(默认行为)")
    temp_group.add_argument("--delete-temp", action="store_true",
                             help="处理完成后静默删除临时文件夹")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    title = args.title.strip()
    subtitle = args.subtitle.strip()
    margins_mm = parse_margins(args.margins)
    pct_lo, pct_hi = parse_percentile(args.percentile)
    crop_ratio = parse_crop_ratio(args.crop)
    crop_y = parse_crop_y(args.crop_y)

    out_base = os.path.splitext(args.output)[0]
    out_dir = out_base + "_temp"
    log_path = out_base + "_grouping_log.txt"
    os.makedirs(out_dir, exist_ok=True)
    print(f"临时文件夹: {out_dir}")
    print(f"分组/丢弃记录将生成于: {log_path}")

    log_lines = []
    def log(msg):
        log_lines.append(msg)

    # ---- [1/8] 打开视频 ----
    print("[1/8] 打开视频...")

    def detect_codec(path):
        try:
            for p in [r"C:\Tools\ffmpeg\ffmpeg.exe", "ffprobe"]:
                cmd = [p, "-v", "quiet", "-show_streams", "-select_streams", "v:0", "-i", path]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                for line in (r.stdout + r.stderr).splitlines():
                    if "codec_name=" in line:
                        return line.split("codec_name=")[1].strip()
        except Exception:
            pass
        return None

    video_path = args.video
    codec = detect_codec(video_path)
    converted_path = None
    if codec and "av1" in codec.lower():
        converted_path = os.path.splitext(video_path)[0] + "_h264_tmp.mp4"
        print(f"  检测到 AV1 编码，正在转为 H.264 以加速读取...")
        ffmpeg_bin = r"C:\Tools\ffmpeg\ffmpeg.exe"
        if not os.path.exists(ffmpeg_bin):
            ffmpeg_bin = "ffmpeg"
        cmd = [ffmpeg_bin, "-y", "-i", video_path,
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-an", converted_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(converted_path):
            print(f"  转换完成: {converted_path}")
            video_path = converted_path
        else:
            print(f"  转换失败，使用原始文件继续: {r.stderr[:200]}")
            converted_path = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video}")
        sys.exit(1)
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_sec = parse_time(args.start) or 0.0
    end_sec = parse_time(args.end)
    start_frame = int(start_sec * fps_native)
    end_frame = int(end_sec * fps_native) if end_sec is not None else total
    end_frame = min(end_frame, total)
    if end_frame <= start_frame:
        print(f"错误: --end({args.end}) 必须晚于 --start({args.start})")
        sys.exit(1)
    print(f"  原始fps={fps_native:.2f}, 总帧数={total}")
    print(f"  处理区间: 帧[{start_frame}, {end_frame}) "
          f"(~{fmt_time(start_frame/fps_native)} ~ {fmt_time(end_frame/fps_native)})")

    log(f"视频: {args.video}")
    log(f"处理区间: 帧[{start_frame}, {end_frame})  "
        f"时间 {fmt_time(start_frame/fps_native)} ~ {fmt_time(end_frame/fps_native)}")
    log("")

    # ---- [2/8] 确定裁剪区域 ----
    crop = None
    if crop_ratio is not None:
        print("[2/8] 使用比例裁剪(--crop)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取起始帧以确定画面尺寸")
            sys.exit(1)
        full_h, full_w = frame.shape[:2]
        r0, r1 = crop_ratio
        y0 = max(0, int(round(full_h * r0)))
        y1 = min(full_h, int(round(full_h * r1)))
        crop = (0, y0, full_w, y1 - y0)
        print(f"  裁剪区域: x=0 y={y0} w={full_w} h={y1 - y0} (比例 {r0}:{r1})")
        log(f"裁剪方式: 比例 --crop {r0}:{r1} -> 矩形 {crop}")
    elif crop_y is not None:
        print("[2/8] 使用像素裁剪(--crop-y)，保留全宽")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取起始帧以确定画面宽度")
            sys.exit(1)
        full_w = frame.shape[1]
        y0, y1 = crop_y
        y1 = min(y1, frame.shape[0])
        crop = (0, y0, full_w, y1 - y0)
        print(f"  裁剪区域: x=0 y={y0} w={full_w} h={y1 - y0}")
        log(f"裁剪方式: 像素 --crop-y {y0}:{y1} -> 矩形 {crop}")
    elif args.auto_crop:
        print("[2/8] 自动检测乐谱区域...")
        crop = auto_detect_crop(cap, start_frame, end_frame)
        if crop:
            print(f"  检测到区域: x={crop[0]} y={crop[1]} w={crop[2]} h={crop[3]}")
            log(f"裁剪方式: --auto-crop 检测结果 = {crop}")
        else:
            print("  未能检测到稳定白底区域，使用整帧")
            log("裁剪方式: --auto-crop 未检测到稳定区域，使用整帧")
    else:
        print("[2/8] 未指定裁剪方式，使用整帧")
        log("裁剪方式: 未指定，使用整帧")
    log("")

    # ---- [3/8] 采样 + 乐谱检测过滤 ----
    print("[3/8] 采样帧并过滤乐谱内容(seek模式)...")
    step = max(1, int(round(fps_native / max(args.fps, 1e-6))))
    target_frames = list(range(start_frame, end_frame, step))
    print(f"  共需采样 {len(target_frames)} 个时间点(步长={step}帧, ~{args.fps}fps)")

    candidates = []
    rejected = []
    prog = Progress(len(target_frames), "采样检测", every=max(1, len(target_frames)//20))
    for fidx in target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        prog.update()
        if not ret:
            continue
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if crop:
            x, y, w, h = crop
            gray_full = gray_full[y:y + h, x:x + w]
        if has_sheet(gray_full, args.notation, args.white_min, args.cover_min,
                     debug=args.debug):
            candidates.append((fidx, gray_full))
        else:
            rejected.append(fidx)
    prog.done()
    cap.release()
    print(f"  命中乐谱帧: {len(candidates)} 个  |  判定非乐谱丢弃: {len(rejected)} 个")

    log(f"采样点总数: {len(target_frames)}  命中乐谱: {len(candidates)}  "
        f"判定非乐谱丢弃: {len(rejected)}")
    if rejected:
        log(f"被丢弃的采样帧编号(判定为非乐谱): {rejected}")
    log("")

    if not candidates:
        log("错误: 未检测到含乐谱的帧")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print("错误: 未检测到含乐谱的帧(请检查裁剪区域，或用 --debug 观察阈值)")
        sys.exit(1)

    raw_dir = os.path.join(out_dir, "01_raw_clean")
    os.makedirs(raw_dir, exist_ok=True)
    for i, (fi, g) in enumerate(candidates):
        imwrite_unicode(os.path.join(raw_dir, f"{i:04d}_f{fi}.png"), g)

    # ---- [4/8] 三级去重 ----
    print("[4/8] 三级去重...")
    s1 = stage1_group_consecutive(candidates, args.group_t, args.diff_max)
    print(f"  第一级(相邻折叠): {len(s1)} 段")
    s2 = stage2_merge_repeats(s1, args.merge_t, args.diff_max)
    print(f"  第二级(全局合并): {len(s2)} 段")
    s3 = stage3_final_dedup(s2, args.dedup_t, args.diff_max)
    print(f"  第三级(保险去重): {len(s3)} 段")

    log("=" * 60)
    log(f"三级去重结果: 采样命中{len(candidates)} -> 第一级{len(s1)} "
        f"-> 第二级{len(s2)} -> 第三级{len(s3)}")
    log("=" * 60)
    log("")
    log("最终分组明细(每组: 采用帧 + 折叠/吸收的其他原始帧):")
    for gi, (img, rep, members) in enumerate(s3, 1):
        others = sorted(set(members) - {rep})
        if others:
            log(f"  第{gi:02d}组 -> 采用帧 #{rep}  (另折叠了 {len(others)} 帧: {others})")
        else:
            log(f"  第{gi:02d}组 -> 采用帧 #{rep}  (无重复帧被折叠)")
    log("")

    group_dir = os.path.join(out_dir, "02_groups")
    os.makedirs(group_dir, exist_ok=True)
    for gi, (img, rep, members) in enumerate(s3, 1):
        imwrite_unicode(os.path.join(group_dir, f"group{gi:02d}_rep{rep}.png"), img)

    # ---- [5/8] 组内合成（来自1.1版的优化画面处理） ----
    print("[5/8] 组内合成(中位数+背景平整)...")
    t5 = time.time()

    cand_dict = {fi: gray for fi, gray in candidates}

    strips = []
    for img, rep, members in s3:
        if len(members) > 1:
            inks = []
            for fi in members:
                if fi in cand_dict:
                    g = cand_dict[fi]
                    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if len(g.shape) == 2 else g
                    inks.append(to_ink(bgr))
            if inks:
                strips.append(flatten_bg(combine(inks)))
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
                strips.append(flatten_bg(to_ink(bgr)))
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
            strips.append(flatten_bg(to_ink(bgr)))

    print(f'  合成完成 ({time.time()-t5:.1f}s)')

    strip_dir = os.path.join(out_dir, "03_strips")
    os.makedirs(strip_dir, exist_ok=True)
    for i, s in enumerate(strips):
        imwrite_unicode(os.path.join(strip_dir, f"{i:04d}.png"), s)

    # ---- [6/8] 增强放大（来自1.1版enhance_strip） ----
    print("[6/8] 增强并放大到打印分辨率...")
    t6 = time.time()

    content_w_probe = int((8.27 - (margins_mm[2] + margins_mm[3]) / 25.4) * args.dpi)
    strips = [enhance_strip(s, content_w_probe, pct_lo, pct_hi,
                            args.sharpen, args.sharpen_sigma)
              for s in strips]

    print(f'  增强完成 ({time.time()-t6:.1f}s)')

    # ---- [7/8] 测量标题区域并排版（来自1.0版layout_to_a4） ----
    print("[7/8] 测量标题区域并按比例排入A4页面...")
    t7 = time.time()

    page_w_probe = mm_to_px(210.0, args.dpi)
    page_h_probe = mm_to_px(297.0, args.dpi)
    m_top_probe, m_bot_probe, m_left_probe, m_right_probe = [
        mm_to_px(m, args.dpi) for m in margins_mm]
    content_w_probe_total = page_w_probe - m_left_probe - m_right_probe

    (title_reserve_px, font_title, title_size,
     font_sub, sub_size, title_gap_px) = measure_title_block(
        page_w_probe, page_h_probe, content_w_probe_total, title, subtitle)

    pages, geometry = layout_to_a4(
        strips, dpi=args.dpi, margins_mm=margins_mm,
        gap_ratio=args.gap_ratio, justify=not args.no_justify,
        title_reserve_px=title_reserve_px,
    )
    page_w, page_h, m_top, m_left, content_w_final = geometry

    if pages and (title or subtitle):
        pages[0] = render_title_header(
            pages[0], m_top, m_left, content_w_final, title, subtitle,
            font_title, title_size, font_sub, sub_size, title_gap_px,
        )

    print(f'  共生成 {len(pages)} 页')
    log(f"最终生成PDF页数: {len(pages)}")
    if title:
        log(f"标题最终字号: {title_size}px (单行，自适应页宽)")
    if subtitle:
        log(f"副标题最终字号: {sub_size}px (单行，自适应页宽)")
    log(f"  排版完成 ({time.time()-t7:.1f}s)")

    # ---- [8/8] 导出PDF + 临时文件处理 ----
    print("[8/8] 导出PDF...")
    t8 = time.time()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    png_paths = []
    page_dir = os.path.join(out_dir, "04_pages")
    os.makedirs(page_dir, exist_ok=True)
    for i, page_arr in enumerate(pages):
        fp = os.path.join(page_dir, f"page_{i:03d}.png")
        imwrite_unicode(fp, page_arr)
        png_paths.append(fp)
    with open(args.output, 'wb') as f:
        f.write(img2pdf.convert(png_paths))
    print(f"  PDF已导出: {args.output}")
    print(f"  导出完成 ({time.time()-t8:.1f}s)")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"  分组/丢弃日志已写入: {log_path}")

    # ---- 临时文件处理 ----
    print("[8/8] 处理临时文件...")
    print(f"  临时文件夹路径: {out_dir}")
    print(f"    01_raw_clean/ -> 每个采样命中乐谱帧")
    print(f"    02_groups/    -> 三级去重后每组的最终代表帧")
    print(f"    03_strips/    -> 裁边+增强后的最终乐谱条")
    print(f"    04_pages/     -> 最终排版好的每页PNG")

    if args.delete_temp:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"临时文件夹已删除: {out_dir}")
    else:
        print(f"临时文件保留在: {out_dir}")

    if converted_path and os.path.exists(converted_path):
        os.remove(converted_path)
        print(f"  已清理 H.264 临时文件: {converted_path}")

    total_elapsed = time.time() - t_total_start
    print(f"完成，总耗时 {total_elapsed:.1f}s。")


if __name__ == "__main__":
    main()
