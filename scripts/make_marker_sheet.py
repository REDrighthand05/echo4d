#!/usr/bin/env python3
"""Generate the printable AprilTag marker sheet (assets/markers_sheet.pdf).

Four A4 pages at 300 dpi, one DICT_APRILTAG_36h11 marker per page
(IDs 0-3), marker side exactly 160 mm, generous white quiet zone.
ID 0 goes flat on the floor; IDs 1-3 go on walls. The pipeline
(echo_scene/markers.py) reads these back from filmed frames to recover
true metric scale and a level floor reference.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 300
PAGE_W, PAGE_H = 2480, 3508  # A4 at 300 dpi
PAGE_W_MM = 210.0
MARKER_SIDE_PX = 1890
MARKER_SIDE_MM = MARKER_SIDE_PX / PAGE_W * PAGE_W_MM
assert abs(MARKER_SIDE_MM - 160.0) < 1.0, f"marker side is {MARKER_SIDE_MM:.2f} mm, not 160"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
OUT_DIR = Path(__file__).resolve().parent.parent / "assets"

PLACEMENT = {
    0: "贴法：平贴在地板中央，尽量平整、不要翘边。",
    1: "贴法：贴在墙面上，尽量平整、不要翘边。",
    2: "贴法：贴在另一面墙上，尽量平整、不要翘边。",
    3: "贴法：贴在墙面或大件家具的竖直面上，尽量平整。",
}


def make_page(marker_id: int) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIDE_PX)
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    marker_img = Image.fromarray(marker)
    x0 = (PAGE_W - MARKER_SIDE_PX) // 2
    y0 = 420
    page.paste(marker_img, (x0, y0))

    draw = ImageDraw.Draw(page)
    font_big = ImageFont.truetype(FONT_PATH, 110)
    font_mid = ImageFont.truetype(FONT_PATH, 72)
    font_small = ImageFont.truetype(FONT_PATH, 60)

    def centered(text, y, font, fill=(20, 20, 20)):
        bbox = draw.textbbox((0, 0), text, font=font)
        draw.text(((PAGE_W - (bbox[2] - bbox[0])) / 2, y), text, font=font, fill=fill)

    centered(f"ECHO 标记纸 · {marker_id} 号", 130, font_big)
    ty = y0 + MARKER_SIDE_PX + 110
    centered(f"本页标记编号：{marker_id}", ty, font_mid)
    centered("打印时请选择实际大小（100%），勿缩放", ty + 120, font_mid, fill=(180, 30, 30))
    centered(PLACEMENT[marker_id], ty + 240, font_small)
    centered("拍视频时慢慢走过，让标记完整入镜几秒即可。", ty + 350, font_small)
    return page


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = [make_page(i) for i in range(4)]
    pdf_path = OUT_DIR / "markers_sheet.pdf"
    pages[0].save(
        pdf_path, "PDF", save_all=True, append_images=pages[1:],
        resolution=float(DPI),
    )
    # Preview: 2x2 contact sheet.
    thumb_w, thumb_h = 620, 877
    sheet = Image.new("RGB", (thumb_w * 2, thumb_h * 2), "white")
    for i, page in enumerate(pages):
        thumb = page.resize((thumb_w, thumb_h), Image.LANCZOS)
        sheet.paste(thumb, ((i % 2) * thumb_w, (i // 2) * thumb_h))
    preview_path = OUT_DIR / "markers_sheet_preview.png"
    sheet.save(preview_path)
    print(f"marker side: {MARKER_SIDE_MM:.2f} mm (target 160)")
    print(f"wrote {pdf_path} ({len(pages)} pages) and {preview_path}")


if __name__ == "__main__":
    main()
