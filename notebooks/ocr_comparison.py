"""
OCR Comparison: EasyOCR vs PaddleOCR at different confidence levels
Run: python notebooks/ocr_comparison.py
Generates: reports/ocr_comparison.png
"""

import csv
import random
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from pathlib import Path
import numpy as np
import textwrap

ROOT = Path(__file__).resolve().parent.parent

# --- Load data ---
easy = {}
with open(ROOT / "data/processed/metadata_ocr.csv") as f:
    for row in csv.DictReader(f):
        easy[row["filename"]] = row

paddle = {}
for pf in (ROOT / "data").rglob("ocr_paddle.csv"):
    with open(pf) as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "")
            paddle[fn] = dict(row)
            paddle[fn]["source_path"] = str(pf.parent / fn)

common = set(easy.keys()) & set(paddle.keys())

# --- Select examples at 3 confidence levels ---
targets = [
    (0.6, 0.08, "Low (~0.6)"),
    (0.75, 0.08, "Medium (~0.75)"),
    (0.92, 0.08, "High (~0.92)"),
]

random.seed(42)

selected = []
for target, delta, label in targets:
    candidates = []
    for fn in sorted(common):
        ec = float(easy[fn].get("confidence", 0))
        pc = float(paddle[fn].get("confidence", 0))
        et = easy[fn].get("ocr_text", "").strip()
        pt = paddle[fn].get("ocr_text", "").strip()
        sp = easy[fn].get("source_path", "")
        if not sp:
            sp = paddle[fn].get("source_path", "")
        if (
            abs(ec - target) < delta
            and ec > 0
            and et
            and pt
            and Path(sp).exists()
        ):
            candidates.append((fn, ec, pc, et, pt, sp, label))
    random.shuffle(candidates)
    selected.extend(candidates[:3])

# --- Plot ---
n = len(selected)
fig, axes = plt.subplots(n, 2, figsize=(16, 4.5 * n), gridspec_kw={"width_ratios": [1, 1.8]})
fig.suptitle("OCR Comparison: EasyOCR vs PaddleOCR at Different Confidence Levels",
             fontsize=16, fontweight="bold", y=0.995)

for i, (fn, ec, pc, et, pt, sp, label) in enumerate(selected):
    ax_img = axes[i, 0]
    ax_text = axes[i, 1]

    # Image
    try:
        img = Image.open(sp).convert("RGB")
        img.thumbnail((400, 400))
        ax_img.imshow(np.array(img))
    except Exception:
        ax_img.text(0.5, 0.5, "Image\nnot found", ha="center", va="center", fontsize=14)
    ax_img.set_title(f"{label}\n{fn}", fontsize=10, fontweight="bold")
    ax_img.axis("off")

    # Text comparison
    et_wrapped = textwrap.fill(et[:150], width=60)
    pt_wrapped = textwrap.fill(pt[:150], width=60)
    text_block = (
        f"EasyOCR  [conf={ec:.3f}]:\n{et_wrapped}\n\n"
        f"PaddleOCR [conf={pc:.3f}]:\n{pt_wrapped}"
    )
    ax_text.text(
        0.05, 0.95, text_block,
        transform=ax_text.transAxes,
        fontsize=11, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    # Color-code confidence
    easy_color = "green" if ec >= 0.8 else ("orange" if ec >= 0.6 else "red")
    paddle_color = "green" if pc >= 0.8 else ("orange" if pc >= 0.6 else "red")
    ax_text.barh([1, 0], [ec, pc], color=[easy_color, paddle_color], height=0.3, left=0, alpha=0.3)
    ax_text.set_xlim(0, 1.0)
    ax_text.set_yticks([1, 0])
    ax_text.set_yticklabels(["EasyOCR", "PaddleOCR"], fontsize=10)
    ax_text.set_xlabel("Confidence", fontsize=10)
    ax_text.axvline(x=0.6, color="red", linestyle="--", alpha=0.5, label="threshold=0.6")
    ax_text.legend(fontsize=8)

plt.tight_layout()
out_path = ROOT / "reports" / "ocr_comparison.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
