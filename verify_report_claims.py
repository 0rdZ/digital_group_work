"""逐条核验组员修复报告 D1-D9 / N1-N5 在当前本地代码中的实现状态。"""
from __future__ import annotations

import inspect

import cv2
import numpy as np

from docscan.core import colorspace, freq, metrics
from docscan.experiments import synth
from docscan.experiments.utils import render_text_page
from docscan.pipeline.stages import (
    ENHANCE_SPEC, RESTORE_SPEC, stage_enhance, stage_restore,
)


def g01(x):
    a = np.asarray(x, dtype=np.float32)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return np.clip(a, 0, 1)


print("=" * 78)
page, _ = render_text_page(width=1024, height=1024, font_px=30)
page_g = g01(page)

# ---------------- D1 ----------------
sig = inspect.signature(freq.find_periodic_peaks)
print("D1  参数:", list(sig.parameters))
print("D1  max_period_px 存在:", "max_period_px" in sig.parameters)
moire, base = synth.add_moire(page_g, wavelength=28, angle_deg=30.0, return_base=True)
spec = freq.magnitude_spectrum(moire)
peaks = freq.find_periodic_peaks(spec, top_n=2)
print("D1  检出峰:", peaks[:2])
th = (1024 / 28.0) * np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
print("D1  理论峰:", tuple(np.round(th, 1)))
if peaks:
    print("D1  首峰-理论距离: %.1f bin" % np.hypot(peaks[0][0] - th[0], peaks[0][1] - th[1]))
nt = freq.notch_filter(moire, peaks, radius=12)
print("D1  陷波后 SSIM %.4f（退化图自身 %.4f）"
      % (metrics.ssim(base, g01(nt)), metrics.ssim(base, moire)))

# ---------------- D2 ----------------
from docscan.experiments import exp_e2e
src = inspect.getsource(exp_e2e)
print("\nD2  'defocus' in STAGE_POLICY:", "defocus" in exp_e2e.STAGE_POLICY)
print("D2  策略:", exp_e2e.STAGE_POLICY.get("defocus"))
deg_d, gt_d = synth.synthesize(page_g, "defocus")
print("D2  defocus 退化图 SSIM(gt,deg) = %.4f（<1 说明真退化）" % metrics.ssim(g01(gt_d), g01(deg_d)))
print("D2  未知场景抛错:", "raise ValueError" in src and "未知场景" in src)
print("D2  未知场景静默回退 normal:", 'scene = "normal"' in src)

# ---------------- D3 ----------------
print("\nD3  bgr_to_hsv/hsv_to_bgr/apply_on_value:",
      hasattr(colorspace, "bgr_to_hsv"), hasattr(colorspace, "hsv_to_bgr"),
      hasattr(colorspace, "apply_on_value"))
img = np.zeros((1, 6, 3), dtype=np.float32)
for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0),
                       (0, 255, 255), (255, 255, 0), (255, 0, 255)]):
    img[0, i] = np.array(c, dtype=np.float32) / 255.0
H, S, V = colorspace.bgr_to_hsv(img)
print("D3  六原色 H(度):", np.round(H[0, :6], 1))
print("D3  量程 H∈[%.0f,%.0f] S∈[%.2f,%.2f] V∈[%.2f,%.2f]"
      % (H.min(), H.max(), S.min(), S.max(), V.min(), V.max()))
ref = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("D3  vs cv2: max|ΔH|=%.2e° max|ΔS|=%.2e max|ΔV|=%.2e"
      % (np.abs(H - ref[..., 0]).max(), np.abs(S - ref[..., 1]).max(),
         np.abs(V - ref[..., 2]).max()))
print("D3  V 量程约定:", "0..255（本地版）" if V.max() > 1.5 else "0..1（组员版）")

# ---------------- D4 ----------------
cast_c, _ = synth.synthesize(page, "cast")
cast_g, _ = synth.synthesize(page_g, "cast")
print("\nD4  cast(彩图) 通道均值 B/G/R:", np.round(np.asarray(cast_c).reshape(-1, 3).mean(0), 1))
print("D4  cast(灰度输入) 原样返回:", np.array_equal(cast_g, page_g))
print("D4  exp_e2e 对 cast 喂彩图:", 'synthesize(page, "cast")' in src)

# ---------------- D5 ----------------
print("\nD5  ENHANCE_SPEC method 选项:", [k for k, *_ in ENHANCE_SPEC[0].choices])
outs = {}
for m in ("clahe", "histeq", "stretch", "none"):
    o, note, _ = stage_enhance(page, {"method": m})
    outs[m] = o
    print("D5    %-8s -> %s" % (m, note))
print("D5  histeq 生效:", not np.array_equal(outs["histeq"], outs["clahe"]),
      "| stretch 生效:", not np.array_equal(outs["stretch"], outs["clahe"]))
try:
    stage_enhance(page, {"method": "bogus"})
    print("D5  未知方法: 静默回退 clahe（未抛错）")
except ValueError as e:
    print("D5  未知方法抛错:", e)

# ---------------- D6 ----------------
print("\nD6  RESTORE_SPEC psf_type 选项:", [k for k, *_ in RESTORE_SPEC[0].choices])
deg, gt = synth.synthesize(page_g, "defocus")
gt = g01(gt)
dg = g01(deg)
r1, n1, _ = stage_restore(deg, {"psf_type": "defocus", "defocus_radius": 5, "K": 0.03})
r2, n2, _ = stage_restore(deg, {"psf_type": "motion", "psf_mode": "manual",
                                "length": 11, "angle_deg": 0, "K": 0.03})
print("D6  不处理       SSIM %.4f" % metrics.ssim(gt, dg))
print("D6  圆盘 PSF     SSIM %.4f  (%s)" % (metrics.ssim(gt, g01(r1)), n1))
print("D6  线段 PSF     SSIM %.4f  (%s)" % (metrics.ssim(gt, g01(r2)), n2))
try:
    stage_restore(deg, {"psf_type": "bogus"})
    print("D6  未知 psf_type: 静默回退 motion（未抛错）")
except ValueError as e:
    print("D6  未知 psf_type 抛错:", e)

# ---------------- D8 ----------------
print("\nD8  exp_e2e 用 csv 模块:", "import csv" in src, "| 手工 join:", '",".join' in src)

# ---------------- D9 ----------------
print("D9  exp_e2e 是否引用 exp_ablation:", "exp_ablation" in src)
print("D9  freq.py 是否引用 notch_radius.csv:", "notch_radius" in inspect.getsource(freq))
print("=" * 78)
