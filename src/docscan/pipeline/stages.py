"""Stage 抽象与默认处理链。

每个 Stage 是一个可独立开关的处理步骤：
* ``func(image, params) -> (image, notes, metrics)``，纯函数式，便于测试；
* ``params_spec`` 描述可调参数（数值型 NumParam / 枚举型 EnumParam），
  UI 据此自动生成滑杆或下拉框；
* 关闭的阶段由 Runner 直接跳过（函数不被调用）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from docscan.core import (
    binarize,
    color,
    denoise,
    enhance,
    freq,
    geometry,
    illumination,
    restore,
    sharpen,
)
from docscan.core.types import Image, to_uint8


@dataclass(frozen=True)
class NumParam:
    """数值参数：UI 渲染为滑杆 + 数值框。"""

    key: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 0.1
    integer: bool = False  # 为 True 时 UI 取整显示，stage 内部也会 int() 兜底


@dataclass(frozen=True)
class EnumParam:
    """枚举参数：UI 渲染为只读下拉框。choices 为 (取值, 中文标签) 序列。"""

    key: str
    label: str
    default: str
    choices: tuple[tuple[str, str], ...]

    def label_of(self, value: str) -> str:
        return dict(self.choices).get(value, value)


ParamSpec = NumParam | EnumParam


@dataclass
class Stage:
    """处理链中的一个步骤。"""

    name: str
    label: str
    func: Callable[[Image, dict], tuple[Image, str, dict[str, float]]]
    params_spec: list[ParamSpec] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    enabled: bool = True

    def reset_params(self) -> None:
        """把参数恢复为默认值（spec 外的临时键，如频谱手动选点，一并清除）。"""
        self.params = {spec.key: spec.default for spec in self.params_spec}


def stage_warp(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """几何校正阶段。

    灰度输入也能处理：检测器要求三通道（它要用 HSV 的 S/V），
    因此灰度图先临时转 BGR 只用于**检测**，变换仍作用在原图上，
    避免「灰度链里几何校正直接报错跳过」这种静默降级。
    """
    min_conf = float(params.get("min_confidence", 0.6))
    probe = img if img.ndim == 3 else cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
    quad = geometry.detect_quad(probe, min_confidence=min_conf)
    if quad.is_fallback:
        note = f"角点检测失败（置信度 {quad.confidence:.2f}），保留整图"
        return img, note, {"confidence": quad.confidence, "warped": 0.0}
    out = geometry.warp_perspective(img, quad)
    note = f"角点置信度 {quad.confidence:.2f}"
    return out, note, {"confidence": quad.confidence, "warped": 1.0}


def stage_binarize(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """二值化阶段：自适应 / Otsu / Sauvola 三选一（彩色输入先转灰度）。"""
    method = str(params.get("method", "adaptive"))
    gray = binarize.to_gray(img)
    if method == "otsu":
        out = binarize.otsu(gray)
        detail = "全局 Otsu"
    elif method == "sauvola":
        window = int(params.get("window", 25))
        k = float(params.get("k", 0.2))
        out = binarize.sauvola(gray, window=window, k=k)
        detail = f"Sauvola w={window}, k={k:g}"
    else:
        block = int(params.get("block_size", 31))
        c = float(params.get("c", 10.0))
        out = binarize.adaptive(gray, block_size=block, c=c)
        detail = f"自适应 blockSize={block}, C={c:g}"
    out = binarize.clean_small_components(out, min_area=int(params.get("min_area", 8)))
    ink = float((out < 128).mean())
    return out, detail, {"ink_ratio": ink}


def stage_illumination(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """光照归一阶段：背景估计去阴影。"""
    sigma = float(params.get("sigma", 30.0))
    out = illumination.remove_shading(img, sigma=sigma)
    before = float(np.asarray(to_uint8(img)).std())
    after = float((to_uint8(out)).std())
    return out, f"sigma={sigma:g}", {"std_before": before, "std_after": after}


def stage_white_balance(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """白平衡阶段：灰世界 / 白块估计。"""
    method = str(params.get("method", "white_patch"))
    strength = float(params.get("strength", 1.0))
    if method == "gray_world":
        out = color.gray_world(img, strength=strength)
    elif method == "white_patch":
        out = color.white_patch(img, strength=strength)
    else:
        raise ValueError(f"未知白平衡方法：{method}")
    return out, f"{method}, strength={strength:g}", {}


def stage_enhance(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """对比度增强阶段：CLAHE / 直方图均衡 / 对比度拉伸 / 仅伽马，均只在亮度通道。"""
    method = str(params.get("method", "clahe"))
    if method == "none":
        out = img
        detail = "不增强"
    elif method == "histeq":
        out = enhance.hist_equalize(img)
        detail = "直方图均衡"
    elif method == "stretch":
        out = enhance.contrast_stretch(img)
        detail = "对比度拉伸 1%~99%"
    else:
        clip = float(params.get("clip_limit", 2.0))
        grid = int(params.get("grid_size", 8))
        out = enhance.clahe(img, clip_limit=clip, grid_size=grid)
        detail = f"CLAHE clip={clip:g}, grid={grid}"
    gamma = float(params.get("gamma", 1.0))
    if gamma != 1.0:
        out = enhance.apply_gamma(out, gamma)
        detail += f", γ={gamma:g}"
    return out, detail, {}


def stage_denoise(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """去噪阶段：中值 / 双边（放在二值化与锐化之前，避免噪点被放大成笔画）。"""
    method = str(params.get("method", "median"))
    if method == "bilateral":
        sigma_color = float(params.get("sigma_color", 0.05))
        out = denoise.bilateral(img, sigma_color=sigma_color)
        return out, f"双边 σc={sigma_color:g}", {}
    ksize = int(params.get("ksize", 3))
    out = denoise.median(img, ksize=ksize)
    return out, f"中值 {max(3, ksize | 1)}×{max(3, ksize | 1)}", {}


def stage_freq_denoise(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """频域去网纹阶段：自动找峰或使用 UI 手动选点。

    自动找峰失败时**必须可见**：陷波后若峰能量几乎没降，说明命中的是伪峰，
    此时在 note 里写明并置 ``notch_ok=0``，而不是拿一个漂亮的 dB 数糊过去。
    """
    manual = params.get("manual_peaks") or []
    if manual:
        peaks = [(int(du), int(dv)) for du, dv in manual]
        source = "手动选点"
    else:
        spec = freq.magnitude_spectrum(img)
        dc_radius = int(params.get("dc_radius", 0)) or None
        peaks = freq.find_periodic_peaks(
            spec, top_n=int(params.get("top_n", 4)), dc_radius=dc_radius
        )
        source = "自动找峰"
    if not peaks:
        return img, "未发现周期峰，跳过滤波", {"peaks": 0.0, "notch_ok": 0.0}

    kind = str(params.get("kind", "butterworth"))
    out = freq.notch_filter(
        img, peaks, kind=kind,
        radius=int(params.get("radius", 0)) or None,
        order=int(params.get("order", 4)),
    )
    drop = freq.energy_drop_db(img, out, peaks[0])
    note = f"{source} {len(peaks)} 点: {peaks[:4]}"
    ok = 1.0
    if drop > -6.0:
        ok = 0.0
        note += "（峰能量未下降，疑似伪峰，请在频谱图上手动选点）"
    return out, note, {"peaks": float(len(peaks)), "peak_drop_db": drop, "notch_ok": ok}


def stage_restore(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """退化复原阶段：运动/散焦 PSF（倒谱估计或手动）→ 维纳/逆滤波。"""
    psf_type = str(params.get("psf_type", "motion"))
    metrics: dict[str, float] = {}

    if psf_type == "defocus":
        radius = int(params.get("defocus_radius", 5))
        psf = restore.disk_psf(max(31, radius * 4 | 1), radius)
        note = f"散焦 PSF: r={radius}"
        metrics["psf_radius"] = float(radius)
    else:
        if str(params.get("psf_mode", "est")) == "est":
            try:
                angle, length = restore.estimate_psf_cepstrum(img)
            except RuntimeError as exc:
                return img, f"PSF 估计失败（{exc}），跳过复原", {"psf_length": 0.0}
            note = f"倒谱估计 PSF: L={length}, θ={angle:.1f}°"
        else:
            length = int(params.get("length", 9))
            angle = float(params.get("angle_deg", 0.0))
            note = f"手动 PSF: L={length}, θ={angle:.1f}°"
        psf = restore.motion_psf(max(31, length * 2 | 1), length, angle)
        metrics["psf_length"] = float(length)
        metrics["psf_angle"] = angle

    if str(params.get("filter", "wiener")) == "inverse":
        eps = float(params.get("eps", 1e-3))
        out = restore.inverse_filter(img, psf, eps=eps)
        note += f", 逆滤波 eps={eps:g}"
    else:
        k = float(params.get("K", 0.01))
        out = restore.wiener(img, psf, K=k)
        note += f", 维纳 K={k:g}"
    return out, note, metrics


def stage_sharpen(
    img: Image, params: dict
) -> tuple[Image, str, dict[str, float]]:
    """USM 锐化阶段（在二值化之前补偿轻度离焦）。"""
    amount = float(params.get("amount", 1.0))
    radius = float(params.get("radius", 2.0))
    out = sharpen.unsharp_mask(img, radius=radius, amount=amount)
    return out, f"USM amount={amount:g}, σ={radius:g}", {}


WARP_SPEC: list[ParamSpec] = [
    NumParam("min_confidence", "角点置信度阈值", 0.6, 0.0, 1.0, 0.05),
]
ILLUM_SPEC: list[ParamSpec] = [
    NumParam("sigma", "光照场尺度", 30.0, 5.0, 120.0, 1.0),
]
WB_SPEC: list[ParamSpec] = [
    EnumParam(
        "method", "白平衡方法", "white_patch",
        (("white_patch", "白块估计"), ("gray_world", "灰世界")),
    ),
    NumParam("strength", "校正强度", 1.0, 0.0, 1.0, 0.05),
]
ENHANCE_SPEC: list[ParamSpec] = [
    EnumParam(
        "method", "增强方法", "clahe",
        (
            ("clahe", "CLAHE"),
            ("histeq", "直方图均衡"),
            ("stretch", "对比度拉伸"),
            ("none", "不增强（仅伽马）"),
        ),
    ),
    NumParam("clip_limit", "CLAHE 削波", 2.0, 0.1, 10.0, 0.1),
    NumParam("grid_size", "CLAHE 网格", 8, 2, 16, 1, integer=True),
    NumParam("gamma", "伽马", 1.0, 0.2, 3.0, 0.05),
]
DENOISE_SPEC: list[ParamSpec] = [
    EnumParam(
        "method", "去噪方法", "median",
        (("median", "中值滤波"), ("bilateral", "双边滤波")),
    ),
    NumParam("ksize", "中值窗口(奇数)", 3, 3, 9, 2, integer=True),
    NumParam("sigma_color", "双边色彩σ", 0.05, 0.01, 0.30, 0.01),
]
BINARIZE_SPEC: list[ParamSpec] = [
    EnumParam(
        "method", "二值化方法", "adaptive",
        (("adaptive", "自适应阈值"), ("otsu", "全局 Otsu"), ("sauvola", "Sauvola")),
    ),
    NumParam("block_size", "邻域边长(奇数)", 31, 3, 151, 2, integer=True),
    NumParam("c", "阈值偏置 C", 10.0, 0.0, 40.0, 0.5),
    NumParam("window", "Sauvola 窗口(奇数)", 25, 7, 101, 2, integer=True),
    NumParam("k", "Sauvola 敏感度 k", 0.2, 0.05, 0.6, 0.01),
    NumParam("min_area", "最小连通域", 8, 0, 200, 1, integer=True),
]
FREQ_SPEC: list[ParamSpec] = [
    EnumParam(
        "kind", "陷波器类型", "butterworth",
        (("butterworth", "Butterworth"), ("ideal", "理想陷波")),
    ),
    NumParam("top_n", "自动找峰数量", 4, 1, 12, 1, integer=True),
    NumParam("radius", "陷波半径(0=自适应)", 0, 0, 40, 1, integer=True),
    NumParam("order", "Butterworth 阶数", 4.0, 1.0, 16.0, 1.0),
    NumParam("dc_radius", "DC 屏蔽半径(0=自适应)", 0, 0, 96, 1, integer=True),
]
RESTORE_SPEC: list[ParamSpec] = [
    EnumParam(
        "psf_type", "PSF 类型", "motion",
        (("motion", "运动模糊"), ("defocus", "散焦")),
    ),
    EnumParam(
        "psf_mode", "PSF 来源", "est",
        (("est", "倒谱自动估计"), ("manual", "手动指定")),
    ),
    NumParam("length", "模糊长度(手动)", 9.0, 1.0, 51.0, 1.0, integer=True),
    NumParam("angle_deg", "模糊角度(手动)", 0.0, -90.0, 90.0, 1.0),
    NumParam("defocus_radius", "散焦半径(像素)", 5, 1, 20, 1, integer=True),
    EnumParam(
        "filter", "复原滤波器", "wiener",
        (("wiener", "维纳滤波"), ("inverse", "逆滤波")),
    ),
    NumParam("K", "维纳噪信比", 0.01, 1e-5, 1.0, 0.001),
    NumParam("eps", "逆滤波正则 ε", 1e-3, 1e-6, 0.5, 0.001),
]
SHARPEN_SPEC: list[ParamSpec] = [
    NumParam("amount", "锐化强度", 1.0, 0.0, 3.0, 0.1),
    NumParam("radius", "模糊半径σ", 2.0, 0.5, 8.0, 0.5),
]


def build_default_stages() -> list[Stage]:
    """默认处理链：校正 → 光照 → 白平衡 → 增强 → 去噪 → 去网纹 → 复原 → 锐化 → 二值化。

    去噪排在增强之后、去网纹与复原之前：先让噪声在增强阶段不被同步放大，
    再让后续的陷波与维纳滤波在干净一点的输入上工作（维纳本身会放大噪声）。
    除校正、光照归一、二值化外均默认关闭，按需在 UI 勾选。
    """
    stages = [
        Stage("warp", "几何校正", stage_warp, WARP_SPEC),
        Stage("illumination", "光照归一", stage_illumination, ILLUM_SPEC),
        Stage("white_balance", "白平衡", stage_white_balance, WB_SPEC, enabled=False),
        Stage("enhance", "对比度增强", stage_enhance, ENHANCE_SPEC, enabled=False),
        Stage("denoise", "去噪", stage_denoise, DENOISE_SPEC, enabled=False),
        Stage("freq", "去网纹", stage_freq_denoise, FREQ_SPEC, enabled=False),
        Stage("restore", "退化复原", stage_restore, RESTORE_SPEC, enabled=False),
        Stage("sharpen", "锐化", stage_sharpen, SHARPEN_SPEC, enabled=False),
        Stage("binarize", "二值化", stage_binarize, BINARIZE_SPEC),
    ]
    for s in stages:
        s.reset_params()
    return stages


def make_preview(image: Image, max_side: int = 900) -> Image:
    """生成 UI 预览用缩略图（uint8 BGR）。"""
    arr = to_uint8(image)
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return arr
    scale = max_side / longest
    return cv2.resize(
        arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
