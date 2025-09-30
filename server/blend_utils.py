# blend_utils.py
import cv2
import numpy as np
from PIL import Image

def _to_bgr_uint8(img):
    """Convert PIL.Image or np.ndarray to BGR uint8 array."""
    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = img
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    # assume already BGR if ndarray
    return arr

def _fg_to_bgr_and_mask(fg):
    """Return (fg_bgr, mask) from PIL or ndarray; use alpha if present."""
    if isinstance(fg, Image.Image):
        if fg.mode == "RGBA":
            rgba = np.array(fg)
            fg_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            mask = (rgba[:, :, 3] > 0).astype(np.uint8) * 255
            return fg_bgr, mask
        rgb = np.array(fg.convert("RGB"))
        fg_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = fg_bgr.shape[:2]
        return fg_bgr, np.full((h, w), 255, dtype=np.uint8)
    # numpy input
    arr = fg
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        # assume RGBA in RGB order
        fg_bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        mask = (arr[:, :, 3] > 0).astype(np.uint8) * 255
        return fg_bgr, mask
    h, w = arr.shape[:2]
    return arr, np.full((h, w), 255, dtype=np.uint8)

def _fit_foreground_to_bg(bg_bgr: np.ndarray, fg_bgr: np.ndarray, mask: np.ndarray, x: int, y: int):
    """
    Ensure (x,y)+(fw,fh) fits in bg. If not, resize foreground and mask to fit available space.
    Returns (fg_bgr_resized, mask_resized, x_clamped, y_clamped).
    """
    H, W = bg_bgr.shape[:2]
    # Clamp top-left within image bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))

    fh, fw = fg_bgr.shape[:2]
    avail_w = max(1, W - x)
    avail_h = max(1, H - y)

    if fw > avail_w or fh > avail_h:
        scale = min(avail_w / max(1, fw), avail_h / max(1, fh))
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        fg_bgr = cv2.resize(fg_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # Ensure mask is single-channel uint8
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)

    return fg_bgr, mask, x, y

def _detect_blob_mask(bg_bgr: np.ndarray) -> np.ndarray:
    """
    Heuristic blob detector for DF-GAN artifacts:
    - High saturation and brightness in HSV
    - Cyan/green hue band often seen in artifacts (H ~ 80..140)
    Returns a binary mask (uint8) where 255 marks artifact pixels.
    """
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # High saturation and bright
    high_sat_bright = (s > 140) & (v > 160)
    # Cyan/green band (tuneable)
    cyan_band = (h >= 80) & (h <= 140) & (s > 110) & (v > 140)
    mask = (high_sat_bright | cyan_band).astype(np.uint8) * 255
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def _clean_dfgan_background(bg_bgr: np.ndarray, exclude_rect=None, strength: float = 1.0) -> np.ndarray:
    """
    Inpaint artifact blobs on DF-GAN background while preserving structure.
    exclude_rect: (x,y,w,h) region to keep (e.g., planned subject area).
    strength controls inpaint radius.
    """
    mask = _detect_blob_mask(bg_bgr)

    # Exclude planned subject area from cleaning if provided
    if exclude_rect is not None:
        x, y, w, h = exclude_rect
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(bg_bgr.shape[1], x + w), min(bg_bgr.shape[0], y + h)
        mask[y0:y1, x0:x1] = 0

    radius = max(1, int(3 * strength))
    cleaned = cv2.inpaint(bg_bgr, mask, radius, cv2.INPAINT_TELEA)
    return cleaned

# --- NEW: subject appearance matching (color, blur, noise) ---
def _reinhard_color_transfer(src_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    eps = 1e-6
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ms, ss = cv2.meanStdDev(src_lab.reshape(-1, 3))
    mr, sr = cv2.meanStdDev(ref_lab.reshape(-1, 3))
    ss = ss.flatten() + eps
    sr = sr.flatten()
    ms = ms.flatten()
    mr = mr.flatten()
    out = src_lab.copy()
    for c in range(3):
        out[:, :, c] = ((out[:, :, c] - ms[c]) * (sr[c] / ss[c])) + mr[c]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)

def _var_laplacian(img_bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

def _estimate_noise_sigma(img_bgr: np.ndarray) -> float:
    # crude noise estimate: std of high-frequency residual
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    return float(np.std(residual))

def _match_blur_and_noise(fg_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    fg = fg_bgr.copy()
    v_fg = _var_laplacian(fg_bgr)
    v_ref = _var_laplacian(ref_bgr)
    if v_fg > v_ref * 1.15:  # subject sharper than background
        sigma = 1.2 if v_fg < v_ref * 2.0 else 1.8
        fg = cv2.GaussianBlur(fg, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # match grain
    n_ref = _estimate_noise_sigma(ref_bgr)
    n_fg = _estimate_noise_sigma(fg)
    if n_ref > n_fg + 0.5:
        noise = np.random.normal(0, n_ref - n_fg, fg.shape).astype(np.float32)
        fg = np.clip(fg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return fg

def _match_subject_to_bg(fg_bgr: np.ndarray, bg_bgr: np.ndarray, x: int, y: int) -> np.ndarray:
    fh, fw = fg_bgr.shape[:2]
    H, W = bg_bgr.shape[:2]
    # local reference patch
    mx, my = max(8, fw // 6), max(8, fh // 6)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + fw + mx), min(H, y + fh + my)
    if x1 <= x0 + 5 or y1 <= y0 + 5:
        return fg_bgr
    ref = bg_bgr[y0:y1, x0:x1]
    try:
        fg_ct = _reinhard_color_transfer(fg_bgr, ref)
    except Exception:
        fg_ct = fg_bgr
    return _match_blur_and_noise(fg_ct, ref)

# --- NEW: soft feather for the subject mask using distance transform ---
def _feather_mask(mask: np.ndarray, feather_px: int = 6) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    # distance from edge inward
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    dist = np.clip(dist / max(1.0, feather_px), 0, 1)
    soft = (dist * 255).astype(np.uint8)
    return soft

# --- NEW: ROI helpers and multiband blending ---
def _roi_bounds(W: int, H: int, x: int, y: int, w: int, h: int, pad: int = 16):
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    return x0, y0, x1, y1

def _gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gp.append(img.astype(np.float32))
    return gp

def _laplacian_pyramid(gp: list) -> list:
    lp = [gp[-1]]
    for i in range(len(gp) - 1, 0, -1):
        size = (gp[i-1].shape[1], gp[i-1].shape[0])
        ge = cv2.pyrUp(gp[i], dstsize=size)
        lp.append(gp[i-1] - ge)
    return lp[::-1]

def _reconstruct_from_laplace(lp: list) -> np.ndarray:
    img = lp[-1]
    for i in range(len(lp) - 2, -1, -1):
        size = (lp[i].shape[1], lp[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lp[i]
    return np.clip(img, 0, 255).astype(np.uint8)

def _multiband_blend_roi(bg_roi: np.ndarray, fg_roi: np.ndarray, mask_roi: np.ndarray, levels: int = 4) -> np.ndarray:
    # normalize mask to [0,1] 3-ch
    m = (mask_roi.astype(np.float32) / 255.0)
    if m.ndim == 2:
        m3 = np.dstack([m, m, m])
    else:
        m3 = m[:, :, :1].repeat(3, axis=2)
    # pyramids
    gp_bg = _gaussian_pyramid(bg_roi, levels)
    gp_fg = _gaussian_pyramid(fg_roi, levels)
    gp_m = _gaussian_pyramid(m3, levels)
    lp_bg = _laplacian_pyramid(gp_bg)
    lp_fg = _laplacian_pyramid(gp_fg)
    blended_lp = []
    for lb, lf, gm in zip(lp_bg, lp_fg, gp_m):
        blended_lp.append(lb * (1.0 - gm) + lf * gm)
    return _reconstruct_from_laplace(blended_lp)

def blend_images(background, foreground, position=(50, 50)):
    """
    Poisson/multiband-blend foreground entity into background with artifact cleanup.
    Accepts PIL.Image or numpy arrays. Returns numpy array (RGB).
    """
    # NEW: BigGAN-only bypass — if foreground is a direct BigGAN image, skip blending
    if isinstance(foreground, Image.Image) and getattr(foreground, "_biggan_full_only", False):
        return np.array(foreground.convert("RGB"))

    bg_bgr = _to_bgr_uint8(background)
    fg_bgr, mask = _fg_to_bgr_and_mask(foreground)

    # Fit foreground within background at the given position
    x, y = int(position[0]), int(position[1])
    fg_bgr, mask, x, y = _fit_foreground_to_bg(bg_bgr, fg_bgr, mask, x, y)

    # Clean DF-GAN blobs on background, excluding the subject placement rect (slightly padded)
    pad = 12
    excl = (max(0, x - pad), max(0, y - pad),
            min(bg_bgr.shape[1] - x + pad, fg_bgr.shape[1] + 2 * pad),
            min(bg_bgr.shape[0] - y + pad, fg_bgr.shape[0] + 2 * pad))
    bg_bgr_clean = _clean_dfgan_background(bg_bgr, exclude_rect=excl, strength=1.0)

    # Match subject appearance to local road patch
    fg_bgr = _match_subject_to_bg(fg_bgr, bg_bgr_clean, x, y)

    # Feather the mask to reduce seams
    soft_mask = _feather_mask(mask, feather_px=8)

    # Multiband blend on a local ROI first
    fh, fw = fg_bgr.shape[:2]
    H, W = bg_bgr_clean.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(W, H, x, y, fw, fh, pad=16)
    roi_w, roi_h = x1 - x0, y1 - y0

    # Compose fg/mask canvases within ROI
    fg_canvas = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)
    mask_canvas = np.zeros((roi_h, roi_w), dtype=np.uint8)
    fx, fy = x - x0, y - y0
    fg_canvas[fy:fy+fh, fx:fx+fw] = fg_bgr
    mask_canvas[fy:fy+fh, fx:fx+fw] = soft_mask

    bg_roi = bg_bgr_clean[y0:y1, x0:x1]
    levels = max(2, min(5, int(np.floor(np.log2(max(1, min(roi_w, roi_h)))) - 3)))

    try:
        blended_roi = _multiband_blend_roi(bg_roi, fg_canvas, mask_canvas, levels=levels)
        out_bgr = bg_bgr_clean.copy()
        out_bgr[y0:y1, x0:x1] = blended_roi
        blended_bgr = out_bgr
    except Exception:
        # Fallback: Poisson (mixed) if multiband fails
        center = (x + fw // 2, y + fh // 2)
        blended_bgr = cv2.seamlessClone(fg_bgr, bg_bgr_clean, soft_mask, center, cv2.MIXED_CLONE)

    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    return blended_rgb  # return np.ndarray (H,W,3) RGB

def poisson_blend_rgba(bg_bgr: np.ndarray, fg_rgba: np.ndarray, position=(0, 0), mixed=False) -> np.ndarray:
    """
    Blend an RGBA foreground onto BGR background using seamlessClone.
    bg_bgr: (H,W,3) uint8 BGR
    fg_rgba: (h,w,4) uint8 RGBA
    """
    x, y = position
    fh, fw = fg_rgba.shape[:2]
    center = (x + fw // 2, y + fh // 2)
    fg_bgr = cv2.cvtColor(fg_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = fg_rgba[:, :, 3]
    mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    mode = cv2.MIXED_CLONE if mixed else cv2.NORMAL_CLONE
    return cv2.seamlessClone(fg_bgr, bg_bgr, mask, center, mode)

def inpaint_rect(bg_bgr: np.ndarray, rect, method="telea") -> np.ndarray:
    """
    Inpaint a rectangular region (x,y,w,h) on bg using Telea or NS methods.
    rect: (x, y, w, h)
    """
    x, y, w, h = rect
    mask = np.zeros(bg_bgr.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    algo = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(bg_bgr, mask, 3, algo)
