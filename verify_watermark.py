"""
Standalone script to verify if an invisible watermark (DWT-DCT) is present in an image.
Usage: Modify the paths below and run the script.
"""

import numpy as np
import pywt
from PIL import Image
from scipy.fftpack import dct
import os
import argparse
from PIL import ImageDraw, ImageFont
from typing import Optional, Tuple  # Python 3.9 compatible

# --- CONFIGURATION ---
WATERMARKED_IMAGE = r'd:\WatermarkGAN\WatermarkDCT_DWT\result\watermarked.png'  # Image to verify (raw string)
ORIGINAL_WATERMARK = None  # Optional path to original watermark image
TEXT_ORIGINAL = "Nandan Upadhyaya"  # Optional text watermark to verify
WATERMARK_SIZE = 11  # MUST match test.py's capacity-matched size (11 for redundancy=8)
MODEL = 'haar'
LEVEL = 1
REDUNDANCY = 8  # MUST match test.py
DETECTION_THRESHOLD = 0.85  # INCREASED: Much stricter threshold for watermark presence
# NEW: Blind detection thresholds
MER_POS = (5, 5)  # position used by embedder in each 8x8 block
MER_COMP_POS = [(3, 4), (4, 3), (4, 4), (5, 3), (3, 5)]  # nearby mid-band coeffs
MER_MIN_RATIO = 1.10  # mean(|c55|) must be 10% higher than neighbors on average
MER_Z_MIN = 1.75      # z-score threshold for significance


def apply_dct(image_array):
    """Applies 8x8 block-wise DCT."""
    size = len(image_array)
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct


def extract_watermark_dwt_dct(Y_channel, wm_size):
    """
    FIXED: Match test.py's extraction logic - pairwise majority vote with redundancy.
    Returns map in {-1,+1} of shape (wm_size, wm_size)
    """
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    h, w = dct_ll.shape
    votes = np.zeros((wm_size * wm_size,), dtype=np.int32)

    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bit_index = idx // REDUNDANCY
            if bit_index >= votes.size:
                break
            blk = dct_ll[i:i+8, j:j+8]
            # FIXED: Use (3,4) - (4,3) pairwise comparison like test.py
            diff = blk[3, 4] - blk[4, 3]
            votes[bit_index] += 1 if diff >= 0 else -1
            idx += 1
        if (idx // REDUNDANCY) >= votes.size:
            break

    bits = np.where(votes >= 0, 1.0, -1.0)
    return bits.reshape(wm_size, wm_size)


def _infer_wm_size_from_image(image_path):
    """Infer square watermark side from LL capacity with redundancy."""
    img = Image.open(image_path).convert('RGB')
    first_channel = np.array(img, dtype=np.float64)[:, :, 0]
    LL = pywt.wavedec2(data=first_channel, wavelet=MODEL, level=LEVEL)[0]
    h, w = LL.shape
    blocks_h, blocks_w = (h // 8), (w // 8)
    cap_blocks = blocks_h * blocks_w
    # Account for redundancy (same as test.py)
    s = int(np.floor(np.sqrt(max(1, cap_blocks // REDUNDANCY))))
    return max(8, s)


def _create_text_watermark(text, size, font_size=None):
    """
    FIXED: Render centered text, then BINARIZE to {-1,+1} like test.py's _prepare_capacity_and_wm
    """
    if font_size is None:
        font_size = max(12, size // 6)
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)
    x = (size - tw) // 2
    y = (size - th) // 2
    draw.text((x, y), text, fill=255, font=font)
    
    # FIXED: Binarize to {-1,+1} using median threshold (matching test.py)
    wm_arr = np.array(img, dtype=np.float64)
    thr = float(np.median(wm_arr))
    bits01 = (wm_arr >= thr).astype(np.uint8)
    ref_map = (bits01 * 2 - 1).astype(np.float64)  # {-1, +1}
    return ref_map


def extract_watermark_from_image(image_path, wm_size=None):
    """Extract watermark from an image file."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float64)
    first_channel = img_array[:, :, 0]
    
    # Infer size with redundancy if not provided
    if wm_size is None:
        wm_size = _infer_wm_size_from_image(image_path)
    
    # FIXED: Use the new extraction function matching test.py
    watermark = extract_watermark_dwt_dct(first_channel, wm_size)
    return watermark, wm_size


def measure_ncc(original_wm, recovered_wm):
    """
    FIXED: Use test.py's robust NCC on {-1,+1} maps.
    """
    a = np.sign(np.asarray(original_wm, dtype=np.float64).flatten())
    b = np.sign(np.asarray(recovered_wm, dtype=np.float64).flatten())
    # Ensure {-1,+1}
    a[a == 0] = 1.0
    b[b == 0] = 1.0
    # Correlation-like score in [-1, 1]
    return float(np.mean(a * b))


# NEW: Blind MER detector
def _compute_mer_scores(dct_ll):
    """
    Compute mid-band energy ratio (MER) and z-score across 8x8 blocks in DWT-LL.
    Returns (ratio, z_score, num_blocks).
    """
    H, W = dct_ll.shape
    mb, nb = H // 8, W // 8
    c55_vals = []
    comp_vals = []
    for bi in range(mb):
        for bj in range(nb):
            by, bx = bi * 8, bj * 8
            block = dct_ll[by:by+8, bx:bx+8]
            c55_vals.append(abs(block[MER_POS[0], MER_POS[1]]))
            neigh = [abs(block[p[0], p[1]]) for p in MER_COMP_POS]
            comp_vals.append(np.mean(neigh))
    c55_vals = np.array(c55_vals, dtype=np.float64)
    comp_vals = np.array(comp_vals, dtype=np.float64) + 1e-8
    ratio = float(np.mean(c55_vals / comp_vals))
    diff = c55_vals - comp_vals
    z = float((np.mean(diff)) / (np.std(diff) + 1e-8))
    return ratio, z, len(c55_vals)


def _extract_ll_dct_first_channel(image_path):
    """
    Helper: load image, take first channel, DWT-LL, then 8x8 DCT of LL.
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float64)
    first_channel = img_array[:, :, 0]
    coeffs = pywt.wavedec2(data=first_channel, wavelet=MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)
    return dct_ll, LL.shape


def verify_watermark(watermarked_image_path, original_watermark_path=None,
                     original_text=None,
                     wm_size=None, ncc_threshold=DETECTION_THRESHOLD,
                     use_blind=True, mer_ratio_min=MER_MIN_RATIO, mer_z_min=MER_Z_MIN):
    """
    Verify if watermark is present in the image.
    Returns (present: bool, details: dict)
    """
    print("="*60)
    print("        INVISIBLE WATERMARK VERIFICATION (DWT-DCT)")
    print("="*60)

    # Normalize paths to avoid issues with backslash escape sequences
    watermarked_image_path = os.path.abspath(os.path.normpath(str(watermarked_image_path)))
    if original_watermark_path:
        original_watermark_path = os.path.abspath(os.path.normpath(str(original_watermark_path)))

    if not os.path.exists(watermarked_image_path):
        print(f"❌ ERROR: Image not found: {watermarked_image_path}")
        return False, {"error": "image_not_found"}

    print(f"\n📄 Verifying Image: {watermarked_image_path}")

    verdicts = []
    details = {"ncc": None, "ncc_threshold": ncc_threshold, "blind_mer_ratio": None, "blind_mer_z": None, "wm_size": None}

    # If we have a reference (image path or text), do NCC-based decision
    if (original_watermark_path and os.path.exists(original_watermark_path)) or (original_text and len(str(original_text).strip()) > 0):
        try:
            recovered_wm, inferred_size = extract_watermark_from_image(watermarked_image_path, wm_size=wm_size)
            details["wm_size"] = inferred_size
            print("✓ Watermark extraction successful")
            print(f"  Recovered watermark shape: {recovered_wm.shape}")
            # Save recovered for inspection
            os.makedirs('./result', exist_ok=True)
            # FIXED: Recovered is {-1,+1}, convert to [0,255] for visualization
            recovered_normalized = ((recovered_wm + 1.0) * 0.5 * 255.0).astype(np.uint8)
            Image.fromarray(recovered_normalized).save('./result/recovered_watermark.jpg')

            # Build reference watermark (image path or rendered text)
            if original_watermark_path and os.path.exists(original_watermark_path):
                orig_img = Image.open(original_watermark_path).convert('L')
                orig_img = orig_img.resize((inferred_size, inferred_size), Image.Resampling.LANCZOS)
                # FIXED: Binarize image watermark to {-1,+1}
                orig_arr_gray = np.array(orig_img, dtype=np.float64)
                thr = float(np.median(orig_arr_gray))
                bits01 = (orig_arr_gray >= thr).astype(np.uint8)
                orig_array = (bits01 * 2 - 1).astype(np.float64)
                print(f"✓ Using original watermark image as reference")
            else:
                # Already returns {-1,+1}
                orig_array = _create_text_watermark(str(original_text).strip(), inferred_size)
                print(f"✓ Using rendered text as reference: \"{str(original_text).strip()}\"")

            # FIXED: Use the robust measure_ncc matching test.py
            ncc_score = measure_ncc(orig_array, recovered_wm)
            details["ncc"] = float(ncc_score)
            print(f"→ NCC with original: {ncc_score:.4f} (threshold {ncc_threshold:.2f})")
            
            # FIXED: More accurate interpretation with confidence gaps
            confidence_gap = ncc_score - ncc_threshold
            
            if ncc_score >= 0.95:
                print("  ✅ PERFECT MATCH - Watermark is definitively present")
            elif ncc_score >= ncc_threshold:
                if confidence_gap >= 0.1:
                    print("  ✅ STRONG MATCH - Watermark is clearly present")
                else:
                    print(f"  ✅ THRESHOLD MATCH - Watermark detected (confidence gap: +{confidence_gap:.3f})")
            elif ncc_score >= ncc_threshold - 0.05:  # Close to threshold
                print(f"  ⚠️  BORDERLINE - Just below threshold (gap: {confidence_gap:.3f})")
            elif ncc_score >= 0.5:
                print(f"  ❌ WEAK CORRELATION - Likely no watermark (gap: {confidence_gap:.3f})")
            else:
                print(f"  ❌ NO CORRELATION - No watermark detected (gap: {confidence_gap:.3f})")
            
            verdicts.append(ncc_score >= ncc_threshold)

            # Save comparison image
            # FIXED: orig_array is {-1,+1}, convert to [0,255]
            orig_norm = ((orig_array + 1.0) * 0.5 * 255.0).astype(np.uint8)
            comparison = np.hstack([orig_norm, recovered_normalized])
            Image.fromarray(comparison).save('./result/watermark_verification.jpg')
            print("✓ Saved: ./result/watermark_verification.jpg")

        except Exception as e:
            print(f"❌ ERROR during reference-based verification: {e}")
            import traceback
            traceback.print_exc()
            verdicts.append(False)

    # Blind check (MER): useful when no original provided
    if use_blind:
        try:
            dct_ll, _ = _extract_ll_dct_first_channel(watermarked_image_path)
            mer_ratio, mer_z, nb = _compute_mer_scores(dct_ll)
            details["blind_mer_ratio"] = float(mer_ratio)
            details["blind_mer_z"] = float(mer_z)
            
            
            # Blind detection verdict
            blind_detected = (mer_ratio >= mer_ratio_min) and (mer_z >= mer_z_min)
           
            
            verdicts.append(blind_detected)
        except Exception as e:
            print(f"❌ ERROR during blind MER detection: {e}")
            import traceback
            traceback.print_exc()
            verdicts.append(False)

    present = any(verdicts) if verdicts else False
    print("\n" + "-"*60)
    if present:
        print("✅ WATERMARK PRESENT: YES")
        ncc = details["ncc"]
        if ncc and ncc >= 0.95:
            print("   ✅ VERY HIGH CONFIDENCE - Perfect watermark recovery")
        elif ncc and ncc >= ncc_threshold + 0.1:
            print("   ✅ HIGH CONFIDENCE - Strong watermark detection")
        elif ncc and ncc >= ncc_threshold:
            print("   ✅ MEDIUM CONFIDENCE - Watermark detected above threshold")
        else:
            print("   ⚠️  LOW CONFIDENCE - Detection based on weak signals")
    else:
        print("❌ WATERMARK PRESENT: NO")
        ncc = details["ncc"]
        if ncc and ncc >= ncc_threshold - 0.05:
            print(f"   ⚠️  BORDERLINE CASE - Score {ncc:.4f} is close to threshold {ncc_threshold:.2f}")
            print("   🔍 Consider manual inspection or adjusting threshold")
        elif ncc and ncc >= 0.5:
            print(f"   ❌ CLEAR NEGATIVE - Score {ncc:.4f} indicates no watermark")
        else:
            print(f"   ❌ DEFINITIVE NEGATIVE - Score {ncc:.4f} shows no correlation")
    print("-"*60)
    print("\n" + "="*60)
    print("           VERIFICATION COMPLETE")
    print("="*60)
    return present, details


if __name__ == "__main__":
    # CLI for presence check
    parser = argparse.ArgumentParser(description="Verify if DWT-DCT watermark is present")
    parser.add_argument("--image", "-i", default=WATERMARKED_IMAGE, help="Path to image to verify")
    parser.add_argument("--original", "-o", default=None, help="Path to original watermark image (optional)")
    parser.add_argument("--text", "-t", default=None, help="Expected text watermark to verify (optional)")
    parser.add_argument("--wm_size", "-s", type=int, default=None, help="Watermark size side (auto if omitted)")
    parser.add_argument("--ncc_th", type=float, default=DETECTION_THRESHOLD, help="NCC threshold (default 0.85 for strict detection)")
    parser.add_argument("--no_blind", action="store_true", help="Disable blind MER detection")
    parser.add_argument("--mer_ratio", type=float, default=MER_MIN_RATIO, help="Blind MER min ratio")
    parser.add_argument("--mer_z", type=float, default=MER_Z_MIN, help="Blind MER min z-score")
    args = parser.parse_args()

    verify_watermark(
        args.image,
        original_watermark_path=args.original,
        original_text=args.text if args.text else TEXT_ORIGINAL if ORIGINAL_WATERMARK is None else None,
        wm_size=args.wm_size,
        ncc_threshold=args.ncc_th,
        use_blind=not args.no_blind,
        mer_ratio_min=args.mer_ratio,
        mer_z_min=args.mer_z
    )
