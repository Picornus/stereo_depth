"""
Стерео → карта глубины для датасета Middlebury.
Не требует калибровки по шахматке: f и B берутся из параметров Middlebury.
Не требует модуля visualize.
"""
import argparse
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------- параметры Middlebury по умолчанию ----------
MIDDLEBURY_DEFAULTS = {
    "focal": 3740.0,
    "baseline": 160.0,
    "ndisp": 128,
}


def parse_args():
    p = argparse.ArgumentParser("Stereo depth (Middlebury)")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--gt", default=None,
                   help="ground truth диспаратность (например, disp0.pgm)")
    p.add_argument("--focal", type=float, default=MIDDLEBURY_DEFAULTS["focal"])
    p.add_argument("--baseline", type=float, default=MIDDLEBURY_DEFAULTS["baseline"])
    p.add_argument("--num_disp", type=int, default=MIDDLEBURY_DEFAULTS["ndisp"])
    p.add_argument("--block", type=int, default=5)
    p.add_argument("--method", choices=["bm", "sgbm"], default="sgbm")
    p.add_argument("--no_wls", action="store_true")
    p.add_argument("--out", default="results")
    return p.parse_args()


def build_matcher(method, num_disp, block):
    if method == "bm":
        m = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
        m.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        m.setPreFilterSize(9)
        m.setPreFilterCap(31)
        m.setTextureThreshold(10)
        m.setUniquenessRatio(15)
        m.setSpeckleWindowSize(100)
        m.setSpeckleRange(32)
        return m

    ch = 1
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * ch * block ** 2,
        P2=32 * ch * block ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(gray_l, gray_r, matcher, use_wls=True):
    disp = matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
    raw = disp.copy()

    if use_wls and hasattr(cv2, "ximgproc"):
        right_m = cv2.ximgproc.createRightMatcher(matcher)
        disp_r = right_m.compute(gray_r, gray_l).astype(np.float32) / 16.0
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        disp = wls.filter(disp, gray_l, disparity_map_right=disp_r)
    else:
        disp[disp < 0] = 0
        disp = cv2.medianBlur(disp, 5)
    return raw, disp


def disparity_to_depth(disp, focal, baseline, min_disp=1.0):
    depth = np.zeros_like(disp, dtype=np.float32)
    mask = disp > min_disp
    depth[mask] = (focal * baseline) / disp[mask]
    return depth, mask.astype(np.uint8)


def colorize(disp, mask=None):
    d = disp.copy()
    if mask is not None:
        d[mask == 0] = 0
    d[d < 0] = 0
    vmax = np.percentile(d[d > 0], 99) if (d > 0).any() else 1.0
    d = np.clip(d / (vmax + 1e-6), 0, 1)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def bad_pixel(disp_pred, disp_gt, thr=1.0):
    mask = disp_gt > 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(disp_pred[mask] - disp_gt[mask]) > thr).mean() * 100)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    assert img_l is not None, f"Не читается {args.left}"
    assert img_r is not None, f"Не читается {args.right}"
    print(f"[i] Размер: {img_l.shape}")

    matcher = build_matcher(args.method, args.num_disp, args.block)

    t0 = time.time()
    raw, disp = compute_disparity(img_l, img_r, matcher, use_wls=not args.no_wls)
    elapsed = time.time() - t0

    depth, mask = disparity_to_depth(disp, args.focal, args.baseline)

    # метрики
    print(f"[i] Время: {elapsed*1000:.0f} мс")
    print(f"[i] Покрытие маски: {100*mask.mean():.1f}%")
    if (mask > 0).any():
        print(f"[i] Диспаратность: min={disp[mask>0].min():.1f}, "
              f"max={disp[mask>0].max():.1f}")
        print(f"[i] Глубина: min={depth[mask>0].min():.1f}, "
              f"max={depth[mask>0].max():.1f} (в единицах baseline)")

    if args.gt:
        gt = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
        if gt is not None:
            gt = gt.astype(np.float32) / 16.0 if gt.dtype == np.uint16 else gt.astype(np.float32)
            if gt.shape == disp.shape:
                print(f"[i] Bad-pixel 1.0: {bad_pixel(disp, gt, 1.0):.2f}%")
                print(f"[i] Bad-pixel 2.0: {bad_pixel(disp, gt, 2.0):.2f}%")
            else:
                print(f"[!] GT размер {gt.shape} != disp {disp.shape}, пропуск метрик")

    # визуализация
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_l, cmap="gray"); ax[0].set_title("Левое"); ax[0].axis("off")
    ax[1].imshow(colorize(raw));      ax[1].set_title("Raw диспаратность"); ax[1].axis("off")
    ax[2].imshow(colorize(disp));     ax[2].set_title("После WLS/median"); ax[2].axis("off")
    ax[3].imshow(colorize(depth, mask)); ax[3].set_title("Глубина"); ax[3].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "result.png"), dpi=150)
    plt.show()

    # сохранить сырые карты
    cv2.imwrite(os.path.join(args.out, "disparity.png"), disp.astype(np.float32))
    cv2.imwrite(os.path.join(args.out, "depth.png"), depth.astype(np.float32))
    print(f"[i] Результаты в: {args.out}/")


if __name__ == "__main__":
    main()
