"""
Генерация всех иллюстраций для отчёта по варианту 22
(построение карты глубины по стереопаре).

Запуск:
    py report_figures.py --left data/left.png --right data/right.png --out results/figures

Опционально с ground truth:
    py report_figures.py --left data/left.png --right data/right.png \
                         --gt data/middlebury/disp0.pgm --out results/figures
"""
import argparse
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ---------- параметры Middlebury для Teddy ----------
FOCAL = 3740.0
BASELINE = 160.0


# ---------- ядро ----------
def build_matcher(method="sgbm", num_disp=64, block=5):
    if method == "bm":
        m = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
        m.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        m.setPreFilterSize(9); m.setPreFilterCap(31)
        m.setTextureThreshold(10); m.setUniquenessRatio(15)
        m.setSpeckleWindowSize(100); m.setSpeckleRange(32)
        return m
    ch = 1
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8*ch*block**2, P2=32*ch*block**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disp(g_l, g_r, matcher, use_wls=False):
    disp = matcher.compute(g_l, g_r).astype(np.float32) / 16.0
    raw = disp.copy()
    if use_wls and hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createRightMatcher"):
        rm = cv2.ximgproc.createRightMatcher(matcher)
        d_r = rm.compute(g_r, g_l).astype(np.float32) / 16.0
        wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
        wls.setLambda(8000); wls.setSigmaColor(1.5)
        disp = wls.filter(disp, g_l, disparity_map_right=d_r)
    else:
        disp[disp < 0] = 0
        disp = cv2.medianBlur(disp, 5)
    return raw, disp


def colorize(disp, mask=None, cmap=cv2.COLORMAP_TURBO):
    d = disp.copy()
    if mask is not None:
        d[mask == 0] = 0
    d[d < 0] = 0
    vmax = np.percentile(d[d > 0], 99) if (d > 0).any() else 1.0
    d = np.clip(d / (vmax + 1e-6), 0, 1)
    img = (d * 255).astype(np.uint8)
    colored = cv2.applyColorMap(img, cmap)
    if mask is not None:
        colored[mask == 0] = (0, 0, 0)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_depth(disp, focal, baseline, min_disp=1.0):
    depth = np.zeros_like(disp, dtype=np.float32)
    mask = disp > min_disp
    depth[mask] = (focal * baseline) / disp[mask]
    return depth, mask.astype(np.uint8)


def bad_pixel(disp_pred, disp_gt, thr=1.0):
    mask = disp_gt > 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(disp_pred[mask] - disp_gt[mask]) > thr).mean() * 100)


# ---------- фигуры ----------
def fig1_stereopair(img_l, img_r, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img_l, cmap="gray"); ax[0].set_title("Левое изображение"); ax[0].axis("off")
    ax[1].imshow(img_r, cmap="gray"); ax[1].set_title("Правое изображение"); ax[1].axis("off")
    plt.suptitle("Рис. 1. Исходная стереопара (Middlebury Teddy)", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig1_stereopair.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig2_methods(img_l, g_l, g_r, out):
    """BM vs SGBM vs SGBM+median"""
    bm = build_matcher("bm", 64, 5)
    sgbm = build_matcher("sgbm", 64, 5)

    _, d_bm = compute_disp(g_l, g_r, bm, use_wls=False)
    _, d_sgbm_raw = compute_disp(g_l, g_r, sgbm, use_wls=False)
    raw_sgbm, d_sgbm_wls = compute_disp(g_l, g_r, sgbm, use_wls=True)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(img_l, cmap="gray"); ax[0].set_title("Исходное"); ax[0].axis("off")
    ax[1].imshow(colorize(d_bm));   ax[1].set_title("BM (blockSize=5)"); ax[1].axis("off")
    ax[2].imshow(colorize(raw_sgbm)); ax[2].set_title("SGBM (raw)"); ax[2].axis("off")
    ax[3].imshow(colorize(d_sgbm_wls)); ax[3].set_title("SGBM + медиана"); ax[3].axis("off")
    plt.suptitle("Рис. 2. Сравнение методов стереосопоставления", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig2_methods.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig3_numdisp(g_l, g_r, out):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(g_l, cmap="gray"); ax[0].set_title("Исходное"); ax[0].axis("off")
    for i, nd in enumerate([16, 64, 128], start=1):
        m = build_matcher("sgbm", nd, 5)
        _, d = compute_disp(g_l, g_r, m, use_wls=False)
        ax[i].imshow(colorize(d))
        cov = float((d > 1.0).mean() * 100)
        ax[i].set_title(f"num_disp={nd}\nпокрытие {cov:.0f}%")
        ax[i].axis("off")
    plt.suptitle("Рис. 3. Влияние параметра num_disp", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig3_numdisp.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig4_blocksize(g_l, g_r, out):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(g_l, cmap="gray"); ax[0].set_title("Исходное"); ax[0].axis("off")
    for i, b in enumerate([5, 9, 15], start=1):
        m = build_matcher("sgbm", 64, b)
        _, d = compute_disp(g_l, g_r, m, use_wls=False)
        ax[i].imshow(colorize(d))
        ax[i].set_title(f"blockSize={b}")
        ax[i].axis("off")
    plt.suptitle("Рис. 4. Влияние размера блока", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig4_blocksize.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig5_depth_hist(img_l, disp, depth, mask, out):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_l, cmap="gray"); ax1.set_title("Исходное"); ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(colorize(disp)); ax2.set_title("Карта диспаратности"); ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(colorize(depth, mask, cv2.COLORMAP_JET))
    ax3.set_title("Карта глубины (JET)"); ax3.axis("off")
    ax4 = fig.add_subplot(2, 2, 4)
    vals = disp[mask > 0]
    ax4.hist(vals, bins=60, color="steelblue", edgecolor="black")
    ax4.set_xlabel("Диспаратность, px"); ax4.set_ylabel("Число пикселей")
    ax4.set_title(f"Гистограмма диспаратности (покрытие {100*mask.mean():.0f}%)")
    ax4.grid(alpha=0.3)
    plt.suptitle("Рис. 5. Карта глубины и распределение диспаратности", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig5_depth_hist.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig6_pointcloud(img_rgb, disp, out, Q_scale=None):
    """3D-облако точек через Q-матрицу Middlebury (упрощённо)."""
    h, w = disp.shape
    # Q-матрица для Middlebury Teddy
    Q = np.float32([
        [1, 0, 0, -w / 2],
        [0, -1, 0, h / 2],
        [0, 0, 0, FOCAL],
        [0, 0, 1 / BASELINE, 0],
    ])
    points = cv2.reprojectImageTo3D(disp, Q)
    mask = disp > 1.0
    xyz = points[mask]
    rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)[mask]

    # прореживаем для скорости
    if len(xyz) > 120000:
        idx = np.random.choice(len(xyz), 120000, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=rgb / 255.0, s=0.4, marker=".")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z (глубина)")
    ax.invert_yaxis()
    ax.view_init(elev=-70, azim=-90)
    ax.set_title("Рис. 6. 3D-облако точек сцены (цвет — из исходного изображения)")
    plt.tight_layout()
    plt.savefig(f"{out}/fig6_pointcloud.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig7_gt_compare(disp, gt, out, thr=1.0):
    """Сравнение с ground truth + карта ошибок."""
    if gt is None:
        return
    if gt.dtype == np.uint16:
        gt = gt.astype(np.float32) / 16.0
    else:
        gt = gt.astype(np.float32)
    if gt.shape != disp.shape:
        gt = cv2.resize(gt, (disp.shape[1], disp.shape[0]),
                        interpolation=cv2.INTER_NEAREST)

    valid = gt > 0
    err = np.abs(disp - gt)
    err[~valid] = 0

    bpr = bad_pixel(disp, gt, thr)
    rmse = float(np.sqrt((err[valid] ** 2).mean()))

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(colorize(gt, valid)); ax[0].set_title("Ground truth"); ax[0].axis("off")
    ax[1].imshow(colorize(disp));      ax[1].set_title("Наш результат"); ax[1].axis("off")
    im = ax[2].imshow(err, cmap="hot", vmin=0, vmax=10)
    ax[2].set_title(f"Карта ошибок\nBad-pixel({thr})={bpr:.2f}%, RMSE={rmse:.2f}")
    ax[2].axis("off"); plt.colorbar(im, ax=ax[2], fraction=0.046)

    # распределение ошибки
    ax[3].hist(err[valid], bins=80, color="indianred", edgecolor="black")
    ax[3].axvline(thr, color="k", linestyle="--", label=f"порог {thr}")
    ax[3].set_xlabel("|pred − gt|, px"); ax[3].set_ylabel("Число пикселей")
    ax[3].set_title("Распределение ошибки"); ax[3].grid(alpha=0.3); ax[3].legend()
    plt.suptitle("Рис. 7. Сравнение с ground truth Middlebury", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out}/fig7_gt_compare.png", dpi=200, bbox_inches="tight")
    plt.close()
    return bpr, rmse


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--out", default="results/figures")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    img_l = cv2.imread(args.left)
    img_r = cv2.imread(args.right)
    g_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    g_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    print(f"[i] Размер изображения: {g_l.shape}")

    # Рис. 1 — стереопара
    fig1_stereopair(g_l, g_r, args.out)
    print("[+] fig1_stereopair.png")

    # Рис. 2 — методы
    fig2_methods(img_l, g_l, g_r, args.out)
    print("[+] fig2_methods.png")

    # Рис. 3 — num_disp
    fig3_numdisp(g_l, g_r, args.out)
    print("[+] fig3_numdisp.png")

    # Рис. 4 — blockSize
    fig4_blocksize(g_l, g_r, args.out)
    print("[+] fig4_blocksize.png")

    # основной прогон для рис. 5, 6, 7
    m = build_matcher("sgbm", 64, 5)
    t0 = time.time()
    raw, disp = compute_disp(g_l, g_r, m, use_wls=True)
    dt = time.time() - t0
    depth, mask = to_depth(disp, FOCAL, BASELINE)
    print(f"[i] Основной прогон: {dt*1000:.0f} мс, покрытие {100*mask.mean():.1f}%")

    # Рис. 5 — глубина + гистограмма
    fig5_depth_hist(g_l, disp, depth, mask, args.out)
    print("[+] fig5_depth_hist.png")

    # Рис. 6 — 3D-облако
    fig6_pointcloud(img_l, disp, args.out)
    print("[+] fig6_pointcloud.png")

    # Рис. 7 — сравнение с GT (если есть)
    if args.gt and os.path.exists(args.gt):
        gt = cv2.imread(args.gt, cv2.IMREAD_UNCHANGED)
        if gt is not None:
            bpr, rmse = fig7_gt_compare(disp, gt, args.out)
            print(f"[+] fig7_gt_compare.png | Bad-pixel={bpr:.2f}%, RMSE={rmse:.2f} px")
    else:
        print("[!] Ground truth не найден, рис. 7 пропущен")

    print(f"\n[i] Все рисунки в: {args.out}/")


if __name__ == "__main__":
    main()
