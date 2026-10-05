"""
Метрики качества карты диспаратности.
"""
import numpy as np


def bad_pixel_rate(disp_pred, disp_gt, threshold=1.0, valid=None):
    """
    Доля пикселей, где |pred - gt| > threshold.
    valid: маска валидных пикселей (например, gt > 0).
    """
    if valid is None:
        valid = disp_gt > 0
    diff = np.abs(disp_pred[valid] - disp_gt[valid])
    return float((diff > threshold).mean() * 100.0)


def rmse(disp_pred, disp_gt, valid=None):
    if valid is None:
        valid = disp_gt > 0
    diff = disp_pred[valid] - disp_gt[valid]
    return float(np.sqrt((diff ** 2).mean()))


def coverage(mask):
    """Процент валидной маски."""
    return float((mask > 0).mean() * 100.0)


def report(disp_pred, disp_gt=None, mask=None, elapsed=None):
    lines = []
    if mask is not None:
        lines.append(f"Покрытие валидной маски: {coverage(mask):.2f}%")
    if disp_gt is not None:
        lines.append(f"Bad-pixel (1.0): {bad_pixel_rate(disp_pred, disp_gt):.3f}%")
        lines.append(f"Bad-pixel (2.0): {bad_pixel_rate(disp_pred, disp_gt, 2.0):.3f}%")
        lines.append(f"RMSE: {rmse(disp_pred, disp_gt):.3f} px")
    if elapsed is not None:
        lines.append(f"Время: {elapsed*1000:.1f} мс")
    return "\n".join(lines)
