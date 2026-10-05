"""
Калибровка стереокамеры по снимкам шахматной доски.
Используется: findChessboardCorners + calibrateCamera + stereoCalibrate + stereoRectify.
"""
import cv2
import numpy as np
import glob
import os


class StereoCalibrator:
    def __init__(self, pattern_size=(9, 6), square_size=0.025):
        """
        pattern_size: внутренние углы шахматной доски (cols, rows)
        square_size:  размер клетки в метрах
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.objpoints = []   # 3D-точки в системе доски
        self.imgpoints_l = []
        self.imgpoints_r = []

        # 3D-координаты углов доски
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        self.objp = objp

    def add_images(self, left_dir, right_dir, show=False):
        left_files = sorted(glob.glob(os.path.join(left_dir, "*.png")) +
                            glob.glob(os.path.join(left_dir, "*.jpg")))
        right_files = sorted(glob.glob(os.path.join(right_dir, "*.png")) +
                             glob.glob(os.path.join(right_dir, "*.jpg")))
        assert len(left_files) == len(right_files), "Число кадров не совпадает"

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

        for lf, rf in zip(left_files, right_files):
            img_l = cv2.imread(lf)
            img_r = cv2.imread(rf)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            ok_l, corners_l = cv2.findChessboardCorners(gray_l, self.pattern_size, None)
            ok_r, corners_r = cv2.findChessboardCorners(gray_r, self.pattern_size, None)

            if ok_l and ok_r:
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                self.objpoints.append(self.objp)
                self.imgpoints_l.append(corners_l)
                self.imgpoints_r.append(corners_r)

                if show:
                    cv2.drawChessboardCorners(img_l, self.pattern_size, corners_l, ok_l)
                    cv2.drawChessboardCorners(img_r, self.pattern_size, corners_r, ok_r)
                    cv2.imshow("left", img_l)
                    cv2.imshow("right", img_r)
                    cv2.waitKey(200)

        print(f"[calib] Валидных пар: {len(self.objpoints)}")
        cv2.destroyAllWindows()

    def calibrate(self, image_size):
        """Возвращает словарь с параметрами калибровки."""
        ret_l, K_l, dist_l, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, image_size, None, None)
        ret_r, K_r, dist_r, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, image_size, None, None)

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            K_l, dist_l, K_r, dist_r, image_size,
            criteria=criteria, flags=flags)

        print(f"[calib] RMS error: {ret:.4f}")
        print(f"[calib] База (B): {np.linalg.norm(T):.4f} м")

        # Ректификация
        R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
            K_l, dist_l, K_r, dist_r, image_size, R, T,
            alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)

        return {
            "K_l": K_l, "dist_l": dist_l, "K_r": K_r, "dist_r": dist_r,
            "R": R, "T": T, "E": E, "F": F,
            "R_l": R_l, "R_r": R_r, "P_l": P_l, "P_r": P_r, "Q": Q,
            "roi_l": roi_l, "roi_r": roi_r,
            "baseline": float(np.linalg.norm(T)),
            "focal": float(P_l[0, 0]),
        }


def save_calibration(calib, path="calib.npz"):
    np.savez(path, **{k: v for k, v in calib.items()
                      if isinstance(v, (np.ndarray, float, int))})
    print(f"[calib] Сохранено: {path}")


def load_calibration(path="calib.npz"):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
