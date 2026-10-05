"""
Ядро стереозрения:
- ректификация пары,
- вычисление диспаратности (BM и SGBM),
- постобработка WLS,
- перевод в глубину через Q-матрицу.
"""
import cv2
import numpy as np


class StereoMatcher:
    def __init__(self, calib, method="sgbm", num_disp=128, block=5,
                 use_wls=True, use_clahe=True):
        """
        method: 'bm' или 'sgbm'
        num_disp: кратно 16
        """
        self.calib = calib
        self.method = method
        self.num_disp = num_disp
        self.block = block
        self.use_wls = use_wls
        self.use_clahe = use_clahe
        self._matcher = self._build_matcher()

    # ---------- построение матчера ----------
    def _build_matcher(self):
        if self.method == "bm":
            m = cv2.StereoBM_create(numDisparities=self.num_disp,
                                    blockSize=self.block)
            m.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
            m.setPreFilterSize(9)
            m.setPreFilterCap(31)
            m.setTextureThreshold(10)
            m.setMinDisparity(0)
            m.setUniquenessRatio(15)
            m.setSpeckleWindowSize(100)
            m.setSpeckleRange(32)
            m.setDisp12MaxDiff(1)
            return m

        # SGBM
        channels = 1
        m = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disp,
            blockSize=self.block,
            P1=8 * channels * self.block ** 2,
            P2=32 * channels * self.block ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        return m

    # ---------- предобработка ----------
    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    # ---------- ректификация ----------
    def rectify(self, img_l, img_r):
        c = self.calib
        size = (img_l.shape[1], img_l.shape[0])
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            c["K_l"], c["dist_l"], c["R_l"], c["P_l"], size, cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            c["K_r"], c["dist_r"], c["R_r"], c["P_r"], size, cv2.CV_32FC1)
        rec_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rec_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)
        return rec_l, rec_r

    # ---------- основная функция ----------
    def compute(self, img_l, img_r, raw=False):
        """
        Возвращает dict: disparity (float32, в пикселях),
                         disparity_raw, depth (в метрах), mask.
        """
        rec_l, rec_r = self.rectify(img_l, img_r)
        g_l = self._preprocess(rec_l)
        g_r = self._preprocess(rec_r)

        disp = self._matcher.compute(g_l, g_r).astype(np.float32) / 16.0
        disp_raw = disp.copy()

        if self.use_wls and self.method == "sgbm":
            disp = self._apply_wls(g_l, g_r, disp)
        else:
            disp = self._median_filter(disp)

        depth, mask = self._disparity_to_depth(disp)
        return {
            "rect_left": rec_l, "rect_right": rec_r,
            "gray_left": g_l, "gray_right": g_r,
            "disparity_raw": disp_raw,
            "disparity": disp,
            "depth": depth,
            "mask": mask,
        }

    def _apply_wls(self, g_l, g_r, disp_left):
        right_matcher = cv2.ximgproc.createRightMatcher(self._matcher)
        disp_right = right_matcher.compute(g_r, g_l).astype(np.float32) / 16.0

        wls = cv2.ximgproc.createDisparityWLSFilter(self._matcher)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        wls.setDepthDiscontinuityRadius(3)
        wls.setLRCthresh(24)

        filtered = wls.filter(disp_left, g_l,
                              disparity_map_right=disp_right)
        return filtered

    @staticmethod
    def _median_filter(disp):
        d = disp.copy()
        d[d < 0] = 0
        return cv2.medianBlur(d.astype(np.float32), 5)

    def _disparity_to_depth(self, disp, min_disp=1.0):
        f = self.calib["focal"]
        B = self.calib["baseline"]
        depth = np.zeros_like(disp, dtype=np.float32)
        mask = disp > min_disp
        depth[mask] = (f * B) / disp[mask]
        # отсекаем выбросы
        mask &= depth < np.percentile(depth[mask], 99) if mask.any() else mask
        depth[~mask] = 0
        return depth, mask.astype(np.uint8) * 255

    # ---------- 3D-облако ----------
    def to_point_cloud(self, disp, img_l):
        """Возвращает Nx3 XYZ и Nx3 RGB через Q-матрицу."""
        Q = self.calib["Q"]
        points = cv2.reprojectImageTo3D(disp.astype(np.float32), Q)
        mask = disp > disp.min() + 1
        xyz = points[mask]
        rgb = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)[mask]
        return xyz, rgb
