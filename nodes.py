import json
import math
from typing import Dict, List

import cv2
import numpy as np
import torch


class WorkflowNodeError(RuntimeError):
    """Error with actionable details for workflow tuning."""


def _to_numpy_rgb(image) -> np.ndarray:
    # ComfyUI IMAGE is usually torch.Tensor [B,H,W,C], but some integrations may pass numpy arrays.
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif not isinstance(image, np.ndarray):
        raise WorkflowNodeError(
            f"Expected IMAGE as torch.Tensor or numpy.ndarray, got: {type(image)}"
        )

    if image.ndim != 4:
        raise WorkflowNodeError(
            f"Expected ComfyUI image shape [B,H,W,C], got: {image.shape}. "
            "Use a node that outputs IMAGE tensors."
        )
    if image.shape[0] < 1 or image.shape[-1] != 3:
        raise WorkflowNodeError(
            f"Expected at least one RGB frame, got shape: {image.shape}"
        )
    arr = image[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _to_comfy_image(image: np.ndarray) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise WorkflowNodeError(f"Expected HWC RGB uint8 image, got {image.shape}")
    out = image.astype(np.float32) / 255.0
    out = np.expand_dims(out, axis=0)
    return torch.from_numpy(out)


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise WorkflowNodeError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def _line_to_homogeneous(line: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = line
    p1 = np.array([x1, y1, 1.0], dtype=np.float64)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64)
    l = np.cross(p1, p2)
    if np.linalg.norm(l[:2]) < 1e-8:
        raise WorkflowNodeError(f"Degenerate line found: {line}")
    return l


class ImagePreprocessNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "undistort": ("BOOLEAN", {"default": False}),
                "camera_matrix_json": ("STRING", {"default": "[[1000,0,512],[0,1000,512],[0,0,1]]"}),
                "dist_coeffs_json": ("STRING", {"default": "[0,0,0,0,0]"}),
                "to_gray": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess"
    CATEGORY = "Vision/Calibration"

    def preprocess(self, image, undistort, camera_matrix_json, dist_coeffs_json, to_gray):
        rgb = _to_numpy_rgb(image)
        try:
            if undistort:
                k = np.array(json.loads(camera_matrix_json), dtype=np.float64)
                d = np.array(json.loads(dist_coeffs_json), dtype=np.float64)
                rgb = cv2.undistort(rgb, k, d)
            if to_gray:
                g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            raise WorkflowNodeError(
                f"Preprocess failed. undistort={undistort}, to_gray={to_gray}, "
                f"camera_matrix_json={camera_matrix_json}, dist_coeffs_json={dist_coeffs_json}. "
                f"OpenCV error: {e}"
            )
        return (_to_comfy_image(rgb),)


class LineDetectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["lsd", "hough"],),
                "hough_threshold": ("INT", {"default": 80, "min": 1, "max": 500}),
                "hough_min_line_length": ("INT", {"default": 50, "min": 5, "max": 2000}),
                "hough_max_line_gap": ("INT", {"default": 10, "min": 1, "max": 200}),
            }
        }

    RETURN_TYPES = ("VW_LINES", "IMAGE", "INT")
    RETURN_NAMES = ("lines", "preview", "line_count")
    FUNCTION = "detect"
    CATEGORY = "Vision/Calibration"

    def detect(self, image, method, hough_threshold, hough_min_line_length, hough_max_line_gap):
        rgb = _to_numpy_rgb(image)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        lines: List[List[float]] = []
        if method == "lsd":
            lsd = cv2.createLineSegmentDetector()
            detected = lsd.detect(gray)[0]
            if detected is not None:
                for l in detected[:, 0, :]:
                    lines.append([float(v) for v in l])
        else:
            detected = cv2.HoughLinesP(
                gray,
                rho=1,
                theta=np.pi / 180.0,
                threshold=hough_threshold,
                minLineLength=hough_min_line_length,
                maxLineGap=hough_max_line_gap,
            )
            if detected is not None:
                for l in detected[:, 0, :]:
                    lines.append([float(v) for v in l])

        if len(lines) < 6:
            raise WorkflowNodeError(
                f"Too few lines detected ({len(lines)}). Try lowering Hough threshold or using LSD."
            )

        preview = rgb.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        return (lines, _to_comfy_image(preview), len(lines))


class LineClusteringNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lines": ("VW_LINES",),
                "angle_threshold_deg": ("FLOAT", {"default": 15.0, "min": 3.0, "max": 60.0}),
            }
        }

    RETURN_TYPES = ("VW_CLUSTERS", "STRING")
    RETURN_NAMES = ("clusters", "cluster_debug_json")
    FUNCTION = "cluster"
    CATEGORY = "Vision/Calibration"

    def cluster(self, lines, angle_threshold_deg):
        if not isinstance(lines, list) or len(lines) < 6:
            raise WorkflowNodeError("Invalid lines input: expect list with >= 6 lines.")

        # simple orientation clustering with kmeans (3 Manhattan directions)
        dirs = []
        for l in lines:
            x1, y1, x2, y2 = l
            v = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            try:
                v = _normalize(v)
            except WorkflowNodeError:
                continue
            dirs.append(v)

        if len(dirs) < 6:
            raise WorkflowNodeError("Not enough valid lines after removing degenerate ones.")

        angles = np.array([math.atan2(v[1], v[0]) for v in dirs], dtype=np.float32)
        feats = np.stack([np.cos(2 * angles), np.sin(2 * angles)], axis=1).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        _, labels, centers = cv2.kmeans(feats, 3, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

        clusters: Dict[int, List[List[float]]] = {0: [], 1: [], 2: []}
        for i, line in enumerate(lines[: len(labels)]):
            clusters[int(labels[i, 0])].append(line)

        counts = {k: len(v) for k, v in clusters.items()}
        if min(counts.values()) < 2:
            raise WorkflowNodeError(
                f"Unbalanced direction clusters {counts}. Adjust line detector or angle settings."
            )

        debug = {"counts": counts, "centers": centers.tolist(), "angle_threshold_deg": angle_threshold_deg}
        return (clusters, json.dumps(debug, ensure_ascii=False, indent=2))


class VanishingPointDetectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clusters": ("VW_CLUSTERS",)}}

    RETURN_TYPES = ("VW_VPS", "STRING")
    RETURN_NAMES = ("vanishing_points", "vp_debug_json")
    FUNCTION = "detect_vp"
    CATEGORY = "Vision/Calibration"

    def _solve_vp(self, cluster_lines: List[List[float]]) -> np.ndarray:
        A = []
        for line in cluster_lines:
            l = _line_to_homogeneous(np.array(line, dtype=np.float64))
            A.append(l)
        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        vp = vh[-1, :]
        if abs(vp[2]) < 1e-8:
            raise WorkflowNodeError("Vanishing point at infinity. Need stronger perspective view.")
        return vp / vp[2]

    def detect_vp(self, clusters):
        if not isinstance(clusters, dict) or len(clusters) != 3:
            raise WorkflowNodeError("clusters must be dict with 3 entries from LineClusteringNode")

        vps = {}
        for k, ls in clusters.items():
            if len(ls) < 2:
                raise WorkflowNodeError(f"Cluster {k} has <2 lines, cannot solve VP")
            vp = self._solve_vp(ls)
            vps[str(k)] = vp.tolist()

        return (vps, json.dumps(vps, ensure_ascii=False, indent=2))


class FocalLengthEstimationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vanishing_points": ("VW_VPS",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("VW_INTRINSICS", "FLOAT", "STRING")
    RETURN_NAMES = ("intrinsics", "focal_length", "intrinsics_debug_json")
    FUNCTION = "estimate_f"
    CATEGORY = "Vision/Calibration"

    def estimate_f(self, vanishing_points, image):
        rgb = _to_numpy_rgb(image)
        h, w = rgb.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        if not isinstance(vanishing_points, dict) or len(vanishing_points) != 3:
            raise WorkflowNodeError("Need 3 vanishing points for orthogonal constraint.")

        v = [np.array(vanishing_points[str(i)], dtype=np.float64) for i in range(3)]

        # f^2 = -((vx-c) dot (vy-c)) for orthogonal directions
        c = np.array([cx, cy], dtype=np.float64)
        f2_candidates = []
        pairs = [(0, 1), (1, 2), (0, 2)]
        for i, j in pairs:
            a = v[i][:2] - c
            b = v[j][:2] - c
            f2 = -float(np.dot(a, b))
            if f2 > 1e-6:
                f2_candidates.append(f2)

        if not f2_candidates:
            raise WorkflowNodeError(
                "No positive focal-length candidate from VP pairs. "
                "Likely VP assignment or clustering is wrong."
            )

        f = float(np.sqrt(np.median(f2_candidates)))
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        out = {"K": K.tolist(), "f": f, "cx": cx, "cy": cy}
        return (out, f, json.dumps(out, ensure_ascii=False, indent=2))


class CameraPoseRecoveryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vanishing_points": ("VW_VPS",),
                "intrinsics": ("VW_INTRINSICS",),
            }
        }

    RETURN_TYPES = ("VW_POSE", "STRING")
    RETURN_NAMES = ("pose", "pose_debug_json")
    FUNCTION = "recover_pose"
    CATEGORY = "Vision/Calibration"

    def recover_pose(self, vanishing_points, intrinsics):
        K = np.array(intrinsics["K"], dtype=np.float64)
        K_inv = np.linalg.inv(K)

        rays = []
        for i in range(3):
            vp = np.array(vanishing_points[str(i)], dtype=np.float64)
            d = _normalize(K_inv @ vp)
            rays.append(d)

        # Orthonormalize basis
        r1 = _normalize(rays[0])
        r2 = _normalize(rays[1] - np.dot(rays[1], r1) * r1)
        r3 = _normalize(np.cross(r1, r2))
        R = np.stack([r1, r2, r3], axis=1)

        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        pose = {"R": R.tolist()}
        return (pose, json.dumps(pose, ensure_ascii=False, indent=2))


class GroundHomographyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("VW_POSE",),
                "intrinsics": ("VW_INTRINSICS",),
                "plane_height": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("VW_H", "STRING")
    RETURN_NAMES = ("homography", "homography_debug_json")
    FUNCTION = "compute_h"
    CATEGORY = "Vision/Calibration"

    def compute_h(self, pose, intrinsics, plane_height):
        # Simplified H=K[r1 r2 t], with unit translation placeholder
        R = np.array(pose["R"], dtype=np.float64)
        K = np.array(intrinsics["K"], dtype=np.float64)
        t = np.array([[0.0], [0.0], [1.0 + plane_height]], dtype=np.float64)
        H = K @ np.concatenate([R[:, :2], t], axis=1)
        if abs(np.linalg.det(H)) < 1e-8:
            raise WorkflowNodeError("Computed homography is singular. Check pose/intrinsics.")
        H = H / H[2, 2]
        out = {"H": H.tolist()}
        return (out, json.dumps(out, ensure_ascii=False, indent=2))


class BirdEyeViewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "homography": ("VW_H",),
                "output_width": ("INT", {"default": 1024, "min": 128, "max": 4096}),
                "output_height": ("INT", {"default": 1024, "min": 128, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "warp"
    CATEGORY = "Vision/Calibration"

    def warp(self, image, homography, output_width, output_height):
        rgb = _to_numpy_rgb(image)
        H = np.array(homography["H"], dtype=np.float64)
        warped = cv2.warpPerspective(rgb, H, (output_width, output_height))
        return (_to_comfy_image(warped),)


class SizeMeasurementNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point1_x": ("FLOAT", {"default": 0}),
                "point1_y": ("FLOAT", {"default": 0}),
                "point2_x": ("FLOAT", {"default": 100}),
                "point2_y": ("FLOAT", {"default": 0}),
                "meters_per_pixel": ("FLOAT", {"default": 0.005, "min": 1e-6, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("distance_m", "measurement_debug")
    FUNCTION = "measure"
    CATEGORY = "Vision/Calibration"

    def measure(self, point1_x, point1_y, point2_x, point2_y, meters_per_pixel):
        p1 = np.array([point1_x, point1_y], dtype=np.float64)
        p2 = np.array([point2_x, point2_y], dtype=np.float64)
        px_dist = float(np.linalg.norm(p2 - p1))
        meters = px_dist * meters_per_pixel
        debug = {
            "point1": p1.tolist(),
            "point2": p2.tolist(),
            "pixel_distance": px_dist,
            "meters_per_pixel": meters_per_pixel,
            "distance_m": meters,
        }
        return (meters, json.dumps(debug, ensure_ascii=False, indent=2))


NODE_CLASS_MAPPINGS = {
    "ImagePreprocessNode": ImagePreprocessNode,
    "LineDetectionNode": LineDetectionNode,
    "LineClusteringNode": LineClusteringNode,
    "VanishingPointDetectionNode": VanishingPointDetectionNode,
    "FocalLengthEstimationNode": FocalLengthEstimationNode,
    "CameraPoseRecoveryNode": CameraPoseRecoveryNode,
    "GroundHomographyNode": GroundHomographyNode,
    "BirdEyeViewNode": BirdEyeViewNode,
    "SizeMeasurementNode": SizeMeasurementNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePreprocessNode": "图像预处理 / Image Preprocess",
    "LineDetectionNode": "直线检测 / Line Detection",
    "LineClusteringNode": "直线聚类 / Line Clustering",
    "VanishingPointDetectionNode": "消失点检测 / Vanishing Point",
    "FocalLengthEstimationNode": "焦距估计 / Focal Length",
    "CameraPoseRecoveryNode": "相机姿态恢复 / Camera Pose",
    "GroundHomographyNode": "地面单应矩阵 / Ground Homography",
    "BirdEyeViewNode": "鸟瞰图生成 / Bird-eye View",
    "SizeMeasurementNode": "尺寸计算 / Size Measurement",
}
