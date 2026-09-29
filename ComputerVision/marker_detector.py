"""
Marker Detector — Computer Vision Recruitment Task

Implement the MarkerDetector class and the utility functions below.
See README.md for full task description.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


class MarkerDetector:
    """
    Detects colored markers in images using classical computer vision techniques.

    Each detection is a dictionary with the following fields:
        - 'color':   str             — one of 'red', 'green', 'blue', 'yellow'
        - 'bbox':    (x, y, w, h)    — bounding rectangle of the detected contour
        - 'center':  (cx, cy)        — center coordinates of the bounding box
        - 'area':    float           — area of the detected contour

    Optionally, if you attempt the bonus task:
        - 'shape':   str             — one of 'circle', 'triangle', 'rectangle'
    """

    COLOR_RANGES = {
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([179, 255, 255]))
        ],
        "green": [
            (np.array([35, 80, 80]), np.array([85, 255, 255]))
        ],
        "blue": [
            (np.array([90, 80, 80]), np.array([130, 255, 255]))
        ],
        "yellow": [
            (np.array([20, 80, 80]), np.array([35, 255, 255]))
        ]
    }

    # Minimum contour area to consider (filters noise)
    min_area = 500

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect colored markers in the given BGR image.

        Args:
            image: Input image in BGR format (as loaded by cv2.imread).

        Returns:
            A list of detection dictionaries, each containing:
            'color', 'bbox', 'center', and 'area' keys.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        raw_detections = []

        for color_name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)

                    # Bonus task: shape classification
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                    num_vertices = len(approx)

                    if num_vertices == 3:
                        shape = "triangle"
                    elif num_vertices == 4:
                        shape = "rectangle"
                    else:
                        shape = "circle"

                    raw_detections.append({
                        "color": color_name,
                        "bbox": (int(x), int(y), int(w), int(h)),
                        "center": (cx, cy),
                        "area": area,
                        "shape": shape
                    })

        return filter_detections(raw_detections)


def compute_iou(box_a: Tuple, box_b: Tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Each box is represented as (x, y, w, h) where:
        - (x, y) is the top-left corner
        - (w, h) is the width and height

    Args:
        box_a: First bounding box as (x, y, w, h).
        box_b: Second bounding box as (x, y, w, h).

    Returns:
        IoU value as a float between 0.0 (no overlap) and 1.0 (perfect overlap).
    """
    x1_a, y1_a, w_a, h_a = box_a
    x1_b, y1_b, w_b, h_b = box_b

    x2_a, y2_a = x1_a + w_a, y1_a + h_a
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    xA = max(x1_a, x1_b)
    yA = max(y1_a, y1_b)
    xB = min(x2_a, x2_b)
    yB = min(y2_a, y2_b)

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    box_a_area = w_a * h_a
    box_b_area = w_b * h_b

    union_area = float(box_a_area + box_b_area - inter_area)
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def filter_detections(
    detections: List[Dict], iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Filter overlapping detections using Non-Maximum Suppression (NMS).

    When two detections overlap (IoU > iou_threshold), keep the one with the
    larger area and discard the other.

    Args:
        detections: List of detection dictionaries (each must have 'bbox' and 'area').
        iou_threshold: IoU threshold above which two detections are considered overlapping.

    Returns:
        Filtered list of detections with overlapping duplicates removed.
    """
    # Sort detections by area descending
    sorted_dets = sorted(detections, key=lambda d: d["area"], reverse=True)
    kept = []

    for det in sorted_dets:
        overlap = False
        for k in kept:
            if compute_iou(det["bbox"], k["bbox"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)

    return kept
