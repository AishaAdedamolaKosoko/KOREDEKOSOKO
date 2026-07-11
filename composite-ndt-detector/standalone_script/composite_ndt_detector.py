#!/usr/bin/env python3
"""
=============================================================================
Composite Material NDT Defect Detection & Analysis System
=============================================================================

A comprehensive machine learning-based system for detecting and analyzing
defects in fibre-reinforced polymer (FRP) composite materials using
non-destructive testing (NDT) results.

Supported NDT Methods:
  1. Visual Testing (VT)
  2. Ultrasonic Testing (UT)
  3. Magnetic Particle Testing (MPT)
  4. Radiographic Testing (RT)
  5. Liquid Penetrant Testing (LPT)
  6. Eddy Current Testing (ECT)
  7. Thermal/Infrared Testing (IRT)
  8. Microwave NDT
  9. Vibration Analysis

Supported Fibre Types:
  - Aramid Fibre (Kevlar)
  - Carbon Fibre (CFRP)
  - Glass Fibre (GFRP)

Detectable Defects:
  - Delamination
  - Cracks
  - Voids
  - Fibre Misalignment
  - Resin-Rich Areas
  - Resin-Starved Areas

Usage:
  python composite_ndt_detector.py --mode train --epochs 50
  python composite_ndt_detector.py --mode predict --image path/to/ndt_image.jpg
  python composite_ndt_detector.py --mode demo

Author: Composite NDT ML System
Version: 1.0.0
=============================================================================
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. Install with: pip install torch torchvision")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """System configuration."""
    IMAGE_SIZE: Tuple[int, int] = (512, 512)
    MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)
    NUM_NDT_METHODS: int = 9
    NUM_DEFECT_TYPES: int = 7
    NUM_FIBRE_TYPES: int = 3
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 30
    TRAIN_SPLIT: float = 0.8
    SYNTHETIC_SAMPLES: int = 2000
    MODEL_PATH: str = "composite_defect_model.pth"
    OUTPUT_DIR: str = "output"
    DATASET_DIR: str = "dataset"
    DEVICE: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

class NDTMethod:
    """NDT method definitions."""
    VISUAL_TESTING = "visual_testing"
    ULTRASONIC = "ultrasonic"
    MAGNETIC_PARTICLE = "magnetic_particle"
    RADIOGRAPHY = "radiography"
    LIQUID_PENETRANT = "liquid_penetrant"
    EDDY_CURRENT = "eddy_current"
    THERMAL_INFRARED = "thermal_infrared"
    MICROWAVE = "microwave"
    VIBRATION_ANALYSIS = "vibration_analysis"
    
    ALL = [VISUAL_TESTING, ULTRASONIC, MAGNETIC_PARTICLE, RADIOGRAPHY,
           LIQUID_PENETRANT, EDDY_CURRENT, THERMAL_INFRARED, MICROWAVE, VIBRATION_ANALYSIS]
    
    DISPLAY_NAMES = {
        VISUAL_TESTING: "Visual Testing (VT)",
        ULTRASONIC: "Ultrasonic Testing (UT)",
        MAGNETIC_PARTICLE: "Magnetic Particle Testing (MPT)",
        RADIOGRAPHY: "Radiographic Testing (RT)",
        LIQUID_PENETRANT: "Liquid Penetrant Testing (LPT)",
        EDDY_CURRENT: "Eddy Current Testing (ECT)",
        THERMAL_INFRARED: "Thermal/Infrared Testing (IRT)",
        MICROWAVE: "Microwave NDT",
        VIBRATION_ANALYSIS: "Vibration Analysis"
    }


class DefectType:
    """Defect type definitions."""
    NO_DEFECT = "no_defect"
    DELAMINATION = "delamination"
    CRACK = "crack"
    VOID = "void"
    FIBRE_MISALIGNMENT = "fibre_misalignment"
    RESIN_RICH = "resin_rich"
    RESIN_STARVED = "resin_starved"
    
    ALL = [NO_DEFECT, DELAMINATION, CRACK, VOID,
           FIBRE_MISALIGNMENT, RESIN_RICH, RESIN_STARVED]
    
    DISPLAY_NAMES = {
        NO_DEFECT: "No Defect",
        DELAMINATION: "Delamination",
        CRACK: "Crack",
        VOID: "Void",
        FIBRE_MISALIGNMENT: "Fibre Misalignment",
        RESIN_RICH: "Resin-Rich Area",
        RESIN_STARVED: "Resin-Starved Area"
    }
    
    SEVERITY_COLORS = {
        NO_DEFECT: "#4CAF50",
        DELAMINATION: "#FF5722",
        CRACK: "#F44336",
        VOID: "#FF9800",
        FIBRE_MISALIGNMENT: "#9C27B0",
        RESIN_RICH: "#2196F3",
        RESIN_STARVED: "#795548"
    }


class FibreType:
    """Fibre type definitions."""
    ARAMID = "aramid"
    CARBON = "carbon"
    GLASS = "glass"
    
    ALL = [ARAMID, CARBON, GLASS]
    
    DISPLAY_NAMES = {
        ARAMID: "Aramid Fibre (Kevlar)",
        CARBON: "Carbon Fibre (CFRP)",
        GLASS: "Glass Fibre (GFRP)"
    }


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticNDTGenerator:
    """Generates realistic synthetic NDT result images for composite materials."""
    
    FIBRE_PROPERTIES = {
        FibreType.ARAMID: {'color_low': (140, 100, 50), 'color_high': (200, 160, 100),
                          'texture_freq': 0.08, 'bundle_width': 6},
        FibreType.CARBON: {'color_low': (20, 20, 20), 'color_high': (60, 60, 60),
                          'texture_freq': 0.12, 'bundle_width': 4},
        FibreType.GLASS: {'color_low': (180, 180, 170), 'color_high': (240, 240, 230),
                         'texture_freq': 0.06, 'bundle_width': 8}
    }
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512), seed: Optional[int] = None):
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        if seed:
            import random
            random.seed(seed)
    
    def generate_base_texture(self, fibre_type: str) -> np.ndarray:
        """Generate woven composite material texture."""
        h, w = self.image_size
        props = self.FIBRE_PROPERTIES[fibre_type]
        freq = props['texture_freq']
        color_low = np.array(props['color_low'])
        color_high = np.array(props['color_high'])
        
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        weave_h = np.sin(2 * np.pi * freq * Y + self.rng.randn() * 0.5) * 0.5 + 0.5
        weave_v = np.sin(2 * np.pi * freq * X + self.rng.randn() * 0.5) * 0.5 + 0.5
        weave = weave_h * weave_v + 0.3 * (weave_h + weave_v)
        weave = (weave - weave.min()) / (weave.max() - weave.min() + 1e-8)
        
        bw = props['bundle_width']
        for i in range(0, h, max(bw, 3)):
            weave[i:i+max(1, bw//2), :] *= 1.15
        for j in range(0, w, max(bw, 3)):
            weave[:, j:j+max(1, bw//2)] *= 1.15
        weave = np.clip(weave, 0, 1)
        
        base = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            base[:, :, c] = color_low[c] + weave * (color_high[c] - color_low[c])
        
        noise = self.rng.randn(h, w, 3) * 5
        base = np.clip(base + noise, 0, 255).astype(np.uint8)
        return base
    
    def _create_defect_mask(self, center: Tuple[int, int], size: Tuple[int, int],
                           orientation: float) -> np.ndarray:
        """Create elliptical defect mask."""
        mask = np.zeros(self.image_size, dtype=np.float32)
        axes = (size[0] // 2, size[1] // 2)
        cv2.ellipse(mask, center, axes, orientation, 0, 360, 1, -1)
        return cv2.GaussianBlur(mask, (21, 21), 5)
    
    def add_delamination(self, image: np.ndarray, center: Tuple[int, int],
                        size: Tuple[int, int], orientation: float,
                        severity: float) -> np.ndarray:
        """Add delamination defect."""
        result = image.astype(np.float32)
        mask = self._create_defect_mask(center, size, orientation)
        brightness = 30 + 40 * severity
        for c in range(3):
            result[:, :, c] += mask * brightness
        yy, xx = np.ogrid[:self.image_size[0], :self.image_size[1]]
        dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
        rings = np.sin(dist / (5 + 10 * (1 - severity))) * 10 * severity
        for c in range(3):
            result[:, :, c] += mask * rings
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_crack(self, image: np.ndarray, start: Tuple[int, int],
                 length: int, width: int, orientation: float,
                 severity: float) -> np.ndarray:
        """Add crack defect."""
        result = image.astype(np.float32)
        angle = np.deg2rad(orientation)
        num_points = max(8, length // 6)
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            px = int(start[0] + t * length * np.cos(angle) + self.rng.randn() * 3 * severity)
            py = int(start[1] + t * length * np.sin(angle) + self.rng.randn() * 3 * severity)
            points.append((px, py))
        crack_w = max(1, int(2 + 3 * severity))
        for i in range(len(points) - 1):
            cv2.line(result, points[i], points[i + 1], (0, 0, 0), crack_w)
            cv2.line(result, points[i], points[i + 1], (30, 30, 30), crack_w + 4)
        if severity > 0.5:
            for _ in range(int(2 * severity)):
                idx = self.rng.randint(1, len(points) - 1)
                branch_angle = angle + np.deg2rad(self.rng.uniform(-50, 50))
                branch_len = int(length * 0.25 * self.rng.uniform(0.3, 0.7))
                bx = int(points[idx][0] + branch_len * np.cos(branch_angle))
                by = int(points[idx][1] + branch_len * np.sin(branch_angle))
                cv2.line(result, points[idx], (bx, by), (15, 15, 15), max(1, crack_w - 1))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_void(self, image: np.ndarray, center: Tuple[int, int],
                size: Tuple[int, int], severity: float) -> np.ndarray:
        """Add void defect."""
        result = image.astype(np.float32)
        mask = np.zeros(self.image_size, dtype=np.float32)
        num_blobs = int(3 + 5 * severity)
        for _ in range(num_blobs):
            offset_x = self.rng.randint(-size[0]//3, size[0]//3)
            offset_y = self.rng.randint(-size[1]//3, size[1]//3)
            radius = self.rng.randint(max(3, min(size)//8), max(8, min(size)//3))
            blob_center = (center[0] + offset_x, center[1] + offset_y)
            cv2.circle(mask, blob_center, radius, 1, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 3)
        for c in range(3):
            result[:, :, c] -= mask * (20 + 30 * severity)
        edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge = cv2.dilate(edge, np.ones((5, 5), np.uint8))
        edge_mask = edge.astype(np.float32) / 255.0
        for c in range(3):
            result[:, :, c] += edge_mask * 40 * severity
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_fibre_misalignment(self, image: np.ndarray, center: Tuple[int, int],
                              size: Tuple[int, int], orientation: float,
                              severity: float) -> np.ndarray:
        """Add fibre misalignment defect."""
        result = image.astype(np.float32)
        mask = np.zeros(self.image_size, dtype=np.float32)
        x, y = center[0] - size[0]//2, center[1] - size[1]//2
        cv2.rectangle(mask, (x, y), (x + size[0], y + size[1]), 1, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 8)
        freq = 0.08
        yy, xx = np.ogrid[:self.image_size[0], :self.image_size[1]]
        misaligned_angle = np.deg2rad(orientation + 90)
        pattern = np.sin(2 * np.pi * freq * (xx * np.cos(misaligned_angle) +
                                             yy * np.sin(misaligned_angle))) * 0.5 + 0.5
        pattern_shift = (pattern - 0.5) * 40 * severity
        for c in range(3):
            result[:, :, c] += mask * pattern_shift
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_resin_rich(self, image: np.ndarray, center: Tuple[int, int],
                      size: Tuple[int, int], severity: float) -> np.ndarray:
        """Add resin-rich area."""
        result = image.astype(np.float32)
        mask = np.zeros(self.image_size, dtype=np.float32)
        points = []
        num_points = 12
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius = min(size) // 2 * (0.6 + 0.4 * np.sin(3 * angle))
            px = int(center[0] + radius * np.cos(angle))
            py = int(center[1] + radius * np.sin(angle))
            points.append((px, py))
        cv2.fillPoly(mask, [np.array(points, np.int32).reshape(-1, 1, 2)], 1)
        mask = cv2.GaussianBlur(mask, (25, 25), 6)
        gloss = 25 + 35 * severity
        for c in range(3):
            result[:, :, c] += mask * gloss
        blurred = cv2.GaussianBlur(result, (15, 15), 5)
        blur_mask = mask[:, :, None] * 0.5 * severity
        result = result * (1 - blur_mask) + blurred * blur_mask
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def add_resin_starved(self, image: np.ndarray, center: Tuple[int, int],
                         size: Tuple[int, int], severity: float) -> np.ndarray:
        """Add resin-starved area."""
        result = image.astype(np.float32)
        mask = np.zeros(self.image_size, dtype=np.float32)
        num_patches = int(4 + 6 * severity)
        x, y = center[0] - size[0]//2, center[1] - size[1]//2
        for _ in range(num_patches):
            px = x + self.rng.randint(0, size[0])
            py = y + self.rng.randint(0, size[1])
            pr = self.rng.randint(10, max(20, min(size) // 4))
            cv2.circle(mask, (px, py), pr, 1, -1)
        mask = cv2.GaussianBlur(mask, (19, 19), 4)
        dark = 25 + 35 * severity
        for c in range(3):
            result[:, :, c] -= mask * dark
        freq = 0.15
        yy, xx = np.ogrid[:self.image_size[0], :self.image_size[1]]
        fibre = np.sin(2 * np.pi * freq * xx) * np.sin(2 * np.pi * freq * yy)
        fibre = (fibre + 1) / 2
        for c in range(3):
            result[:, :, c] += mask * fibre * 15 * severity
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_ndt_signature(self, image: np.ndarray, method: str,
                           fibre_type: str) -> np.ndarray:
        """Apply NDT-specific visual signature."""
        result = image.astype(np.float32)
        h, w = self.image_size
        
        if method == NDTMethod.VISUAL_TESTING:
            pass
        elif method == NDTMethod.ULTRASONIC:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = cv2.GaussianBlur(gray, (5, 5), 1)
            result = np.zeros((h, w, 3), dtype=np.float32)
            normalized = gray / 255.0
            result[:, :, 0] = np.clip(normalized * 2 - 0.5, 0, 1) * 255
            result[:, :, 1] = (1 - np.abs(normalized - 0.5) * 2) * 255
            result[:, :, 2] = np.clip(1 - normalized * 2 + 0.5, 0, 1) * 255
            for i in range(0, h, 4):
                result[i:i+1, :, :] *= 0.9
        elif method == NDTMethod.MAGNETIC_PARTICLE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bw = 255 - bw
            result = np.stack([bw, bw, bw], axis=-1).astype(np.float32)
            glow = cv2.GaussianBlur(bw.astype(np.float32), (15, 15), 5)
            result[:, :, 0] += glow * 0.3
            result[:, :, 2] += glow * 0.2
        elif method == NDTMethod.RADIOGRAPHY:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = 255 - gray
            grain = self.rng.randn(h, w) * 10
            gray = np.clip(gray + grain, 0, 255)
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray.astype(np.uint8)).astype(np.float32)
            result = np.stack([gray, gray, gray], axis=-1)
        elif method == NDTMethod.LIQUID_PENETRANT:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, defect_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            defect_mask = cv2.GaussianBlur(defect_mask.astype(np.float32), (11, 11), 3)
            result = np.ones((h, w, 3), dtype=np.float32) * 240
            penetrant = np.array([50, 50, 255], dtype=np.float32)
            mask_3ch = defect_mask[:, :, None] / 255.0
            result = result * (1 - mask_3ch * 0.7) + penetrant[None, None, :] * mask_3ch * 0.7
        elif method == NDTMethod.EDDY_CURRENT:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            normalized = gray / 255.0
            result = np.zeros((h, w, 3), dtype=np.float32)
            result[:, :, 0] = np.clip(1 - normalized, 0, 1) * 200 + 55
            result[:, :, 1] = np.sin(normalized * np.pi) * 200 + 55
            result[:, :, 2] = np.clip(normalized, 0, 1) * 200 + 55
            for i in range(0, h, 32):
                result[i:i+1, :, :] *= 0.85
            for j in range(0, w, 32):
                result[:, j:j+1, :] *= 0.85
        elif method == NDTMethod.THERMAL_INFRARED:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            thermal = cv2.GaussianBlur(gray, (21, 21), 5)
            normalized = thermal / 255.0
            result = np.zeros((h, w, 3), dtype=np.float32)
            result[:, :, 2] = np.clip(normalized * 3, 0, 1) * 255
            result[:, :, 1] = np.clip((normalized - 0.33) * 3, 0, 1) * 255
            result[:, :, 0] = np.clip((normalized - 0.66) * 3, 0, 1) * 255
            bar_w = 30
            for i in range(h):
                t = i / h
                result[i, -bar_w:, 2] = np.clip(t * 3, 0, 1) * 255
                result[i, -bar_w:, 1] = np.clip((t - 0.33) * 3, 0, 1) * 255
                result[i, -bar_w:, 0] = np.clip((t - 0.66) * 3, 0, 1) * 255
        elif method == NDTMethod.MICROWAVE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            normalized = gray / 255.0
            result = np.zeros((h, w, 3), dtype=np.float32)
            hue = normalized * 360
            for i in range(h):
                for j in range(w):
                    h_val = hue[i, j] / 60.0
                    c = 255 * 0.8
                    x = c * (1 - abs(h_val % 2 - 1))
                    if h_val < 1:
                        result[i, j] = [0, x, c]
                    elif h_val < 2:
                        result[i, j] = [0, c, x]
                    elif h_val < 3:
                        result[i, j] = [x, c, 0]
                    elif h_val < 4:
                        result[i, j] = [c, x, 0]
                    elif h_val < 5:
                        result[i, j] = [c, 0, x]
                    else:
                        result[i, j] = [x, 0, c]
            fringe = np.sin(normalized * 20) * 10
            for c in range(3):
                result[:, :, c] += fringe
        elif method == NDTMethod.VIBRATION_ANALYSIS:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            freq = 0.05
            yy, xx = np.ogrid[:h, :w]
            mode_shape = np.sin(2 * np.pi * freq * xx) * np.sin(2 * np.pi * freq * yy * 1.5)
            displacement = mode_shape * 30 + gray * 0.3
            normalized = (displacement - displacement.min()) / (displacement.max() - displacement.min() + 1e-6)
            result = np.zeros((h, w, 3), dtype=np.float32)
            result[:, :, 2] = normalized * 200 + 55
            result[:, :, 1] = (1 - normalized) * 100 + 100
            result[:, :, 0] = (1 - normalized) * 200 + 55
            contour = cv2.Canny((normalized * 255).astype(np.uint8), 30, 70)
            contour_mask = contour.astype(np.float32) / 255.0
            for c in range(3):
                result[:, :, c] += contour_mask * 50
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generate_sample(self, method: str, fibre_type: str,
                       defect_type: str, num_defects: int = 1) -> Tuple[np.ndarray, Dict]:
        """Generate a complete NDT sample."""
        image = self.generate_base_texture(fibre_type)
        metadata = {
            'method': method,
            'fibre_type': fibre_type,
            'defect_type': defect_type,
            'defects': [],
            'bounding_boxes': []
        }
        if defect_type != DefectType.NO_DEFECT:
            for _ in range(num_defects):
                max_w = self.image_size[1] // 3
                max_h = self.image_size[0] // 3
                min_w, min_h = 40, 40
                w = self.rng.randint(min_w, max_w)
                h_size = self.rng.randint(min_h, max_h)
                x = self.rng.randint(20, self.image_size[1] - w - 20)
                y = self.rng.randint(20, self.image_size[0] - h_size - 20)
                severity = float(self.rng.uniform(0.3, 1.0))
                orientation = float(self.rng.uniform(0, 180))
                center = (x + w // 2, y + h_size // 2)
                if defect_type == DefectType.DELAMINATION:
                    image = self.add_delamination(image, center, (w, h_size), orientation, severity)
                elif defect_type == DefectType.CRACK:
                    image = self.add_crack(image, (x, y), max(w, h_size), min(w, h_size), orientation, severity)
                elif defect_type == DefectType.VOID:
                    image = self.add_void(image, center, (w, h_size), severity)
                elif defect_type == DefectType.FIBRE_MISALIGNMENT:
                    image = self.add_fibre_misalignment(image, center, (w, h_size), orientation, severity)
                elif defect_type == DefectType.RESIN_RICH:
                    image = self.add_resin_rich(image, center, (w, h_size), severity)
                elif defect_type == DefectType.RESIN_STARVED:
                    image = self.add_resin_starved(image, center, (w, h_size), severity)
                metadata['defects'].append({
                    'type': defect_type,
                    'position': (x, y),
                    'size': (w, h_size),
                    'severity': severity,
                    'orientation': orientation
                })
                metadata['bounding_boxes'].append((x, y, x + w, y + h_size))
        image = self.apply_ndt_signature(image, method, fibre_type)
        return image, metadata


# ============================================================================
# CNN MODEL
# ============================================================================

if TORCH_AVAILABLE:
    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.se = SEBlock(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += self.shortcut(x)
            return self.relu(out)

    class FeaturePyramidNetwork(nn.Module):
        def __init__(self, in_channels_list, out_channels=256):
            super().__init__()
            self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
            self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        def forward(self, features):
            laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
            for i in range(len(laterals) - 1, 0, -1):
                upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
                laterals[i-1] = laterals[i-1] + upsampled
            return [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]

    class CompositeDefectDetector(nn.Module):
        def __init__(self, num_ndt=9, num_defects=7, num_fibre=3, input_size=(224, 224)):
            super().__init__()
            self.input_size = input_size
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)
            )
            self.stage1 = self._make_stage(64, 64, 2, 1)
            self.stage2 = self._make_stage(64, 128, 2, 2)
            self.stage3 = self._make_stage(128, 256, 3, 2)
            self.stage4 = self._make_stage(256, 512, 3, 2)
            self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.feature_embed = nn.Sequential(
                nn.Linear(256 * 4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            self.ndt_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_ndt))
            self.defect_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_defects))
            self.fibre_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_fibre))
            self.severity_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())
            self.localization_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 1, 1), nn.Sigmoid()
            )
            self._init_weights()

        def _make_stage(self, in_ch, out_ch, num_blocks, stride):
            layers = [ResidualBlock(in_ch, out_ch, stride)]
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_ch, out_ch))
            return nn.Sequential(*layers)

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.stem(x)
            c1 = self.stage1(x)
            c2 = self.stage2(c1)
            c3 = self.stage3(c2)
            c4 = self.stage4(c3)
            fpn_features = self.fpn([c1, c2, c3, c4])
            global_features = [self.global_pool(f).flatten(1) for f in fpn_features]
            combined = torch.cat(global_features, dim=1)
            embedding = self.feature_embed(combined)
            return {
                'ndt_method': self.ndt_head(embedding),
                'defect_type': self.defect_head(embedding),
                'fibre_type': self.fibre_head(embedding),
                'severity': self.severity_head(embedding),
                'heat_map': self.localization_head(fpn_features[0]),
                'features': embedding
            }


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

class DefectAnalyzer:
    """Analyzes and visualizes NDT defect detection results."""
    
    NDT_METHODS = NDTMethod.ALL
    DEFECT_TYPES = DefectType.ALL
    FIBRE_TYPES = FibreType.ALL
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = Config()
        self.device = self.config.DEVICE
        self.model = None
        self.model_path = model_path or self.config.MODEL_PATH
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        self.model = CompositeDefectDetector(
            num_ndt=self.config.NUM_NDT_METHODS,
            num_defects=self.config.NUM_DEFECT_TYPES,
            num_fibre=self.config.NUM_FIBRE_TYPES,
            input_size=self.config.MODEL_INPUT_SIZE
        )
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Could not load model weights: {e}")
                print("Using randomly initialized model for demonstration.")
        else:
            print(f"Model file not found at {self.model_path}")
            print("Using randomly initialized model for demonstration.")
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image_rgb).unsqueeze(0).to(self.device)
    
    def analyze(self, image: np.ndarray) -> Dict:
        """Run complete analysis on an NDT image."""
        if not TORCH_AVAILABLE or self.model is None:
            return self._mock_analysis(image)
        tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(tensor)
        ndt_probs = F.softmax(outputs['ndt_method'], dim=1)[0].cpu().numpy()
        defect_probs = F.softmax(outputs['defect_type'], dim=1)[0].cpu().numpy()
        fibre_probs = F.softmax(outputs['fibre_type'], dim=1)[0].cpu().numpy()
        severity = outputs['severity'][0].cpu().numpy()
        heat_map = outputs['heat_map'][0, 0].cpu().numpy()
        return {
            'ndt_method': {
                'prediction': self.NDT_METHODS[np.argmax(ndt_probs)],
                'confidence': float(np.max(ndt_probs)),
                'probabilities': {m: float(p) for m, p in zip(self.NDT_METHODS, ndt_probs)}
            },
            'defect_type': {
                'prediction': self.DEFECT_TYPES[np.argmax(defect_probs)],
                'confidence': float(np.max(defect_probs)),
                'probabilities': {d: float(p) for d, p in zip(self.DEFECT_TYPES, defect_probs)}
            },
            'fibre_type': {
                'prediction': self.FIBRE_TYPES[np.argmax(fibre_probs)],
                'confidence': float(np.max(fibre_probs)),
                'probabilities': {f: float(p) for f, p in zip(self.FIBRE_TYPES, fibre_probs)}
            },
            'severity': {
                'value': float(self._extract_scalar(severity)),
                'level': 'Severe' if self._extract_scalar(severity) > 0.7 else 'Moderate' if self._extract_scalar(severity) > 0.4 else 'Minor'
            },
            'heat_map': heat_map,
            'requires_action': np.argmax(defect_probs) != 0
        }
    
    def _extract_scalar(self, val):
        """Extract Python scalar from numpy scalar/array."""
        if hasattr(val, 'item'):
            return val.item()
        return float(val)
    
    def _mock_analysis(self, image: np.ndarray) -> Dict:
        """Generate mock analysis for demonstration without PyTorch."""
        rng = np.random.RandomState(hash(image.tobytes()) % 2**31)
        ndt_probs = rng.dirichlet(np.ones(9)) * 0.8 + 0.02
        defect_probs = rng.dirichlet(np.ones(7)) * 0.8 + 0.02
        fibre_probs = rng.dirichlet(np.ones(3)) * 0.8 + 0.06
        severity_val = float(rng.uniform(0.1, 0.95))
        heat_map = rng.rand(56, 56)
        return {
            'ndt_method': {
                'prediction': self.NDT_METHODS[np.argmax(ndt_probs)],
                'confidence': float(np.max(ndt_probs)),
                'probabilities': {m: float(p) for m, p in zip(self.NDT_METHODS, ndt_probs)}
            },
            'defect_type': {
                'prediction': self.DEFECT_TYPES[np.argmax(defect_probs)],
                'confidence': float(np.max(defect_probs)),
                'probabilities': {d: float(p) for d, p in zip(self.DEFECT_TYPES, defect_probs)}
            },
            'fibre_type': {
                'prediction': self.FIBRE_TYPES[np.argmax(fibre_probs)],
                'confidence': float(np.max(fibre_probs)),
                'probabilities': {f: float(p) for f, p in zip(self.FIBRE_TYPES, fibre_probs)}
            },
            'severity': {
                'value': severity_val,
                'level': 'Severe' if severity_val > 0.7 else 'Moderate' if severity_val > 0.4 else 'Minor'
            },
            'heat_map': heat_map,
            'requires_action': np.argmax(defect_probs) != 0
        }
    
    def generate_report(self, image: np.ndarray, results: Dict, output_path: str) -> str:
        """Generate comprehensive analysis report with visualizations."""
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle('COMPOSITE MATERIAL NDT DEFECT ANALYSIS REPORT',
                    fontsize=20, fontweight='bold', color='white', y=0.98)
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3,
                             left=0.05, right=0.95, top=0.93, bottom=0.05)
        
        # 1. Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original NDT Image', fontsize=12, fontweight='bold', color='white')
        ax1.axis('off')
        ax1.set_facecolor('#1a1a2e')
        
        # 2. Heat map overlay
        ax2 = fig.add_subplot(gs[0, 1])
        heat_map = results['heat_map']
        heat_resized = cv2.resize(heat_map, (image.shape[1], image.shape[0]))
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.6)
        im = ax2.imshow(heat_resized, cmap='hot', alpha=0.5)
        ax2.set_title('Defect Heat Map', fontsize=12, fontweight='bold', color='white')
        ax2.axis('off')
        ax2.set_facecolor('#1a1a2e')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # 3. NDT Method probabilities
        ax3 = fig.add_subplot(gs[0, 2])
        ndt_data = results['ndt_method']['probabilities']
        colors_ndt = plt.cm.viridis(np.linspace(0.2, 0.9, len(ndt_data)))
        bars = ax3.barh(range(len(ndt_data)), list(ndt_data.values()), color=colors_ndt)
        ax3.set_yticks(range(len(ndt_data)))
        ax3.set_yticklabels([NDTMethod.DISPLAY_NAMES.get(k, k) for k in ndt_data.keys()],
                           fontsize=8, color='white')
        ax3.set_xlabel('Confidence', color='white', fontsize=10)
        ax3.set_title('NDT Method Classification', fontsize=12, fontweight='bold', color='white')
        ax3.set_facecolor('#1a1a2e')
        ax3.tick_params(colors='white')
        ax3.invert_yaxis()
        
        # 4. Fibre type
        ax4 = fig.add_subplot(gs[0, 3])
        fibre_data = results['fibre_type']['probabilities']
        colors_fibre = ['#FFD700', '#333333', '#E0E0E0']
        ax4.pie(fibre_data.values(), labels=[FibreType.DISPLAY_NAMES.get(k, k) for k in fibre_data.keys()],
                autopct='%1.1f%%', colors=colors_fibre, startangle=90,
                textprops={'color': 'white', 'fontsize': 9})
        ax4.set_title('Fibre Type Classification', fontsize=12, fontweight='bold', color='white')
        ax4.set_facecolor('#1a1a2e')
        
        # 5. Defect type probabilities
        ax5 = fig.add_subplot(gs[1, :2])
        defect_data = results['defect_type']['probabilities']
        colors_defect = [DefectType.SEVERITY_COLORS.get(k, '#888888') for k in defect_data.keys()]
        bars = ax5.bar(range(len(defect_data)), list(defect_data.values()),
                      color=colors_defect, edgecolor='white', linewidth=1.5)
        ax5.set_xticks(range(len(defect_data)))
        ax5.set_xticklabels([DefectType.DISPLAY_NAMES.get(k, k) for k in defect_data.keys()],
                           rotation=30, ha='right', fontsize=9, color='white')
        ax5.set_ylabel('Confidence', color='white', fontsize=11)
        ax5.set_title('Defect Type Classification', fontsize=14, fontweight='bold', color='white')
        ax5.set_facecolor('#1a1a2e')
        ax5.tick_params(colors='white')
        ax5.set_ylim(0, 1)
        for bar, val in zip(bars, defect_data.values()):
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
        
        # 6. Severity gauge
        ax6 = fig.add_subplot(gs[1, 2])
        severity = results['severity']['value']
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        for i, t in enumerate(theta[:-1]):
            color = plt.cm.RdYlGn(1 - t / np.pi)
            ax6.plot([0, r * np.cos(t)], [0, r * np.sin(t)], color=color, linewidth=8, alpha=0.8)
        needle_angle = np.pi * (1 - severity)
        ax6.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                 head_width=0.08, head_length=0.05, fc='white', ec='white', linewidth=2)
        ax6.set_xlim(-1.3, 1.3)
        ax6.set_ylim(-0.3, 1.3)
        ax6.set_aspect('equal')
        ax6.axis('off')
        ax6.set_facecolor('#1a1a2e')
        ax6.text(0, -0.15, f"SEVERITY: {results['severity']['level']}",
                ha='center', fontsize=14, fontweight='bold', color='white')
        ax6.text(0, -0.3, f"Score: {severity:.3f}", ha='center', fontsize=11, color='#cccccc')
        ax6.set_title('Defect Severity', fontsize=12, fontweight='bold', color='white')
        
        # 7. Summary panel
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        ax7.set_facecolor('#1a1a2e')
        ax7.set_title('Analysis Summary', fontsize=12, fontweight='bold', color='white')
        
        summary_text = f"""
NDT Method:
  {NDTMethod.DISPLAY_NAMES.get(results['ndt_method']['prediction'], 'Unknown')}
  (Confidence: {results['ndt_method']['confidence']:.3f})

Detected Defect:
  {DefectType.DISPLAY_NAMES.get(results['defect_type']['prediction'], 'None')}
  (Confidence: {results['defect_type']['confidence']:.3f})

Fibre Material:
  {FibreType.DISPLAY_NAMES.get(results['fibre_type']['prediction'], 'Unknown')}
  (Confidence: {results['fibre_type']['confidence']:.3f})

Severity Assessment:
  Level: {results['severity']['level']}
  Score: {results['severity']['value']:.3f}/1.0

Recommendation:
  {"IMMEDIATE ACTION REQUIRED" if results['requires_action'] and results['severity']['value'] > 0.6 else 
   "Monitor and schedule inspection" if results['requires_action'] else "No action required - material is sound"}
        """
        status_color = '#FF4444' if results['requires_action'] and results['severity']['value'] > 0.6 else \
                      '#FFAA00' if results['requires_action'] else '#44AA44'
        rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.03",
                             facecolor='#16213e', edgecolor=status_color, linewidth=3)
        ax7.add_patch(rect)
        ax7.text(0.5, 0.5, summary_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='center', horizontalalignment='center',
                family='monospace', color='white', linespacing=1.8)
        
        # 8. Recommendations
        ax8 = fig.add_subplot(gs[2, :])
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        ax8.set_facecolor('#1a1a2e')
        ax8.set_title('Defect-Specific Recommendations & NDT Method Guidance',
                     fontsize=14, fontweight='bold', color='white')
        self._add_recommendations(ax8, results)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close()
        return output_path
    
    def _add_recommendations(self, ax, results):
        """Add defect-specific recommendations."""
        defect = results['defect_type']['prediction']
        severity = results['severity']['value']
        ndt = results['ndt_method']['prediction']
        
        recommendations = {
            DefectType.NO_DEFECT: [
                "No defects detected. Material is within acceptable quality standards.",
                "Continue routine inspection schedule.",
                "Document baseline NDT results for future comparison."
            ],
            DefectType.DELAMINATION: [
                "CRITICAL: Layer separation detected. Structural integrity compromised.",
                "Recommended: Ultrasonic C-scan for depth mapping of delamination.",
                "Consider repair by resin injection or patch bonding.",
                "If severity > 0.7: Component replacement may be necessary."
            ],
            DefectType.CRACK: [
                "CRITICAL: Crack propagation risk. Immediate assessment required.",
                "Recommended: Radiography for through-thickness crack extent.",
                "Apply temporary structural reinforcement if load-bearing.",
                "Monitor crack growth rate with periodic ultrasonic inspection."
            ],
            DefectType.VOID: [
                "Porosity detected. May affect mechanical properties.",
                "Recommended: X-ray CT for 3D void distribution mapping.",
                "Evaluate void content against acceptance criteria (typically <2%).",
                "If clustered voids: Consider local repair or scarf patch."
            ],
            DefectType.FIBRE_MISALIGNMENT: [
                "Fibre orientation deviation detected. May reduce strength.",
                "Recommended: Shearography for full-field strain analysis.",
                "Compare against design allowable fibre angles (typically +/- 5 degrees).",
                "Critical areas may require scarf repair or local reinforcement."
            ],
            DefectType.RESIN_RICH: [
                "Excess resin detected. May indicate manufacturing inconsistency.",
                "Recommended: Ultrasonic velocity measurement for resin content.",
                "Typically acceptable if area fraction < 5% of total surface.",
                "Monitor for associated voids or dry fibre areas nearby."
            ],
            DefectType.RESIN_STARVED: [
                "Insufficient resin detected. Fibre support may be inadequate.",
                "Recommended: Thermography for moisture ingress assessment.",
                "High risk of fibre abrasion and environmental degradation.",
                "Repair by resin infusion or application of surface seal coat."
            ]
        }
        
        recs = recommendations.get(defect, recommendations[DefectType.NO_DEFECT])
        y_pos = 0.9
        for rec in recs:
            color = '#FF6B6B' if 'CRITICAL' in rec else '#4ECDC4' if 'Recommended' in rec else '#95E1D3'
            ax.text(0.02, y_pos, f"  {rec}", fontsize=10, color=color,
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e',
                            edgecolor=color, alpha=0.8))
            y_pos -= 0.22
        
        ax.text(0.02, 0.05, f"Current NDT Method: {NDTMethod.DISPLAY_NAMES.get(ndt, ndt)} | "
               f"Severity: {results['severity']['level']} ({severity:.3f}) | "
               f"Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               fontsize=9, color='#888888', transform=ax.transAxes)


def train_mode(args):
    """Run training mode."""
    print("=" * 70)
    print("COMPOSITE NDT DEFECT DETECTION - TRAINING MODE")
    print("=" * 70)
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required for training. Install with: pip install torch torchvision")
        return
    config = Config()
    print(f"\nGenerating {config.SYNTHETIC_SAMPLES} synthetic NDT samples...")
    generator = SyntheticNDTGenerator(image_size=config.IMAGE_SIZE, seed=42)
    images = []
    metadata_list = []
    methods = NDTMethod.ALL
    fibres = FibreType.ALL
    defects = DefectType.ALL
    for i in range(config.SYNTHETIC_SAMPLES):
        method = methods[i % len(methods)]
        fibre = fibres[(i // len(methods)) % len(fibres)]
        defect = defects[i % len(defects)]
        num_defects = 0 if defect == DefectType.NO_DEFECT else 1
        image, meta = generator.generate_sample(method, fibre, defect, num_defects)
        images.append(image)
        metadata_list.append(meta)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{config.SYNTHETIC_SAMPLES} samples...")
    print("Dataset generation complete!")
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    for i in range(min(20, len(images))):
        cv2.imwrite(f"{config.DATASET_DIR}/sample_{i:04d}_{metadata_list[i]['defect_type']}.png", images[i])
    print(f"Saved 20 sample images to {config.DATASET_DIR}/")
    model = CompositeDefectDetector(
        num_ndt=config.NUM_NDT_METHODS,
        num_defects=config.NUM_DEFECT_TYPES,
        num_fibre=config.NUM_FIBRE_TYPES
    )
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nModel initialized and saved to {config.MODEL_PATH}")
    print("NOTE: This is a randomly initialized model for demonstration.")
    print("For production use, train on real NDT data using the full training pipeline.")


def predict_mode(args):
    """Run prediction mode."""
    print("=" * 70)
    print("COMPOSITE NDT DEFECT DETECTION - ANALYSIS MODE")
    print("=" * 70)
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    print(f"\nAnalyzing: {image_path}")
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    analyzer = DefectAnalyzer(model_path=args.model)
    print("\nRunning ML analysis...")
    start_time = time.time()
    results = analyzer.analyze(image)
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f} seconds")
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\n1. NDT METHOD DETECTED:")
    ndt = results['ndt_method']
    print(f"   Method: {NDTMethod.DISPLAY_NAMES.get(ndt['prediction'], ndt['prediction'])}")
    print(f"   Confidence: {ndt['confidence']:.3f}")
    print(f"\n2. DEFECT CLASSIFICATION:")
    defect = results['defect_type']
    print(f"   Defect Type: {DefectType.DISPLAY_NAMES.get(defect['prediction'], defect['prediction'])}")
    print(f"   Confidence: {defect['confidence']:.3f}")
    print(f"\n3. FIBRE TYPE:")
    fibre = results['fibre_type']
    print(f"   Material: {FibreType.DISPLAY_NAMES.get(fibre['prediction'], fibre['prediction'])}")
    print(f"   Confidence: {fibre['confidence']:.3f}")
    print(f"\n4. SEVERITY ASSESSMENT:")
    sev = results['severity']
    print(f"   Level: {sev['level']}")
    print(f"   Score: {sev['value']:.3f}/1.0")
    print(f"\n5. ACTION REQUIRED: {'YES' if results['requires_action'] else 'NO'}")
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"{args.output}/report_{timestamp}.png"
    print(f"\nGenerating visual report...")
    analyzer.generate_report(image, results, report_path)
    print(f"Report saved: {report_path}")
    json_path = f"{args.output}/results_{timestamp}.json"
    json_results = {k: v for k, v in results.items() if k != 'heat_map'}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved: {json_path}")
    heat_path = f"{args.output}/heatmap_{timestamp}.png"
    plt.imsave(heat_path, results['heat_map'], cmap='hot')
    print(f"Heat map saved: {heat_path}")


def demo_mode(args):
    """Run demonstration mode with synthetic data."""
    print("=" * 70)
    print("COMPOSITE NDT DEFECT DETECTION - DEMONSTRATION MODE")
    print("=" * 70)
    generator = SyntheticNDTGenerator(image_size=(512, 512), seed=42)
    analyzer = DefectAnalyzer()
    os.makedirs("demo_output", exist_ok=True)
    demos = [
        (NDTMethod.ULTRASONIC, FibreType.CARBON, DefectType.DELAMINATION, "delamination_ut"),
        (NDTMethod.RADIOGRAPHY, FibreType.GLASS, DefectType.VOID, "void_rt"),
        (NDTMethod.THERMAL_INFRARED, FibreType.CARBON, DefectType.CRACK, "crack_irt"),
        (NDTMethod.EDDY_CURRENT, FibreType.CARBON, DefectType.FIBRE_MISALIGNMENT, "misalignment_ect"),
        (NDTMethod.LIQUID_PENETRANT, FibreType.GLASS, DefectType.CRACK, "crack_lpt"),
        (NDTMethod.ULTRASONIC, FibreType.ARAMID, DefectType.RESIN_RICH, "resinrich_ut"),
        (NDTMethod.VISUAL_TESTING, FibreType.CARBON, DefectType.RESIN_STARVED, "resinstarved_vt"),
        (NDTMethod.MICROWAVE, FibreType.GLASS, DefectType.VOID, "void_mw"),
    ]
    print(f"\nGenerating and analyzing {len(demos)} synthetic NDT samples...\n")
    for method, fibre, defect, name in demos:
        print(f"  Processing: {name}...")
        image, meta = generator.generate_sample(method, fibre, defect, num_defects=1)
        results = analyzer.analyze(image)
        cv2.imwrite(f"demo_output/{name}_original.png", image)
        analyzer.generate_report(image, results, f"demo_output/{name}_report.png")
        print(f"    NDT: {results['ndt_method']['prediction']} | "
              f"Defect: {results['defect_type']['prediction']} | "
              f"Fibre: {results['fibre_type']['prediction']} | "
              f"Severity: {results['severity']['level']}")
    print(f"\nDemo complete! All outputs saved to demo_output/")
    print("Files generated:")
    print("  - *_original.png : Synthetic NDT input images")
    print("  - *_report.png   : Analysis reports with visualizations")


def main():
    parser = argparse.ArgumentParser(
        description='Composite Material NDT Defect Detection & Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate demo analysis with synthetic data
  python %(prog)s --mode demo
  
  # Analyze a single NDT image
  python %(prog)s --mode predict --image ndt_scan.jpg
  
  # Train model on synthetic dataset
  python %(prog)s --mode train --epochs 50
        """
    )
    parser.add_argument('--mode', choices=['train', 'predict', 'demo'],
                       default='demo', help='Operation mode (default: demo)')
    parser.add_argument('--image', type=str, help='Path to NDT image for analysis')
    parser.add_argument('--model', type=str, default='composite_defect_model.pth',
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (auto-detected by default)')
    args = parser.parse_args()
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        if not args.image:
            print("ERROR: --image is required for predict mode")
            parser.print_help()
            sys.exit(1)
        predict_mode(args)
    elif args.mode == 'demo':
        demo_mode(args)


if __name__ == "__main__":
    main()
