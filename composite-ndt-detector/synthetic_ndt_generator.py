#!/usr/bin/env python3
"""
Synthetic NDT Data Generator for Composite Materials
Generates realistic NDT result images for training and demonstration.
Supports all 8 NDT methods across multiple fibre types and defect categories.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import random


class FibreType(Enum):
    ARAMID = "aramid"
    CARBON = "carbon"
    GLASS = "glass"


class NDTMethod(Enum):
    VISUAL_TESTING = "visual_testing"
    ULTRASONIC = "ultrasonic"
    MAGNETIC_PARTICLE = "magnetic_particle"
    RADIOGRAPHY = "radiography"
    LIQUID_PENETRANT = "liquid_penetrant"
    EDDY_CURRENT = "eddy_current"
    THERMAL_INFRARED = "thermal_infrared"
    MICROWAVE = "microwave"
    VIBRATION_ANALYSIS = "vibration_analysis"


class DefectType(Enum):
    NO_DEFECT = "no_defect"
    DELAMINATION = "delamination"
    CRACK = "crack"
    VOID = "void"
    FIBRE_MISALIGNMENT = "fibre_misalignment"
    RESIN_RICH = "resin_rich"
    RESIN_STARVED = "resin_starved"


@dataclass
class DefectConfig:
    """Configuration for defect generation."""
    defect_type: DefectType
    position: Tuple[int, int]
    size: Tuple[int, int]
    severity: float  # 0.0 to 1.0
    orientation: float  # degrees


class SyntheticNDTGenerator:
    """
    Generates synthetic NDT result images for composite materials.
    Simulates realistic artifacts for all major NDT methods.
    """

    # Characteristic textures and patterns for each fibre type
    FIBRE_PATTERNS = {
        FibreType.ARAMID: {
            'weave_pattern': cv2.imread('shared_assets/aramid_weave.png') if False else None,
            'color_range': ((140, 100, 50), (200, 160, 100)),
            'texture_freq': 0.08,
        },
        FibreType.CARBON: {
            'weave_pattern': None,
            'color_range': ((20, 20, 20), (60, 60, 60)),
            'texture_freq': 0.12,
        },
        FibreType.GLASS: {
            'weave_pattern': None,
            'color_range': ((180, 180, 170), (240, 240, 230)),
            'texture_freq': 0.06,
        }
    }

    def __init__(self, image_size: Tuple[int, int] = (512, 512), seed: Optional[int] = None):
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)
        if seed:
            random.seed(seed)

    def generate_base_texture(self, fibre_type: FibreType) -> np.ndarray:
        """Generate base composite material texture."""
        h, w = self.image_size

        # Create woven fibre pattern
        freq = self.FIBRE_PATTERNS[fibre_type]['texture_freq']
        color_low, color_high = self.FIBRE_PATTERNS[fibre_type]['color_range']

        # Horizontal and vertical weave lines
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)

        # Weave pattern simulation
        weave_h = np.sin(2 * np.pi * freq * Y + self.rng.randn() * 0.5) * 0.5 + 0.5
        weave_v = np.sin(2 * np.pi * freq * X + self.rng.randn() * 0.5) * 0.5 + 0.5

        # Interlaced weave
        weave = weave_h * weave_v + 0.3 * (weave_h + weave_v)
        weave = (weave - weave.min()) / (weave.max() - weave.min())

        # Add fibre bundles
        bundle_width = int(8 / freq / 100)
        for i in range(0, h, max(bundle_width, 4)):
            weave[i:i+max(2, bundle_width//3), :] *= 1.2
        for j in range(0, w, max(bundle_width, 4)):
            weave[:, j:j+max(2, bundle_width//3)] *= 1.2

        weave = np.clip(weave, 0, 1)

        # Map to colour range
        base = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            base[:, :, c] = color_low[c] + weave * (color_high[c] - color_low[c])

        # Add noise
        noise = self.rng.randn(h, w, 3) * 5
        base = np.clip(base + noise, 0, 255).astype(np.uint8)

        return base

    def add_delamination(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add delamination defect - layer separation."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        w, h = config.size
        severity = config.severity

        # Create elliptical delamination
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(mask, center, axes, config.orientation, 0, 360, 1, -1)

        # Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 5)

        # Delamination appears lighter (air gap)
        brightness_shift = 30 + 40 * severity
        for c in range(3):
            result[:, :, c] += mask * brightness_shift

        # Add internal structure - concentric rings
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        dist = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
        rings = np.sin(dist / (5 + 10 * (1 - severity))) * 10 * severity
        for c in range(3):
            result[:, :, c] += mask * rings

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_crack(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add crack defect - fracture line."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        length, width = config.size
        severity = config.severity
        angle = np.deg2rad(config.orientation)

        # Create crack path with some randomness
        num_points = max(10, length // 5)
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            px = int(x + t * length * np.cos(angle) + self.rng.randn() * 3 * severity)
            py = int(y + t * length * np.sin(angle) + self.rng.randn() * 3 * severity)
            points.append((px, py))

        # Draw crack line
        crack_width = max(1, int(2 + 3 * severity))
        for i in range(len(points) - 1):
            cv2.line(result, points[i], points[i + 1], (0, 0, 0), crack_width)
            # Add halo around crack
            cv2.line(result, points[i], points[i + 1], (20, 20, 20), crack_width + 4)

        # Add branching micro-cracks
        if severity > 0.5:
            for _ in range(int(3 * severity)):
                idx = self.rng.randint(1, len(points) - 1)
                branch_angle = angle + np.deg2rad(self.rng.uniform(-60, 60))
                branch_len = int(length * 0.3 * self.rng.uniform(0.3, 0.8))
                bx = int(points[idx][0] + branch_len * np.cos(branch_angle))
                by = int(points[idx][1] + branch_len * np.sin(branch_angle))
                cv2.line(result, points[idx], (bx, by), (10, 10, 10), max(1, crack_width - 1))

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_void(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add void defect - air pocket."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        w, h = config.size
        severity = config.severity

        # Create irregular void shape
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        center = (x + w // 2, y + h // 2)

        # Blobby shape using multiple circles
        num_blobs = int(3 + 5 * severity)
        for _ in range(num_blobs):
            offset_x = self.rng.randint(-w//3, w//3)
            offset_y = self.rng.randint(-h//3, h//3)
            radius = self.rng.randint(max(3, w//8), max(8, w//3))
            blob_center = (center[0] + offset_x, center[1] + offset_y)
            cv2.circle(mask, blob_center, radius, 1, -1)

        # Dilate and blur
        mask = cv2.GaussianBlur(mask, (15, 15), 3)

        # Void appears as darker region with bright edges (acoustic impedance mismatch)
        void_darkness = 20 + 30 * severity
        for c in range(3):
            result[:, :, c] -= mask * void_darkness

        # Bright edge halo
        edge = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        edge = cv2.dilate(edge, np.ones((5, 5), np.uint8))
        edge_mask = edge.astype(np.float32) / 255.0
        for c in range(3):
            result[:, :, c] += edge_mask * 40 * severity

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_fibre_misalignment(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add fibre misalignment defect."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        w, h = config.size
        severity = config.severity

        # Create region with different fibre orientation
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 8)

        # Generate misaligned fibre pattern
        freq = 0.08
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        misaligned_angle = np.deg2rad(config.orientation + 90)

        # New orientation pattern
        new_pattern = np.sin(
            2 * np.pi * freq * (xx * np.cos(misaligned_angle) + yy * np.sin(misaligned_angle))
        ) * 0.5 + 0.5

        # Convert to colour shift
        pattern_shift = (new_pattern - 0.5) * 40 * severity
        for c in range(3):
            result[:, :, c] += mask * pattern_shift

        # Add distortion lines at boundary
        edge_strength = 20 * severity
        edges = cv2.Canny((mask * 255).astype(np.uint8), 30, 100)
        edge_mask = edges.astype(np.float32) / 255.0
        for c in range(3):
            result[:, :, c] += edge_mask * edge_strength

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_resin_rich(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add resin-rich area - excess resin."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        w, h = config.size
        severity = config.severity

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        # Irregular blob shape
        points = []
        num_points = 12
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius = min(w, h) // 2 * (0.6 + 0.4 * np.sin(3 * angle))
            px = int(x + w // 2 + radius * np.cos(angle))
            py = int(y + h // 2 + radius * np.sin(angle))
            points.append((px, py))

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        mask = cv2.GaussianBlur(mask, (25, 25), 6)

        # Resin-rich areas appear smoother and glossier (lighter, less texture)
        gloss_shift = 25 + 35 * severity
        for c in range(3):
            result[:, :, c] += mask * gloss_shift

        # Reduce texture contrast
        blur_mask = mask * 0.5 * severity
        blurred = cv2.GaussianBlur(result, (15, 15), 5)
        result = result * (1 - blur_mask[:, :, None]) + blurred * blur_mask[:, :, None]

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_resin_starved(self, image: np.ndarray, config: DefectConfig) -> np.ndarray:
        """Add resin-starved area - insufficient resin."""
        result = image.copy().astype(np.float32)
        x, y = config.position
        w, h = config.size
        severity = config.severity

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        # Create patchy, irregular region
        num_patches = int(4 + 6 * severity)
        for _ in range(num_patches):
            px = x + self.rng.randint(0, w)
            py = y + self.rng.randint(0, h)
            pr = self.rng.randint(10, max(20, min(w, h) // 4))
            cv2.circle(mask, (px, py), pr, 1, -1)

        mask = cv2.GaussianBlur(mask, (19, 19), 4)

        # Resin-starved areas appear darker, dry, with exposed fibres
        dark_shift = 25 + 35 * severity
        for c in range(3):
            result[:, :, c] -= mask * dark_shift

        # Enhance fibre texture (fibres more visible)
        freq = 0.15
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        fibre_detail = np.sin(2 * np.pi * freq * xx) * np.sin(2 * np.pi * freq * yy)
        fibre_detail = (fibre_detail + 1) / 2
        for c in range(3):
            result[:, :, c] += mask * fibre_detail * 15 * severity

        return np.clip(result, 0, 255).astype(np.uint8)

    def apply_ndt_signature(self, image: np.ndarray, method: NDTMethod, 
                           fibre_type: FibreType) -> np.ndarray:
        """
        Apply NDT-specific visual signature to the image.
        Simulates how each NDT method visualizes defects.
        """
        result = image.copy().astype(np.float32)

        if method == NDTMethod.VISUAL_TESTING:
            # Surface defects visible to naked eye
            pass  # Base image already shows surface features

        elif method == NDTMethod.ULTRASONIC:
            # A-scan/C-scan appearance: grayscale, echo amplitude map
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Apply ultrasonic-like contrast enhancement
            gray = cv2.GaussianBlur(gray, (5, 5), 1)
            # Amplitude-based false coloring (blue-white-red)
            result = np.zeros_like(image.astype(np.float32))
            normalized = gray / 255.0
            # Blue for low amplitude, white for medium, red for high
            result[:, :, 0] = np.clip(normalized * 2 - 0.5, 0, 1) * 255  # B
            result[:, :, 1] = (1 - np.abs(normalized - 0.5) * 2) * 255  # G
            result[:, :, 2] = np.clip(1 - normalized * 2 + 0.5, 0, 1) * 255  # R
            # Add scan lines
            for i in range(0, image.shape[0], 4):
                result[i:i+1, :, :] *= 0.9

        elif method == NDTMethod.MAGNETIC_PARTICLE:
            # Black and white with particle indications
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert (white background, dark indications)
            bw = 255 - bw
            result = np.stack([bw, bw, bw], axis=-1).astype(np.float32)
            # Add fluorescent particle glow around defects
            glow = cv2.GaussianBlur(bw.astype(np.float32), (15, 15), 5)
            result[:, :, 0] += glow * 0.3  # Slight blue glow
            result[:, :, 2] += glow * 0.2  # Slight red glow

        elif method == NDTMethod.RADIOGRAPHY:
            # X-ray style: grayscale, density-based
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # X-ray attenuation (lighter = more dense, darker = less dense/voids)
            gray = 255 - gray  # Invert
            # Add film grain
            grain = self.rng.randn(*gray.shape) * 10
            gray = np.clip(gray + grain, 0, 255)
            # Enhance contrast for defect visibility
            gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray.astype(np.uint8)).astype(np.float32)
            result = np.stack([gray, gray, gray], axis=-1)

        elif method == NDTMethod.LIQUID_PENETRANT:
            # Bright coloured indication on white background
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, defect_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            defect_mask = cv2.GaussianBlur(defect_mask.astype(np.float32), (11, 11), 3)

            # White developer background
            result = np.ones_like(image.astype(np.float32)) * 240
            # Red penetrant indication (fluorescent red)
            penetrant_colour = np.array([50, 50, 255], dtype=np.float32)  # BGR red
            mask_3ch = defect_mask[:, :, None] / 255.0
            result = result * (1 - mask_3ch * 0.7) + penetrant_colour[None, None, :] * mask_3ch * 0.7

        elif method == NDTMethod.EDDY_CURRENT:
            # Impedance plane display style: false colour
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Eddy current responds to conductivity changes
            normalized = gray / 255.0
            result = np.zeros_like(image.astype(np.float32))
            # False color: blue-green-yellow-red
            result[:, :, 0] = np.clip(1 - normalized, 0, 1) * 200 + 55  # B
            result[:, :, 1] = np.sin(normalized * np.pi) * 200 + 55   # G
            result[:, :, 2] = np.clip(normalized, 0, 1) * 200 + 55    # R
            # Add impedance plane grid
            for i in range(0, image.shape[0], 32):
                result[i:i+1, :, :] *= 0.85
            for j in range(0, image.shape[1], 32):
                result[:, j:j+1, :] *= 0.85

        elif method == NDTMethod.THERMAL_INFRARED:
            # IR thermography: heat map coloring
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Temperature distribution (defects disrupt heat flow)
            thermal = cv2.GaussianBlur(gray, (21, 21), 5)
            normalized = thermal / 255.0
            result = np.zeros_like(image.astype(np.float32))
            # Ironbow colormap approximation (black-blue-purple-red-yellow-white)
            result[:, :, 2] = np.clip(normalized * 3, 0, 1) * 255  # R
            result[:, :, 1] = np.clip((normalized - 0.33) * 3, 0, 1) * 255  # G
            result[:, :, 0] = np.clip((normalized - 0.66) * 3, 0, 1) * 255  # B
            # Add temperature scale bar on right edge
            bar_width = 30
            for i in range(image.shape[0]):
                t = i / image.shape[0]
                result[i, -bar_width:, 2] = np.clip(t * 3, 0, 1) * 255
                result[i, -bar_width:, 1] = np.clip((t - 0.33) * 3, 0, 1) * 255
                result[i, -bar_width:, 0] = np.clip((t - 0.66) * 3, 0, 1) * 255

        elif method == NDTMethod.MICROWAVE:
            # Microwave NDT: amplitude and phase display
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Microwave responds to dielectric properties
            normalized = gray / 255.0
            result = np.zeros_like(image.astype(np.float32))
            # Phase-based coloring (HSL-like)
            hue = normalized * 360
            # Convert HSV to RGB-like for display
            h_norm = hue / 60.0
            c = 255 * 0.8  # saturation
            x = c * (1 - np.abs(h_norm % 2 - 1))

            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    h = h_norm[i, j]
                    if h < 1:
                        result[i, j] = [0, x[i, j], c]
                    elif h < 2:
                        result[i, j] = [0, c, x[i, j]]
                    elif h < 3:
                        result[i, j] = [x[i, j], c, 0]
                    elif h < 4:
                        result[i, j] = [c, x[i, j], 0]
                    elif h < 5:
                        result[i, j] = [c, 0, x[i, j]]
                    else:
                        result[i, j] = [x[i, j], 0, c]
            # Add interference fringes
            fringe = np.sin(normalized * 20) * 10
            for c in range(3):
                result[:, :, c] += fringe

        elif method == NDTMethod.VIBRATION_ANALYSIS:
            # Vibration mode shape display
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Modal analysis result - displacement field
            freq = 0.05
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            # Standing wave pattern
            mode_shape = np.sin(2 * np.pi * freq * xx) * np.sin(2 * np.pi * freq * yy * 1.5)
            # Defects alter local stiffness -> change in displacement
            displacement = mode_shape * 30 + gray * 0.3
            normalized = (displacement - displacement.min()) / (displacement.max() - displacement.min() + 1e-6)

            result = np.zeros_like(image.astype(np.float32))
            # Displacement magnitude coloring
            result[:, :, 2] = normalized * 200 + 55  # R
            result[:, :, 1] = (1 - normalized) * 100 + 100  # G
            result[:, :, 0] = (1 - normalized) * 200 + 55  # B

            # Add nodal lines
            contour_lines = cv2.Canny((normalized * 255).astype(np.uint8), 30, 70)
            contour_mask = contour_lines.astype(np.float32) / 255.0
            for c in range(3):
                result[:, :, c] += contour_mask * 50

        return np.clip(result, 0, 255).astype(np.uint8)

    def generate_sample(self, method: NDTMethod, fibre_type: FibreType,
                       defect_type: DefectType, num_defects: int = 1) -> Tuple[np.ndarray, Dict]:
        """
        Generate a complete NDT sample image with specified parameters.

        Returns:
            image: Generated NDT result image
            metadata: Dictionary with defect annotations and bounding boxes
        """
        # Generate base composite texture
        image = self.generate_base_texture(fibre_type)

        metadata = {
            'method': method.value,
            'fibre_type': fibre_type.value,
            'defect_type': defect_type.value,
            'defects': [],
            'bounding_boxes': []
        }

        if defect_type != DefectType.NO_DEFECT:
            for _ in range(num_defects):
                # Random defect position and size
                max_w = self.image_size[1] // 3
                max_h = self.image_size[0] // 3
                min_w, min_h = 40, 40

                w = self.rng.randint(min_w, max_w)
                h = self.rng.randint(min_h, max_h)
                x = self.rng.randint(20, self.image_size[1] - w - 20)
                y = self.rng.randint(20, self.image_size[0] - h - 20)
                severity = float(self.rng.uniform(0.3, 1.0))
                orientation = float(self.rng.uniform(0, 180))

                config = DefectConfig(
                    defect_type=defect_type,
                    position=(x, y),
                    size=(w, h),
                    severity=severity,
                    orientation=orientation
                )

                # Apply defect
                if defect_type == DefectType.DELAMINATION:
                    image = self.add_delamination(image, config)
                elif defect_type == DefectType.CRACK:
                    image = self.add_crack(image, config)
                elif defect_type == DefectType.VOID:
                    image = self.add_void(image, config)
                elif defect_type == DefectType.FIBRE_MISALIGNMENT:
                    image = self.add_fibre_misalignment(image, config)
                elif defect_type == DefectType.RESIN_RICH:
                    image = self.add_resin_rich(image, config)
                elif defect_type == DefectType.RESIN_STARVED:
                    image = self.add_resin_starved(image, config)

                metadata['defects'].append({
                    'type': defect_type.value,
                    'position': (x, y),
                    'size': (w, h),
                    'severity': severity,
                    'orientation': orientation
                })
                metadata['bounding_boxes'].append((x, y, x + w, y + h))

        # Apply NDT-specific visual signature
        image = self.apply_ndt_signature(image, method, fibre_type)

        return image, metadata


# Convenience function for batch generation
def generate_dataset(num_samples: int = 100, image_size: Tuple[int, int] = (512, 512),
                    seed: int = 42) -> List[Tuple[np.ndarray, Dict]]:
    """Generate a diverse synthetic NDT dataset."""
    generator = SyntheticNDTGenerator(image_size=image_size, seed=seed)
    dataset = []

    methods = list(NDTMethod)
    fibres = list(FibreType)
    defects = list(DefectType)

    for i in range(num_samples):
        method = methods[i % len(methods)]
        fibre = fibres[i % len(fibres)]
        defect = defects[i % len(defects)]
        num_defects = 1 if defect != DefectType.NO_DEFECT else 0

        image, metadata = generator.generate_sample(method, fibre, defect, num_defects)
        dataset.append((image, metadata))

    return dataset


if __name__ == "__main__":
    # Demo: generate sample images for each NDT method
    import os

    generator = SyntheticNDTGenerator(image_size=(512, 512), seed=42)
    os.makedirs("demo_output", exist_ok=True)

    methods = [NDTMethod.ULTRASONIC, NDTMethod.RADIOGRAPHY, NDTMethod.THERMAL_INFRARED]
    defects = [DefectType.NO_DEFECT, DefectType.DELAMINATION, DefectType.CRACK, DefectType.VOID]

    for method in methods:
        for defect in defects:
            image, metadata = generator.generate_sample(
                method, FibreType.CARBON, defect, num_defects=1
            )
            filename = f"demo_output/{method.value}_{defect.value}.png"
            cv2.imwrite(filename, image)
            print(f"Generated: {filename}")

    print("Demo images generated in demo_output/")
