#!/usr/bin/env python3
"""Generate evaluation metrics, tables, and charts for the composite defect detector."""

from pathlib import Path
import numpy as np
import cv2
import torch

from cnn_defect_model import CompositeDefectDetector, CompositeDefectDataset, evaluate_model_on_dataset
from synthetic_ndt_generator import SyntheticNDTGenerator, NDTMethod, FibreType, DefectType


def build_demo_dataset(n_samples: int = 40):
    generator = SyntheticNDTGenerator(image_size=(224, 224), seed=42)
    images = []
    metadata = []

    defect_types = [
        DefectType.NO_DEFECT,
        DefectType.DELAMINATION,
        DefectType.CRACK,
        DefectType.VOID,
        DefectType.FIBRE_MISALIGNMENT,
        DefectType.RESIN_RICH,
        DefectType.RESIN_STARVED,
    ]

    for idx in range(n_samples):
        defect = defect_types[idx % len(defect_types)]
        fibre = [FibreType.CARBON, FibreType.GLASS, FibreType.ARAMID][idx % 3]
        method = [NDTMethod.VISUAL_TESTING, NDTMethod.ULTRASONIC, NDTMethod.RADIOGRAPHY][idx % 3]
        image, meta = generator.generate_sample(
            method=method,
            fibre_type=fibre,
            defect_type=defect,
            num_defects=1 if defect != DefectType.NO_DEFECT else 0,
        )
        images.append(image)
        metadata.append(meta)

    return images, metadata


def main():
    images, metadata = build_demo_dataset(40)
    dataset = CompositeDefectDataset(images, metadata, image_size=(224, 224))
    model = CompositeDefectDetector()

    output_dir = Path("evaluation_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate_model_on_dataset(model, dataset, output_dir=str(output_dir))
    print("Evaluation complete")
    print(report["metrics"])
    print(f"CSV: {report['csv_path']}")
    print(f"Chart: {report['chart_path']}")


if __name__ == "__main__":
    main()
