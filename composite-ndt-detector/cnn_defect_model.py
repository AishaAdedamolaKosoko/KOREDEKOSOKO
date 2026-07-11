#!/usr/bin/env python3
"""
CNN-Based Defect Detection Model for Composite Materials NDT
Multi-task model supporting all 8 NDT methods and 6 defect types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_classification_metrics(y_true: List[int], y_pred: List[int], labels: Optional[List[int]] = None, positive_label: Optional[int] = None) -> Dict[str, float]:
    """Compute common classification metrics from ground-truth and predicted labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))

    if len(labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1_score": 0.0, "error_rate": 0.0}

    if positive_label is None:
        positive_label = labels[0]

    tp = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    tn = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    fp = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    fn = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)
    error_rate = 1 - accuracy

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
        "error_rate": float(error_rate),
    }


def create_evaluation_report(y_true: List[int], y_pred: List[int], labels: Optional[List[int]] = None, positive_label: Optional[int] = None, output_dir: str = "evaluation_reports") -> Dict[str, object]:
    """Generate metrics, table, and charts for an evaluation set."""
    metrics = compute_classification_metrics(y_true, y_pred, labels=labels, positive_label=positive_label)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics], index=["model"])
    metrics_df.to_csv(output_path / "evaluation_metrics.csv")

    metric_names = ["accuracy", "precision", "recall", "specificity", "f1_score", "error_rate"]
    values = [metrics[name] for name in metric_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(metric_names, values, color=["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#64748b"])
    axes[0].set_title("Model evaluation metrics")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    for label in axes[0].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    axes[1].bar(["Correct", "Incorrect"], [sum(np.array(y_true) == np.array(y_pred)), sum(np.array(y_true) != np.array(y_pred))], color=["#22c55e", "#ef4444"])
    axes[1].set_title("Prediction outcome")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path / "evaluation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    chart_width = 500
    chart_height = 280
    bar_width = 50
    margin_left = 50
    margin_right = 30
    margin_top = 30
    margin_bottom = 60
    max_value = max(values + [1.0])
    bar_positions = []
    for idx, value in enumerate(values):
        x = margin_left + idx * 75
        y = margin_top + (1 - value / max_value) * (chart_height - margin_top - margin_bottom)
        h = max(8, (value / max_value) * (chart_height - margin_top - margin_bottom))
        bar_positions.append((x, y, h))

    svg_bars = "".join(
        f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{color}" rx="6" />\n' 
        f'<text x="{x + bar_width / 2}" y="{chart_height - 20}" text-anchor="middle" font-size="11">{name}</text>'
        for (x, y, h), name, color in zip(bar_positions, metric_names, ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#64748b"])
    )

    html_report = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Model Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 24px; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #475569; padding: 10px; text-align: left; }}
    th {{ background: #1e293b; }}
    img {{ max-width: 100%; border-radius: 10px; border: 1px solid #334155; }}
  </style>
</head>
<body>
  <h1>Model Evaluation Report</h1>
  <div class=\"card\">
    <h2>Metric table</h2>
    {metrics_df.to_html(classes='table', escape=False)}
  </div>
  <div class=\"card\">
    <h2>Metric bar chart</h2>
    <svg width=\"{chart_width}\" height=\"{chart_height}\" viewBox=\"0 0 {chart_width} {chart_height}\" xmlns=\"http://www.w3.org/2000/svg\">
      <line x1=\"{margin_left}\" y1=\"{chart_height - margin_bottom}\" x2=\"{chart_width - margin_right}\" y2=\"{chart_height - margin_bottom}\" stroke=\"#94a3b8\" />
      <line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{chart_height - margin_bottom}\" stroke=\"#94a3b8\" />
      {svg_bars}
    </svg>
  </div>
  <div class=\"card\">
    <h2>Summary chart</h2>
    <img src=\"evaluation_summary.png\" alt=\"Evaluation summary chart\" />
  </div>
</body>
</html>
"""
    (output_path / "evaluation_report.html").write_text(html_report, encoding="utf-8")

    return {
        "metrics": metrics,
        "table": metrics_df,
        "csv_path": str(output_path / "evaluation_metrics.csv"),
        "chart_path": str(output_path / "evaluation_summary.png"),
        "html_report": str(output_path / "evaluation_report.html"),
    }


class CompositeDefectDataset(Dataset):
    """PyTorch Dataset for composite material NDT images."""

    # Class mappings
    NDT_METHODS = [
        'visual_testing', 'ultrasonic', 'magnetic_particle', 'radiography',
        'liquid_penetrant', 'eddy_current', 'thermal_infrared', 'microwave', 'vibration_analysis'
    ]

    DEFECT_TYPES = [
        'no_defect', 'delamination', 'crack', 'void', 
        'fibre_misalignment', 'resin_rich', 'resin_starved'
    ]

    FIBRE_TYPES = ['aramid', 'carbon', 'glass']

    def __init__(self, images: List[np.ndarray], metadata: List[Dict], 
                 transform=None, image_size: Tuple[int, int] = (224, 224)):
        self.images = images
        self.metadata = metadata
        self.transform = transform
        self.image_size = image_size

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        meta = self.metadata[idx]

        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Encode labels
        ndt_label = self.NDT_METHODS.index(meta.get('method', 'visual_testing'))
        defect_label = self.DEFECT_TYPES.index(meta.get('defect_type', 'no_defect'))
        fibre_label = self.FIBRE_TYPES.index(meta.get('fibre_type', 'carbon'))

        # Defect severity (0.0 for no defect)
        severity = 0.0
        if meta.get('defects'):
            severity = meta['defects'][0].get('severity', 0.5)

        return {
            'image': tensor,
            'ndt_method': torch.tensor(ndt_label, dtype=torch.long),
            'defect_type': torch.tensor(defect_label, dtype=torch.long),
            'fibre_type': torch.tensor(fibre_label, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.float32),
            'metadata': meta
        }


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
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
    """Residual block with SE attention."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class FeaturePyramidNetwork(nn.Module):
    """FPN for multi-scale feature extraction."""
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build laterals
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[2:], 
                mode='nearest'
            )
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply FPN convolutions
        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outputs


class CompositeDefectDetector(nn.Module):
    """
    Multi-task CNN for composite material defect detection.

    Architecture:
    - ResNet-style backbone with SE blocks
    - Feature Pyramid Network for multi-scale features
    - Multi-task heads:
      1. NDT Method Classification (9 classes)
      2. Defect Type Classification (7 classes)
      3. Fibre Type Classification (3 classes)
      4. Defect Severity Regression
      5. Defect Localization (heat map)
    """

    def __init__(self, 
                 num_ndt_methods: int = 9,
                 num_defect_types: int = 7,
                 num_fibre_types: int = 3,
                 input_size: Tuple[int, int] = (224, 224)):
        super(CompositeDefectDetector, self).__init__()

        self.input_size = input_size

        # Backbone: ResNet-style with SE blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # Residual stages
        self.stage1 = self._make_stage(64, 64, 2, stride=1)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 3, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256)

        # Global average pooling features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Shared feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Task-specific heads
        # 1. NDT Method Classification
        self.ndt_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_ndt_methods)
        )

        # 2. Defect Type Classification
        self.defect_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_defect_types)
        )

        # 3. Fibre Type Classification
        self.fibre_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_fibre_types)
        )

        # 4. Severity Regression
        self.severity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 5. Defect Localization (heat map)
        self.localization_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _make_stage(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Backbone
        x = self.stem(x)

        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        # Feature Pyramid
        fpn_features = self.fpn([c1, c2, c3, c4])

        # Global features for classification
        global_features = []
        for feat in fpn_features:
            global_features.append(self.global_pool(feat).flatten(1))

        combined = torch.cat(global_features, dim=1)
        embedding = self.feature_embed(combined)

        # Task outputs
        ndt_logits = self.ndt_head(embedding)
        defect_logits = self.defect_head(embedding)
        fibre_logits = self.fibre_head(embedding)
        severity = self.severity_head(embedding)

        # Defect heat map (use highest resolution FPN feature)
        heat_map = self.localization_head(fpn_features[0])

        return {
            'ndt_method': ndt_logits,
            'defect_type': defect_logits,
            'fibre_type': fibre_logits,
            'severity': severity,
            'heat_map': heat_map,
            'features': embedding
        }


class CompositeDefectLoss(nn.Module):
    """Multi-task loss for composite defect detection."""

    def __init__(self, 
                 ndt_weight: float = 0.2,
                 defect_weight: float = 0.4,
                 fibre_weight: float = 0.1,
                 severity_weight: float = 0.15,
                 localization_weight: float = 0.15):
        super(CompositeDefectLoss, self).__init__()
        self.ndt_weight = ndt_weight
        self.defect_weight = defect_weight
        self.fibre_weight = fibre_weight
        self.severity_weight = severity_weight
        self.localization_weight = localization_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        # NDT method loss
        ndt_loss = self.ce_loss(predictions['ndt_method'], targets['ndt_method'])

        # Defect type loss
        defect_loss = self.ce_loss(predictions['defect_type'], targets['defect_type'])

        # Fibre type loss
        fibre_loss = self.ce_loss(predictions['fibre_type'], targets['fibre_type'])

        # Severity loss
        severity_loss = self.mse_loss(
            predictions['severity'].squeeze(), 
            targets['severity']
        )

        # Localization loss (BCE with heat map)
        # Create target heat map from bounding boxes
        batch_size = predictions['heat_map'].shape[0]
        target_heat = torch.zeros_like(predictions['heat_map'])

        for b in range(batch_size):
            meta = targets.get('metadata', [{}] * batch_size)[b]
            if isinstance(meta, dict) and meta.get('defects'):
                for defect in meta['defects']:
                    if 'position' in defect and 'size' in defect:
                        x, y = defect['position']
                        w, h = defect['size']
                        # Scale to heat map size
                        hm_h, hm_w = target_heat.shape[2:]
                        scale_x = hm_w / 512  # Assuming original image is 512x512
                        scale_y = hm_h / 512
                        cx = int((x + w // 2) * scale_x)
                        cy = int((y + h // 2) * scale_y)
                        rx = max(3, int(w * scale_x // 2))
                        ry = max(3, int(h * scale_y // 2))

                        if cx < hm_w and cy < hm_h:
                            cv2.ellipse(
                                target_heat[b, 0].cpu().numpy(),
                                (cx, cy), (rx, ry), 0, 0, 360, 1, -1
                            )

        target_heat = target_heat.to(predictions['heat_map'].device)
        localization_loss = self.bce_loss(predictions['heat_map'], target_heat)

        # Total weighted loss
        total_loss = (
            self.ndt_weight * ndt_loss +
            self.defect_weight * defect_loss +
            self.fibre_weight * fibre_loss +
            self.severity_weight * severity_loss +
            self.localization_weight * localization_loss
        )

        return {
            'total': total_loss,
            'ndt': ndt_loss,
            'defect': defect_loss,
            'fibre': fibre_loss,
            'severity': severity_loss,
            'localization': localization_loss
        }


def train_model(model: CompositeDefectDetector, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 1e-3,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Train the composite defect detection model."""

    model = model.to(device)
    criterion = CompositeDefectLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            targets = {
                'ndt_method': batch['ndt_method'].to(device),
                'defect_type': batch['defect_type'].to(device),
                'fibre_type': batch['fibre_type'].to(device),
                'severity': batch['severity'].to(device),
                'metadata': batch['metadata']
            }

            optimizer.zero_grad()
            predictions = model(images)
            losses = criterion(predictions, targets)

            losses['total'].backward()
            optimizer.step()

            train_losses.append(losses['total'].item())

        # Validation
        model.eval()
        val_losses = []
        correct_defects = 0
        total_defects = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = {
                    'ndt_method': batch['ndt_method'].to(device),
                    'defect_type': batch['defect_type'].to(device),
                    'fibre_type': batch['fibre_type'].to(device),
                    'severity': batch['severity'].to(device),
                    'metadata': batch['metadata']
                }

                predictions = model(images)
                losses = criterion(predictions, targets)
                val_losses.append(losses['total'].item())

                # Calculate defect classification accuracy
                _, predicted = torch.max(predictions['defect_type'], 1)
                correct_defects += (predicted == targets['defect_type']).sum().item()
                total_defects += targets['defect_type'].size(0)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_acc = correct_defects / total_defects if total_defects > 0 else 0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_composite_defect_model.pth')

        scheduler.step()

    return history


def predict(model: CompositeDefectDetector, 
            image: np.ndarray,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """
    Run inference on a single image.

    Args:
        model: Trained CompositeDefectDetector model
        image: Input image (BGR format from OpenCV)
        device: Computing device

    Returns:
        Dictionary with predictions and confidence scores
    """
    model.eval()
    model = model.to(device)

    # Preprocess
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # Decode predictions
    ndt_probs = F.softmax(outputs['ndt_method'], dim=1)[0].cpu().numpy()
    defect_probs = F.softmax(outputs['defect_type'], dim=1)[0].cpu().numpy()
    fibre_probs = F.softmax(outputs['fibre_type'], dim=1)[0].cpu().numpy()
    severity = outputs['severity'][0].cpu().numpy()
    heat_map = outputs['heat_map'][0, 0].cpu().numpy()

    dataset = CompositeDefectDataset([], [])

    results = {
        'ndt_method': {
            'prediction': dataset.NDT_METHODS[np.argmax(ndt_probs)],
            'confidence': float(np.max(ndt_probs)),
            'all_probabilities': {m: float(p) for m, p in zip(dataset.NDT_METHODS, ndt_probs)}
        },
        'defect_type': {
            'prediction': dataset.DEFECT_TYPES[np.argmax(defect_probs)],
            'confidence': float(np.max(defect_probs)),
            'all_probabilities': {d: float(p) for d, p in zip(dataset.DEFECT_TYPES, defect_probs)}
        },
        'fibre_type': {
            'prediction': dataset.FIBRE_TYPES[np.argmax(fibre_probs)],
            'confidence': float(np.max(fibre_probs)),
            'all_probabilities': {f: float(p) for f, p in zip(dataset.FIBRE_TYPES, fibre_probs)}
        },
        'severity': {
            'value': float(severity),
            'level': 'severe' if severity > 0.7 else 'moderate' if severity > 0.4 else 'minor'
        },
        'heat_map': heat_map,
        'requires_action': np.argmax(defect_probs) != 0  # Not 'no_defect'
    }

    return results


def evaluate_model_on_dataset(model: CompositeDefectDetector, dataset: Dataset, output_dir: str = "evaluation_reports", device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, object]:
    """Run inference over a dataset and export evaluation metrics, tables, and charts."""
    model = model.to(device)
    model.eval()

    predictions = []
    targets = []
    dataset_instance = CompositeDefectDataset([], [])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for sample in dataset:
            image = sample['image']
            if isinstance(image, torch.Tensor):
                tensor = image.unsqueeze(0).to(device)
            else:
                image = np.asarray(image)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensor = transform(image_rgb).unsqueeze(0).to(device)

            outputs = model(tensor)
            probs = F.softmax(outputs['defect_type'], dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            true_idx = int(sample['defect_type'].item())
            predictions.append(pred_idx)
            targets.append(true_idx)

    return create_evaluation_report(
        y_true=targets,
        y_pred=predictions,
        labels=list(range(len(dataset_instance.DEFECT_TYPES))),
        positive_label=1,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    # Model summary
    model = CompositeDefectDetector()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CompositeDefectDetector")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    outputs = model(dummy_input)
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
