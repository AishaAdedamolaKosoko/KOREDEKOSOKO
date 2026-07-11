# Composite Material NDT Defect Detection & Analysis System

A comprehensive machine learning-based system for detecting and analyzing defects in fibre-reinforced polymer (FRP) composite materials using non-destructive testing (NDT) results.

## Overview

This system uses a deep Convolutional Neural Network (CNN) with attention mechanisms to automatically detect, classify, and localize defects in composite materials from NDT result images. It supports all major NDT methods and defect types commonly found in aerospace, automotive, marine, and wind energy applications.

## Features

### Supported NDT Methods (9)
| Method | Code | Best For |
|--------|------|----------|
| Visual Testing | VT | Surface defects, colour variations |
| Ultrasonic Testing | UT | Internal delaminations, voids, thickness measurement |
| Magnetic Particle Testing | MPT | Surface cracks in ferromagnetic materials |
| Radiographic Testing | RT | Internal voids, inclusions, density changes |
| Liquid Penetrant Testing | LPT | Surface-breaking cracks, porosity |
| Eddy Current Testing | ECT | Conductivity changes, fibre orientation |
| Thermal/Infrared Testing | IRT | Subsurface delaminations, disbonds |
| Microwave NDT | MW | Dielectric changes, moisture content |
| Vibration Analysis | VA | Stiffness changes, modal properties |

### Supported Fibre Types (3)
- **Aramid Fibre (Kevlar)** - High impact resistance, ballistic protection
- **Carbon Fibre (CFRP)** - High stiffness, aerospace grade
- **Glass Fibre (GFRP)** - Cost-effective, corrosion resistant

### Detectable Defects (6 + No Defect)
| Defect | Criticality | Description |
|--------|-------------|-------------|
| No Defect | None | Material within quality standards |
| Delamination | **High** | Layer separation in composite |
| Crack | **High** | Fracture in matrix or fibres |
| Void | Medium | Air pocket/porosity |
| Fibre Misalignment | Medium | Deviation from design orientation |
| Resin-Rich Area | Low | Excess resin accumulation |
| Resin-Starved Area | Medium | Insufficient resin |

## System Architecture

### ML Model Architecture
- **Backbone**: ResNet-style CNN with Squeeze-and-Excitation (SE) blocks
- **Feature Extraction**: Feature Pyramid Network (FPN) for multi-scale features
- **Task Heads**:
  - NDT Method Classification (9 classes)
  - Defect Type Classification (7 classes)
  - Fibre Type Classification (3 classes)
  - Severity Regression (0-1 continuous)
  - Defect Localization (heat map)

### Key Components
1. **Synthetic NDT Generator** - Creates realistic training data for all NDT methods
2. **Defect Analyzer** - Performs multi-task inference on uploaded images
3. **Report Generator** - Produces comprehensive visual analysis reports
4. **REST API** - FastAPI backend for web integration
5. **React Frontend** - Modern responsive web interface

## Deliverables

This project includes **three complete deliverables**:

### 1. Standalone Python Script
```bash
cd standalone_script
python composite_ndt_detector.py --mode demo    # Run demo with synthetic data
python composite_ndt_detector.py --mode predict --image path/to/ndt.jpg
python composite_ndt_detector.py --mode train --epochs 50
```

### 2. Gradio Interactive Web App
```bash
cd gradio_app
pip install gradio
python gradio_ndt_app.py
# Open browser at http://localhost:7860
```

### 3. Full Web Application (React + FastAPI)
```bash
# Terminal 1 - Backend
cd web_app/backend
pip install fastapi uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd web_app/frontend
npm install
npm start
# Open browser at http://localhost:3000
```

## Installation

### Requirements
```bash
pip install numpy opencv-python matplotlib torch torchvision
pip install fastapi uvicorn gradio pillow
pip install react-scripts  # For frontend only
```

### Quick Start
```bash
# Clone/navigate to project
cd composite-ndt-detector

# Run standalone demo
python standalone_script/composite_ndt_detector.py --mode demo

# Or launch Gradio app
python gradio_app/gradio_ndt_app.py
```

## API Reference

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/ndt-methods` | GET | List supported NDT methods |
| `/api/defect-types` | GET | List detectable defect types |
| `/api/fibre-types` | GET | List supported fibre types |
| `/api/analyze` | POST | Analyze uploaded NDT image |
| `/api/generate` | POST | Generate synthetic NDT sample |
| `/api/batch` | POST | Batch analyze multiple images |

### Example API Usage
```python
import requests

# Analyze an image
with open('ndt_scan.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/api/analyze', files={'file': f})

results = response.json()
print(f"Defect: {results['defect_type']['prediction']}")
print(f"Severity: {results['severity']['level']}")
```

## Usage Examples

### Generate Synthetic NDT Data
```python
from standalone_script.composite_ndt_detector import SyntheticNDTGenerator, NDTMethod, FibreType, DefectType

generator = SyntheticNDTGenerator(image_size=(512, 512), seed=42)
image, metadata = generator.generate_sample(
    method=NDTMethod.ULTRASONIC,
    fibre_type=FibreType.CARBON,
    defect_type=DefectType.DELAMINATION,
    num_defects=1
)
# image is a numpy array (BGR format)
```

### Run Analysis Programmatically
```python
from standalone_script.composite_ndt_detector import DefectAnalyzer
import cv2

analyzer = DefectAnalyzer()
image = cv2.imread('your_ndt_image.jpg')
results = analyzer.analyze(image)

# Access predictions
print(results['ndt_method']['prediction'])      # e.g., 'ultrasonic'
print(results['defect_type']['prediction'])     # e.g., 'delamination'
print(results['fibre_type']['prediction'])      # e.g., 'carbon'
print(results['severity']['value'])             # e.g., 0.73

# Generate report
analyzer.generate_report(image, results, 'report.png')
```

## Project Structure
```
composite-ndt-detector/
  standalone_script/
    composite_ndt_detector.py      # Main standalone script
  gradio_app/
    gradio_ndt_app.py              # Gradio web application
  web_app/
    backend/
      main.py                       # FastAPI backend server
    frontend/
      public/
        index.html                  # HTML entry point
      src/
        App.js                      # React application
        index.js                    # Entry point
        index.css                   # Tailwind styles
      package.json                  # Frontend dependencies
  synthetic_ndt_generator.py         # Core: Synthetic data generator
  cnn_defect_model.py                # Core: CNN model definition
  README.md                          # This file
```

## Technical Details

### Synthetic Data Generation
The system includes a sophisticated synthetic NDT data generator that simulates realistic defects and NDT method signatures:

- **Defect Simulation**: Delamination, cracks, voids, fibre misalignment, resin-rich, resin-starved
- **Fibre Textures**: Realistic woven patterns for aramid, carbon, and glass fibres
- **NDT Signatures**: Method-specific visual artifacts for all 9 NDT methods
- **Configurable**: Image size, defect severity, random seed control

### Model Training
```python
from cnn_defect_model import CompositeDefectDetector, CompositeDefectDataset, train_model
from torch.utils.data import DataLoader

model = CompositeDefectDetector()
dataset = CompositeDefectDataset(images, metadata)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

history = train_model(model, train_loader, val_loader, num_epochs=50)
```

## License
MIT License - For research and educational purposes.

## Citation
If you use this system in your research, please cite:
```
Composite NDT Defect Detection System v1.0.0
AI-Powered Multi-Modal Defect Analysis for Fibre-Reinforced Polymers
```

## Contact
For questions or contributions, please open an issue on the project repository.
