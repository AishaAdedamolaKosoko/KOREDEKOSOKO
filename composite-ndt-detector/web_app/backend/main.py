#!/usr/bin/env python3
"""
=============================================================================
Composite NDT Defect Detection - FastAPI Backend
=============================================================================

REST API for the composite material NDT defect detection system.

Endpoints:
  POST /api/analyze          - Analyze uploaded NDT image
  POST /api/generate         - Generate synthetic NDT sample
  POST /api/batch            - Batch analyze multiple images
  GET  /api/ndt-methods      - List supported NDT methods
  GET  /api/defect-types     - List detectable defect types
  GET  /api/fibre-types      - List supported fibre types
  GET  /api/health           - Health check

Run: uvicorn main:app --host 0.0.0.0 --port 8000
=============================================================================
"""

import os
import sys
import io
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from standalone_script.composite_ndt_detector import (
    SyntheticNDTGenerator, DefectAnalyzer,
    NDTMethod, DefectType, FibreType
)

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Composite NDT Defect Detection API",
    description="AI-powered defect detection for composite materials NDT",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
analyzer = DefectAnalyzer()
generator = SyntheticNDTGenerator(image_size=(512, 512), seed=42)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class NDTCapability(BaseModel):
    id: str
    name: str
    description: str
    best_for: List[str]

class DefectInfo(BaseModel):
    id: str
    name: str
    description: str
    criticality: str
    color: str

class FibreInfo(BaseModel):
    id: str
    name: str
    description: str
    applications: List[str]

class GenerateRequest(BaseModel):
    ndt_method: str
    fibre_type: str
    defect_type: str
    seed: Optional[int] = 42

class AnalysisResponse(BaseModel):
    success: bool
    timestamp: str
    image_shape: List[int]
    ndt_method: Dict[str, Any]
    defect_type: Dict[str, Any]
    fibre_type: Dict[str, Any]
    severity: Dict[str, Any]
    requires_action: bool
    heat_map_base64: Optional[str] = None
    report_base64: Optional[str] = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

SUPPORTED_EXTENSIONS = {
    "image": {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "gif", "webp"},
    "spreadsheet": {"xls", "xlsx", "csv", "ods"},
    "database": {"mdb", "accdb", "accde", "mde", "ldb"},
    "video": {"mp4", "mov", "avi", "mkv", "wmv", "flv", "webm"},
    "simulation": {"inp", "odb", "dat", "sim", "cas", "rst", "fem", "ans", "bdf", "nas"},
}


def classify_upload(filename: str, contents: bytes) -> Dict[str, Any]:
    """Classify uploaded content by file type and format for richer handling."""
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    name = (filename or "").lower()

    if suffix in SUPPORTED_EXTENSIONS["image"] or any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"]):
        return {"kind": "image", "format": suffix or "image"}

    if suffix in SUPPORTED_EXTENSIONS["spreadsheet"] or suffix in {"xls", "xlsx", "csv", "ods"}:
        return {"kind": "spreadsheet", "format": "excel" if suffix in {"xls", "xlsx"} else suffix}

    if suffix in SUPPORTED_EXTENSIONS["database"] or name.endswith((".mdb", ".accdb", ".accde", ".mde", ".ldb")):
        return {"kind": "database", "format": "access"}

    if suffix in SUPPORTED_EXTENSIONS["video"] or name.endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm")):
        return {"kind": "video", "format": "video"}

    if suffix in SUPPORTED_EXTENSIONS["simulation"] or any(marker in contents[:256].lower() for marker in [b"*heading", b"*node", b"*element", b"*step", b"*begin"]):
        if suffix in {"inp", "dat", "fem", "sim", "cas", "ans", "bdf", "nas"} or "inp" in name or "abaqus" in name:
            return {"kind": "simulation", "format": "abaqus"}
        return {"kind": "simulation", "format": "ansys"}

    if contents.startswith(b"PK\x03\x04"):
        return {"kind": "spreadsheet", "format": "excel"}

    return {"kind": "unknown", "format": suffix or "unknown"}


def image_to_base64(image: np.ndarray, format: str = "png") -> str:
    """Convert numpy image to base64 string."""
    _, buffer = cv2.imencode(f".{format}", image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy image."""
    buffer = base64.b64decode(base64_str)
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/ndt-methods")
async def get_ndt_methods():
    """Get list of supported NDT methods."""
    methods = [
        {
            "id": m,
            "name": NDTMethod.DISPLAY_NAMES[m],
            "best_for": _get_ndt_best_for(m)
        }
        for m in NDTMethod.ALL
    ]
    return {"methods": methods}


def _get_ndt_best_for(method: str) -> List[str]:
    """Get what each NDT method is best for."""
    best_for = {
        NDTMethod.VISUAL_TESTING: ["Surface defects", "Colour variations", "Visible damage"],
        NDTMethod.ULTRASONIC: ["Delaminations", "Voids", "Internal defects", "Thickness measurement"],
        NDTMethod.MAGNETIC_PARTICLE: ["Surface cracks", "Near-surface defects"],
        NDTMethod.RADIOGRAPHY: ["Voids", "Inclusions", "Density changes", "Internal structure"],
        NDTMethod.LIQUID_PENETRANT: ["Surface-breaking cracks", "Porosity at surface"],
        NDTMethod.EDDY_CURRENT: ["Conductivity changes", "Fibre orientation", "Near-surface defects"],
        NDTMethod.THERMAL_INFRARED: ["Subsurface delaminations", "Disbonds", "Moisture ingress"],
        NDTMethod.MICROWAVE: ["Dielectric changes", "Moisture content", "Internal defects in dielectrics"],
        NDTMethod.VIBRATION_ANALYSIS: ["Stiffness changes", "Modal property changes", "Structural damage"]
    }
    return best_for.get(method, ["General defect detection"])


@app.get("/api/defect-types")
async def get_defect_types():
    """Get list of detectable defect types."""
    defects = [
        {
            "id": d,
            "name": DefectType.DISPLAY_NAMES[d],
            "criticality": _get_defect_criticality(d),
            "color": DefectType.SEVERITY_COLORS.get(d, "#888888"),
            "description": _get_defect_description(d)
        }
        for d in DefectType.ALL
    ]
    return {"defect_types": defects}


def _get_defect_criticality(defect: str) -> str:
    criticality = {
        DefectType.NO_DEFECT: "None",
        DefectType.DELAMINATION: "High",
        DefectType.CRACK: "High",
        DefectType.VOID: "Medium",
        DefectType.FIBRE_MISALIGNMENT: "Medium",
        DefectType.RESIN_RICH: "Low",
        DefectType.RESIN_STARVED: "Medium"
    }
    return criticality.get(defect, "Unknown")


def _get_defect_description(defect: str) -> str:
    descriptions = {
        DefectType.NO_DEFECT: "No defects detected. Material is within acceptable quality standards.",
        DefectType.DELAMINATION: "Separation between composite layers due to weak bonding, reducing structural integrity.",
        DefectType.CRACK: "Fracture in the matrix or fibres that can propagate under load.",
        DefectType.VOID: "Air pocket or porosity within the composite material.",
        DefectType.FIBRE_MISALIGNMENT: "Deviation of fibres from the designed orientation.",
        DefectType.RESIN_RICH: "Excess resin accumulation in localized areas.",
        DefectType.RESIN_STARVED: "Insufficient resin leaving fibres inadequately supported."
    }
    return descriptions.get(defect, "Unknown defect type")


@app.get("/api/fibre-types")
async def get_fibre_types():
    """Get list of supported fibre types."""
    fibres = [
        {
            "id": f,
            "name": FibreType.DISPLAY_NAMES[f],
            "description": _get_fibre_description(f),
            "applications": _get_fibre_applications(f)
        }
        for f in FibreType.ALL
    ]
    return {"fibre_types": fibres}


def _get_fibre_description(fibre: str) -> str:
    descriptions = {
        FibreType.ARAMID: "Aromatic polyamide fibre with high strength-to-weight ratio and excellent impact resistance.",
        FibreType.CARBON: "High-stiffness fibre with excellent strength and low weight, widely used in aerospace.",
        FibreType.GLASS: "Cost-effective fibre with good strength and corrosion resistance."
    }
    return descriptions.get(fibre, "Unknown fibre type")


def _get_fibre_applications(fibre: str) -> List[str]:
    apps = {
        FibreType.ARAMID: ["Ballistic protection", "Aerospace", "Marine", "Sports equipment"],
        FibreType.CARBON: ["Aerospace structures", "Automotive", "Wind turbine blades", "Sporting goods"],
        FibreType.GLASS: ["Boat hulls", "Storage tanks", "Pipes", "Construction", "Automotive panels"]
    }
    return apps.get(fibre, ["General composite applications"])


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    ndt_hint: Optional[str] = Form("auto"),
    fibre_hint: Optional[str] = Form("auto")
):
    """Analyze an uploaded NDT artifact, including image and non-image file formats."""
    try:
        contents = await file.read()
        classification = classify_upload(file.filename or "", contents)

        if classification["kind"] == "image":
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Could not decode image file")

            results = analyzer.analyze(image)
            heat_map = results['heat_map']
            heat_colored = plt.cm.hot(heat_map)[:, :, :3]
            heat_colored = (heat_colored * 255).astype(np.uint8)
            heat_bgr = cv2.cvtColor(heat_colored, cv2.COLOR_RGB2BGR)
            heat_resized = cv2.resize(heat_bgr, (image.shape[1], image.shape[0]))
            overlay = cv2.addWeighted(image, 0.6, heat_resized, 0.4, 0)

            response = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "file_type": classification,
                "image_shape": list(image.shape),
                "ndt_method": results['ndt_method'],
                "defect_type": results['defect_type'],
                "fibre_type": results['fibre_type'],
                "severity": results['severity'],
                "requires_action": results['requires_action'],
                "heat_map_base64": image_to_base64(heat_bgr),
                "overlay_base64": image_to_base64(overlay)
            }
        else:
            response = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "file_type": classification,
                "message": "Non-image artifact received. The system will treat this as a structured engineering dataset for review.",
                "ndt_method": {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "probabilities": {m: 0.0 for m in NDTMethod.ALL}
                },
                "defect_type": {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "probabilities": {d: 0.0 for d in DefectType.ALL}
                },
                "fibre_type": {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "probabilities": {f: 0.0 for f in FibreType.ALL}
                },
                "severity": {"value": 0.0, "level": "Minor"},
                "requires_action": False,
                "heat_map_base64": None,
                "overlay_base64": None
            }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate_synthetic(request: GenerateRequest):
    """Generate a synthetic NDT sample."""
    try:
        # Map request to internal types
        method_map = {v: k for k, v in NDTMethod.DISPLAY_NAMES.items()}
        fibre_map = {v: k for k, v in FibreType.DISPLAY_NAMES.items()}
        defect_map = {v: k for k, v in DefectType.DISPLAY_NAMES.items()}

        # Find matching internal keys
        method = None
        for k, v in NDTMethod.DISPLAY_NAMES.items():
            if request.ndt_method.lower() in v.lower() or request.ndt_method == k:
                method = k
                break
        if not method:
            method = NDTMethod.ULTRASONIC

        fibre = None
        for k, v in FibreType.DISPLAY_NAMES.items():
            if request.fibre_type.lower() in v.lower() or request.fibre_type == k:
                fibre = k
                break
        if not fibre:
            fibre = FibreType.CARBON

        defect = None
        for k, v in DefectType.DISPLAY_NAMES.items():
            if request.defect_type.lower() in v.lower() or request.defect_type == k:
                defect = k
                break
        if not defect:
            defect = DefectType.NO_DEFECT

        # Generate
        num_defects = 0 if defect == DefectType.NO_DEFECT else 1
        gen = SyntheticNDTGenerator(image_size=(512, 512), seed=request.seed or 42)
        image, metadata = gen.generate_sample(method, fibre, defect, num_defects)

        return JSONResponse(content={
            "success": True,
            "image_base64": image_to_base64(image),
            "metadata": metadata
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """Analyze multiple NDT artifacts in batch."""
    results = []

    for file in files:
        try:
            contents = await file.read()
            classification = classify_upload(file.filename or "", contents)
            if classification["kind"] == "image":
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is not None:
                    res = analyzer.analyze(image)
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "file_type": classification,
                        "ndt_method": res['ndt_method']['prediction'],
                        "ndt_method_confidence": res['ndt_method']['confidence'],
                        "defect_type": res['defect_type']['prediction'],
                        "defect_type_confidence": res['defect_type']['confidence'],
                        "fibre_type": res['fibre_type']['prediction'],
                        "severity_value": res['severity']['value'],
                        "severity_level": res['severity']['level'],
                        "requires_action": res['requires_action']
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "file_type": classification,
                        "error": "Could not decode image"
                    })
            else:
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "file_type": classification,
                    "message": "Received non-image engineering artifact for review"
                })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return JSONResponse(content={
        "success": True,
        "total": len(files),
        "processed": len([r for r in results if r.get("success")]),
        "results": results
    })


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Composite NDT Defect Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "ndt_methods": "/api/ndt-methods",
            "defect_types": "/api/defect-types",
            "fibre_types": "/api/fibre-types",
            "analyze": "POST /api/analyze",
            "generate": "POST /api/generate",
            "batch": "POST /api/batch"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
