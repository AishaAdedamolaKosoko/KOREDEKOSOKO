#!/usr/bin/env python3
"""
=============================================================================
Composite NDT Defect Detection - Gradio Interactive Web Application
=============================================================================

An interactive web interface for detecting and analyzing defects in composite
materials using non-destructive testing (NDT) results.

Run with: python gradio_ndt_app.py

Features:
- Upload NDT result images for instant analysis
- Generate synthetic NDT samples for testing
- Interactive visualizations of defect detection results
- Comprehensive analysis reports with recommendations

Supported NDT Methods:
  Visual Testing, Ultrasonic, Magnetic Particle, Radiography,
  Liquid Penetrant, Eddy Current, Thermal/Infrared, Microwave, Vibration Analysis
=============================================================================
"""

import os
import sys
import json
import numpy as np
import cv2
import gradio as gr
from datetime import datetime
from typing import Tuple, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import tempfile

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from standalone script
from standalone_script.composite_ndt_detector import (
    SyntheticNDTGenerator, DefectAnalyzer,
    NDTMethod, DefectType, FibreType, Config
)


def create_analysis_report(image: np.ndarray, results: Dict) -> np.ndarray:
    """Create a comprehensive visual analysis report."""
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('NDT DEFECT ANALYSIS REPORT', fontsize=18, fontweight='bold', color='white', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25, left=0.05, right=0.95, top=0.93, bottom=0.05)

    # 1. Original image with heat map overlay
    ax1 = fig.add_subplot(gs[0, 0])
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb, alpha=0.7)
    heat_map = results['heat_map']
    heat_resized = cv2.resize(heat_map, (image.shape[1], image.shape[0]))
    ax1.imshow(heat_resized, cmap='hot', alpha=0.4)
    ax1.set_title('Input with Defect Heat Map', fontsize=11, fontweight='bold', color='white')
    ax1.axis('off')
    ax1.set_facecolor('#0f172a')

    # 2. NDT Method Classification
    ax2 = fig.add_subplot(gs[0, 1])
    ndt_data = results['ndt_method']['probabilities']
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ndt_data)))
    bars = ax2.barh(range(len(ndt_data)), list(ndt_data.values()), color=colors)
    ax2.set_yticks(range(len(ndt_data)))
    labels = [NDTMethod.DISPLAY_NAMES.get(k, k).replace(' (', '\n(') for k in ndt_data.keys()]
    ax2.set_yticklabels(labels, fontsize=7, color='white')
    ax2.set_xlabel('Confidence', color='white', fontsize=9)
    ax2.set_title('NDT Method Classification', fontsize=11, fontweight='bold', color='white')
    ax2.set_facecolor('#0f172a')
    ax2.tick_params(colors='white')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)

    # 3. Defect Type Classification
    ax3 = fig.add_subplot(gs[0, 2])
    defect_data = results['defect_type']['probabilities']
    colors_defect = [DefectType.SEVERITY_COLORS.get(k, '#888888') for k in defect_data.keys()]
    bars = ax3.bar(range(len(defect_data)), list(defect_data.values()), color=colors_defect)
    ax3.set_xticks(range(len(defect_data)))
    labels_d = [DefectType.DISPLAY_NAMES.get(k, k) for k in defect_data.keys()]
    ax3.set_xticklabels(labels_d, rotation=35, ha='right', fontsize=8, color='white')
    ax3.set_ylabel('Confidence', color='white', fontsize=9)
    ax3.set_title('Defect Type Classification', fontsize=11, fontweight='bold', color='white')
    ax3.set_facecolor('#0f172a')
    ax3.tick_params(colors='white')
    ax3.set_ylim(0, 1)

    # 4. Fibre Type & Severity Gauge
    ax4 = fig.add_subplot(gs[1, 0])
    fibre_data = results['fibre_type']['probabilities']
    colors_fibre = ['#FFD700', '#555555', '#E8E8E8']
    wedges, texts, autotexts = ax4.pie(
        fibre_data.values(),
        labels=[FibreType.DISPLAY_NAMES.get(k, k) for k in fibre_data.keys()],
        autopct='%1.1f%%', colors=colors_fibre, startangle=90,
        textprops={'color': 'white', 'fontsize': 10}
    )
    ax4.set_title('Fibre Type', fontsize=11, fontweight='bold', color='white')
    ax4.set_facecolor('#0f172a')

    # 5. Severity Gauge
    ax5 = fig.add_subplot(gs[1, 1])
    severity = results['severity']['value']
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    for i, t in enumerate(theta[:-1]):
        color = plt.cm.RdYlGn(1 - t / np.pi)
        ax5.plot([0, r * np.cos(t)], [0, r * np.sin(t)], color=color, linewidth=10, alpha=0.8)
    needle_angle = np.pi * (1 - severity)
    ax5.annotate('', xy=(0.75 * np.cos(needle_angle), 0.75 * np.sin(needle_angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=4))
    ax5.set_xlim(-1.4, 1.4)
    ax5.set_ylim(-0.4, 1.4)
    ax5.set_aspect('equal')
    ax5.axis('off')
    ax5.set_facecolor('#0f172a')
    ax5.text(0, -0.1, f"{results['severity']['level'].upper()}", ha='center',
            fontsize=16, fontweight='bold', color='white')
    ax5.text(0, -0.28, f"Score: {severity:.3f}/1.0", ha='center', fontsize=11, color='#aaaaaa')
    ax5.set_title('Severity Assessment', fontsize=11, fontweight='bold', color='white')

    # 6. Summary Panel
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_facecolor('#0f172a')

    status = "CRITICAL" if results['requires_action'] and severity > 0.6 else \
             "WARNING" if results['requires_action'] else "PASSED"
    status_color = '#EF4444' if status == "CRITICAL" else '#F59E0B' if status == "WARNING" else '#10B981'

    summary = f"""ANALYSIS SUMMARY

NDT Method:
  {NDTMethod.DISPLAY_NAMES.get(results['ndt_method']['prediction'], 'Unknown')}
  Confidence: {results['ndt_method']['confidence']:.2%}

Primary Defect:
  {DefectType.DISPLAY_NAMES.get(results['defect_type']['prediction'], 'None')}
  Confidence: {results['defect_type']['confidence']:.2%}

Fibre Material:
  {FibreType.DISPLAY_NAMES.get(results['fibre_type']['prediction'], 'Unknown')}
  Confidence: {results['fibre_type']['confidence']:.2%}

Status: {status}

Recommendation:
{"REPAIR REQUIRED" if status == "CRITICAL" else "INSPECT FURTHER" if status == "WARNING" else "NO ACTION NEEDED"}
"""
    rect = FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.03",
                         facecolor='#1e293b', edgecolor=status_color, linewidth=4)
    ax6.add_patch(rect)
    ax6.text(0.5, 0.5, summary, transform=ax6.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            family='monospace', color='white', linespacing=1.6)
    ax6.set_title('Analysis Summary', fontsize=11, fontweight='bold', color='white')

    # Save to buffer
    buf = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(buf.name, dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    report_img = cv2.imread(buf.name)
    os.unlink(buf.name)
    return report_img


def analyze_uploaded_image(image: np.ndarray, ndt_method_hint: str = "Auto-Detect",
                          fibre_hint: str = "Auto-Detect") -> Tuple:
    """Analyze an uploaded NDT image and return results."""
    if image is None:
        return None, "Please upload an image first.", None

    # Convert from RGB (Gradio) to BGR (OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initialize analyzer
    analyzer = DefectAnalyzer()

    # Run analysis
    results = analyzer.analyze(image_bgr)

    # Create visual report
    report_img = create_analysis_report(image_bgr, results)
    report_rgb = cv2.cvtColor(report_img, cv2.COLOR_BGR2RGB)

    # Generate text report
    text_report = f"""
{'='*60}
COMPOSITE NDT DEFECT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

1. NDT METHOD DETECTED
   Method: {NDTMethod.DISPLAY_NAMES.get(results['ndt_method']['prediction'], results['ndt_method']['prediction'])}
   Confidence: {results['ndt_method']['confidence']:.3f}

   All method probabilities:
"""
    for method, prob in sorted(results['ndt_method']['probabilities'].items(),
                                key=lambda x: -x[1]):
        text_report += f"     {NDTMethod.DISPLAY_NAMES.get(method, method):45s} {prob:.3f}\n"

    text_report += f"""
2. DEFECT CLASSIFICATION
   Primary Defect: {DefectType.DISPLAY_NAMES.get(results['defect_type']['prediction'], results['defect_type']['prediction'])}
   Confidence: {results['defect_type']['confidence']:.3f}

   All defect probabilities:
"""
    for defect, prob in sorted(results['defect_type']['probabilities'].items(),
                                key=lambda x: -x[1]):
        text_report += f"     {DefectType.DISPLAY_NAMES.get(defect, defect):45s} {prob:.3f}\n"

    text_report += f"""
3. FIBRE TYPE IDENTIFICATION
   Material: {FibreType.DISPLAY_NAMES.get(results['fibre_type']['prediction'], results['fibre_type']['prediction'])}
   Confidence: {results['fibre_type']['confidence']:.3f}

4. SEVERITY ASSESSMENT
   Level: {results['severity']['level']}
   Score: {results['severity']['value']:.3f} / 1.000

5. ACTION REQUIRED: {'YES - Immediate attention needed' if results['requires_action'] else 'NO - Material is sound'}

6. RECOMMENDATIONS
"""

    recommendations = {
        DefectType.NO_DEFECT: [
            "No defects detected. Material within quality standards.",
            "Continue routine inspection schedule.",
            "Document results for future baseline comparison."
        ],
        DefectType.DELAMINATION: [
            "CRITICAL: Layer separation compromises structural integrity.",
            "Perform ultrasonic C-scan for depth mapping.",
            "Consider resin injection or patch bonding repair.",
            "Replace component if severity > 0.7."
        ],
        DefectType.CRACK: [
            "CRITICAL: Crack propagation risk - immediate assessment needed.",
            "Use radiography for through-thickness crack sizing.",
            "Apply temporary reinforcement if load-bearing.",
            "Monitor growth with periodic ultrasonic inspection."
        ],
        DefectType.VOID: [
            "Porosity may affect mechanical properties.",
            "Recommended: X-ray CT for 3D void mapping.",
            "Evaluate against acceptance criteria (<2% void content).",
            "Consider local repair for clustered voids."
        ],
        DefectType.FIBRE_MISALIGNMENT: [
            "Fibre deviation may reduce strength.",
            "Recommended: Shearography for strain field analysis.",
            "Compare against +/- 5 degree design tolerance.",
            "Critical areas may need scarf repair."
        ],
        DefectType.RESIN_RICH: [
            "Excess resin indicates manufacturing inconsistency.",
            "Recommended: Ultrasonic velocity for resin content.",
            "Acceptable if <5% area fraction.",
            "Monitor for adjacent voids or dry areas."
        ],
        DefectType.RESIN_STARVED: [
            "Insufficient resin - inadequate fibre support.",
            "Recommended: Thermography for moisture assessment.",
            "High risk of fibre abrasion and degradation.",
            "Repair by resin infusion or seal coat application."
        ]
    }

    recs = recommendations.get(results['defect_type']['prediction'], recommendations[DefectType.NO_DEFECT])
    for rec in recs:
        text_report += f"   - {rec}\n"

    text_report += f"\n{'='*60}\n"

    return report_rgb, text_report, results['heat_map']


def generate_synthetic_sample(ndt_method: str, fibre_type: str,
                              defect_type: str, seed: int) -> Tuple:
    """Generate a synthetic NDT sample."""
    generator = SyntheticNDTGenerator(image_size=(512, 512), seed=seed)

    method_map = {
        "Visual Testing": NDTMethod.VISUAL_TESTING,
        "Ultrasonic (UT)": NDTMethod.ULTRASONIC,
        "Magnetic Particle (MPT)": NDTMethod.MAGNETIC_PARTICLE,
        "Radiography (RT)": NDTMethod.RADIOGRAPHY,
        "Liquid Penetrant (LPT)": NDTMethod.LIQUID_PENETRANT,
        "Eddy Current (ECT)": NDTMethod.EDDY_CURRENT,
        "Thermal/Infrared (IRT)": NDTMethod.THERMAL_INFRARED,
        "Microwave NDT": NDTMethod.MICROWAVE,
        "Vibration Analysis": NDTMethod.VIBRATION_ANALYSIS
    }

    fibre_map = {
        "Aramid (Kevlar)": FibreType.ARAMID,
        "Carbon (CFRP)": FibreType.CARBON,
        "Glass (GFRP)": FibreType.GLASS
    }

    defect_map = {
        "No Defect": DefectType.NO_DEFECT,
        "Delamination": DefectType.DELAMINATION,
        "Crack": DefectType.CRACK,
        "Void": DefectType.VOID,
        "Fibre Misalignment": DefectType.FIBRE_MISALIGNMENT,
        "Resin-Rich Area": DefectType.RESIN_RICH,
        "Resin-Starved Area": DefectType.RESIN_STARVED
    }

    method = method_map[ndt_method]
    fibre = fibre_map[fibre_type]
    defect = defect_map[defect_type]
    num_defects = 0 if defect == DefectType.NO_DEFECT else 1

    image, metadata = generator.generate_sample(method, fibre, defect, num_defects)

    # Convert BGR to RGB for Gradio
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    info = f"""
Synthetic NDT Sample Generated
==============================
NDT Method: {ndt_method}
Fibre Type: {fibre_type}
Defect Type: {defect_type}
Seed: {seed}

Use the Analyze tab to run ML analysis on this image.
"""

    return image_rgb, info


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_app() -> gr.Blocks:
    """Create the Gradio web application."""

    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .tab-nav {
        background-color: #1e293b !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
    .tab-nav button {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .tab-nav button.selected {
        color: #3b82f6 !important;
        border-bottom: 3px solid #3b82f6 !important;
    }
    .panel {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    .title {
        color: #60a5fa !important;
        text-align: center;
    }
    """

    with gr.Blocks(css=custom_css, title="Composite NDT Defect Detector") as app:
        gr.Markdown("""
        # Composite Material NDT Defect Detection & Analysis
        ### AI-Powered Defect Detection for Fibre-Reinforced Polymers
        
        Upload your NDT result images for instant analysis of defects including
        **delamination, cracks, voids, fibre misalignment, resin-rich, and resin-starved areas**.
        """)

        with gr.Tabs() as tabs:
            # ===== ANALYZE TAB =====
            with gr.TabItem("Analyze NDT Image", id="analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload NDT Image")
                        input_image = gr.Image(
                            label="NDT Result Image",
                            type="numpy",
                            height=400
                        )

                        with gr.Row():
                            ndt_hint = gr.Dropdown(
                                choices=["Auto-Detect"] + [NDTMethod.DISPLAY_NAMES[m] for m in NDTMethod.ALL],
                                value="Auto-Detect",
                                label="NDT Method (optional hint)"
                            )
                            fibre_hint = gr.Dropdown(
                                choices=["Auto-Detect"] + [FibreType.DISPLAY_NAMES[f] for f in FibreType.ALL],
                                value="Auto-Detect",
                                label="Fibre Type (optional hint)"
                            )

                        analyze_btn = gr.Button(
                            "Run Analysis",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Analysis Results")
                        with gr.Row():
                            report_image = gr.Image(
                                label="Analysis Report",
                                height=350
                            )
                            heat_map = gr.Image(
                                label="Defect Heat Map",
                                height=350
                            )
                        text_output = gr.Textbox(
                            label="Detailed Analysis Report",
                            lines=20,
                            max_lines=30
                        )

                # Examples
                gr.Markdown("### Example NDT Images (Click to Load)")
                example_gallery = gr.Examples(
                    examples=[],
                    inputs=[input_image],
                    label="Sample Images"
                )

            # ===== GENERATE TAB =====
            with gr.TabItem("Generate Synthetic Sample", id="generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Synthetic Sample Configuration")

                        gen_ndt = gr.Dropdown(
                            choices=[
                                "Visual Testing",
                                "Ultrasonic (UT)",
                                "Magnetic Particle (MPT)",
                                "Radiography (RT)",
                                "Liquid Penetrant (LPT)",
                                "Eddy Current (ECT)",
                                "Thermal/Infrared (IRT)",
                                "Microwave NDT",
                                "Vibration Analysis"
                            ],
                            value="Ultrasonic (UT)",
                            label="NDT Method"
                        )

                        gen_fibre = gr.Dropdown(
                            choices=[
                                "Aramid (Kevlar)",
                                "Carbon (CFRP)",
                                "Glass (GFRP)"
                            ],
                            value="Carbon (CFRP)",
                            label="Fibre Type"
                        )

                        gen_defect = gr.Dropdown(
                            choices=[
                                "No Defect",
                                "Delamination",
                                "Crack",
                                "Void",
                                "Fibre Misalignment",
                                "Resin-Rich Area",
                                "Resin-Starved Area"
                            ],
                            value="Delamination",
                            label="Defect Type"
                        )

                        gen_seed = gr.Slider(
                            minimum=0, maximum=1000, step=1, value=42,
                            label="Random Seed"
                        )

                        generate_btn = gr.Button(
                            "Generate Sample",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Generated Synthetic NDT Image")
                        gen_output = gr.Image(
                            label="Synthetic NDT Result",
                            height=400
                        )
                        gen_info = gr.Textbox(
                            label="Sample Information",
                            lines=10
                        )

                        with gr.Row():
                            send_to_analyze = gr.Button(
                                "Send to Analyze Tab",
                                variant="secondary"
                            )

            # ===== BATCH ANALYSIS TAB =====
            with gr.TabItem("Batch Analysis", id="batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Multiple Images")
                        batch_files = gr.File(
                            label="NDT Images (JPG, PNG, TIFF)",
                            file_count="multiple",
                            file_types=[".jpg", ".jpeg", ".png", ".tif", ".tiff"]
                        )
                        batch_btn = gr.Button(
                            "Run Batch Analysis",
                            variant="primary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Batch Results Summary")
                        batch_results = gr.Dataframe(
                            headers=["File", "NDT Method", "Defect Type",
                                    "Fibre Type", "Severity", "Action Needed"],
                            label="Analysis Summary",
                            wrap=True
                        )

            # ===== INFO TAB =====
            with gr.TabItem("About & Documentation", id="info"):
                gr.Markdown("""
                ## Composite NDT Defect Detection System

                ### Overview
                This system uses a Convolutional Neural Network (CNN) with attention mechanisms
                to automatically detect and classify defects in composite materials from
                non-destructive testing (NDT) result images.

                ### Supported NDT Methods

                | Method | Code | Best For |
                |--------|------|----------|
                | Visual Testing | VT | Surface defects, colour variations |
                | Ultrasonic Testing | UT | Internal delaminations, voids |
                | Magnetic Particle | MPT | Surface cracks in ferromagnetic materials |
                | Radiographic Testing | RT | Internal voids, inclusions, density changes |
                | Liquid Penetrant | LPT | Surface-breaking cracks |
                | Eddy Current | ECT | Conductivity changes, fibre orientation |
                | Thermal/Infrared | IRT | Subsurface delaminations, disbonds |
                | Microwave NDT | MW | Dielectric property changes, moisture |
                | Vibration Analysis | VA | Modal property changes, stiffness loss |

                ### Detectable Defects

                | Defect | Description | Criticality |
                |--------|-------------|-------------|
                | Delamination | Separation between composite layers | HIGH |
                | Crack | Fracture in matrix or fibres | HIGH |
                | Void | Air pocket/porosity | MEDIUM |
                | Fibre Misalignment | Deviation from design orientation | MEDIUM |
                | Resin-Rich | Excess resin accumulation | LOW |
                | Resin-Starved | Insufficient resin | MEDIUM |

                ### Fibre Types
                - **Aramid (Kevlar)**: High impact resistance, ballistic protection
                - **Carbon (CFRP)**: High stiffness, aerospace applications
                - **Glass (GFRP)**: Cost-effective, corrosion resistant

                ### Model Architecture
                - ResNet-style backbone with Squeeze-and-Excitation blocks
                - Feature Pyramid Network for multi-scale feature extraction
                - Multi-task heads for: NDT method, defect type, fibre type, severity, localization

                ### Usage Notes
                1. For best results, use clear NDT result images at least 256x256 pixels
                2. The system works with all major NDT method outputs
                3. Synthetic samples can be generated for testing and validation
                4. Batch analysis supports up to 50 images per run

                ### Version
                v1.0.0 - Composite NDT ML Analysis System
                """)

        # ===== EVENT HANDLERS =====

        analyze_btn.click(
            fn=analyze_uploaded_image,
            inputs=[input_image, ndt_hint, fibre_hint],
            outputs=[report_image, text_output, heat_map]
        )

        generate_btn.click(
            fn=generate_synthetic_sample,
            inputs=[gen_ndt, gen_fibre, gen_defect, gen_seed],
            outputs=[gen_output, gen_info]
        )

        def send_to_analyze_fn(gen_img, gen_info_text):
            return gen_img

        send_to_analyze.click(
            fn=send_to_analyze_fn,
            inputs=[gen_output, gen_info],
            outputs=[input_image]
        )

        def batch_analysis_fn(files):
            if not files:
                return []
            analyzer = DefectAnalyzer()
            results = []
            for f in files:
                try:
                    img = cv2.imread(f.name)
                    if img is not None:
                        res = analyzer.analyze(img)
                        results.append([
                            os.path.basename(f.name),
                            NDTMethod.DISPLAY_NAMES.get(res['ndt_method']['prediction'], res['ndt_method']['prediction']),
                            DefectType.DISPLAY_NAMES.get(res['defect_type']['prediction'], res['defect_type']['prediction']),
                            FibreType.DISPLAY_NAMES.get(res['fibre_type']['prediction'], res['fibre_type']['prediction']),
                            f"{res['severity']['level']} ({res['severity']['value']:.3f})",
                            "YES" if res['requires_action'] else "NO"
                        ])
                except Exception as e:
                    results.append([os.path.basename(f.name), "ERROR", str(e), "-", "-", "-"])
            return results

        batch_btn.click(
            fn=batch_analysis_fn,
            inputs=[batch_files],
            outputs=[batch_results]
        )

    return app


def main():
    """Launch the Gradio application."""
    print("=" * 60)
    print("Composite NDT Defect Detection - Gradio App")
    print("=" * 60)

    app = create_gradio_app()

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None
    )


if __name__ == "__main__":
    main()
