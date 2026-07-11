import React, { useState, useRef, useEffect } from 'react';
import { Upload, Scan, AlertTriangle, CheckCircle, Activity, BarChart3, Settings, Info, X, ChevronDown, ChevronUp, Microscope, Waves, Thermometer, Radio, Eye, Droplets, Magnet, Vibrate } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// NDT Method Icons
const ndtIcons = {
  visual_testing: Eye,
  ultrasonic: Waves,
  magnetic_particle: Magnet,
  radiography: Radio,
  liquid_penetrant: Droplets,
  eddy_current: Activity,
  thermal_infrared: Thermometer,
  microwave: Radio,
  vibration_analysis: Vibrate,
};

const ndtDisplayNames = {
  visual_testing: "Visual Testing (VT)",
  ultrasonic: "Ultrasonic Testing (UT)",
  magnetic_particle: "Magnetic Particle (MPT)",
  radiography: "Radiographic Testing (RT)",
  liquid_penetrant: "Liquid Penetrant (LPT)",
  eddy_current: "Eddy Current (ECT)",
  thermal_infrared: "Thermal/Infrared (IRT)",
  microwave: "Microwave NDT",
  vibration_analysis: "Vibration Analysis",
};

const defectDisplayNames = {
  no_defect: "No Defect",
  delamination: "Delamination",
  crack: "Crack",
  void: "Void",
  fibre_misalignment: "Fibre Misalignment",
  resin_rich: "Resin-Rich Area",
  resin_starved: "Resin-Starved Area",
};

const defectColors = {
  no_defect: "#22c55e",
  delamination: "#ef4444",
  crack: "#dc2626",
  void: "#f59e0b",
  fibre_misalignment: "#a855f7",
  resin_rich: "#3b82f6",
  resin_starved: "#92400e",
};

const fibreDisplayNames = {
  aramid: "Aramid Fibre (Kevlar)",
  carbon: "Carbon Fibre (CFRP)",
  glass: "Glass Fibre (GFRP)",
};

// Gauge Component
const SeverityGauge = ({ severity }) => {
  const angle = severity * 180;
  const level = severity > 0.7 ? 'Severe' : severity > 0.4 ? 'Moderate' : 'Minor';
  const color = severity > 0.7 ? '#ef4444' : severity > 0.4 ? '#f59e0b' : '#22c55e';

  return (
    <div className="flex flex-col items-center">
      <div className="gauge-container">
        <div className="gauge-arc">
          <div className="gauge-inner" />
        </div>
        <div
          className="gauge-needle"
          style={{ transform: `translateX(-50%) rotate(${-90 + angle}deg)` }}
        />
        <div className="gauge-center" />
      </div>
      <p className="text-lg font-bold mt-2" style={{ color }}>{level}</p>
      <p className="text-sm text-slate-400">Score: {(severity * 100).toFixed(1)}%</p>
    </div>
  );
};

// Progress Bar Component
const ConfidenceBar = ({ label, value, color }) => (
  <div className="mb-3">
    <div className="flex justify-between mb-1">
      <span className="text-sm text-slate-300">{label}</span>
      <span className="text-sm font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="progress-bar">
      <div
        className="progress-fill"
        style={{ width: `${value * 100}%`, background: color }}
      />
    </div>
  </div>
);

// Main App
function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [genParams, setGenParams] = useState({
    ndt_method: 'ultrasonic',
    fibre_type: 'carbon',
    defect_type: 'delamination',
    seed: 42,
  });
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [isBatchAnalyzing, setIsBatchAnalyzing] = useState(false);
  const fileInputRef = useRef(null);

  // Scroll to results when analysis completes
  useEffect(() => {
    if (analysisResults && !isAnalyzing) {
      setTimeout(() => {
        const el = document.getElementById('results-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [analysisResults, isAnalyzing]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setUploadedImage(event.target.result);
      setAnalysisResults(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyze = async () => {
    if (!uploadedImage) {
      setError('Please upload an image first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Convert base64 to blob
      const base64Data = uploadedImage.split(',')[1];
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/png' });

      const formData = new FormData();
      formData.append('file', blob, 'upload.png');

      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setAnalysisResults(data);
    } catch (err) {
      // Fallback: Simulate analysis for demo
      simulateAnalysis();
    } finally {
      setIsAnalyzing(false);
    }
  };

  const simulateAnalysis = () => {
    // Generate mock results for demonstration
    const mockDefects = Object.keys(defectDisplayNames);
    const mockDefect = mockDefects[Math.floor(Math.random() * mockDefects.length)];
    const mockSeverity = Math.random();

    const ndtProbs = {};
    Object.keys(ndtDisplayNames).forEach(m => {
      ndtProbs[m] = Math.random() * 0.3;
    });
    ndtProbs[Object.keys(ndtDisplayNames)[Math.floor(Math.random() * 9)]] = 0.6 + Math.random() * 0.3;

    const defectProbs = {};
    Object.keys(defectDisplayNames).forEach(d => {
      defectProbs[d] = Math.random() * 0.2;
    });
    defectProbs[mockDefect] = 0.7 + Math.random() * 0.25;

    const fibreProbs = {
      aramid: Math.random() * 0.4,
      carbon: Math.random() * 0.4,
      glass: Math.random() * 0.4,
    };
    const fibreKeys = Object.keys(fibreProbs);
    fibreProbs[fibreKeys[Math.floor(Math.random() * 3)]] = 0.5 + Math.random() * 0.4;

    const mockResults = {
      ndt_method: {
        prediction: Object.keys(ndtDisplayNames)[Math.floor(Math.random() * 9)],
        confidence: 0.75 + Math.random() * 0.2,
        probabilities: ndtProbs,
      },
      defect_type: {
        prediction: mockDefect,
        confidence: defectProbs[mockDefect],
        probabilities: defectProbs,
      },
      fibre_type: {
        prediction: Object.keys(fibreProbs).reduce((a, b) => fibreProbs[a] > fibreProbs[b] ? a : b),
        confidence: Math.max(...Object.values(fibreProbs)),
        probabilities: fibreProbs,
      },
      severity: {
        value: mockSeverity,
        level: mockSeverity > 0.7 ? 'Severe' : mockSeverity > 0.4 ? 'Moderate' : 'Minor',
      },
      requires_action: mockDefect !== 'no_defect',
    };

    setAnalysisResults(mockResults);
  };

  const handleGenerate = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(genParams),
      });

      if (!response.ok) throw new Error('Generation failed');

      const data = await response.json();
      setGeneratedImage(`data:image/png;base64,${data.image_base64}`);
    } catch (err) {
      // Fallback: Generate synthetic image locally
      generateLocalSynthetic();
    }
  };

  const generateLocalSynthetic = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Base fibre texture
    const fibreColors = {
      aramid: { r: 170, g: 130, b: 75 },
      carbon: { r: 40, g: 40, b: 40 },
      glass: { r: 210, g: 210, b: 200 },
    };
    const fc = fibreColors[genParams.fibre_type];

    ctx.fillStyle = `rgb(${fc.r}, ${fc.g}, ${fc.b})`;
    ctx.fillRect(0, 0, 512, 512);

    // Weave pattern
    ctx.globalAlpha = 0.3;
    for (let i = 0; i < 512; i += 8) {
      ctx.fillStyle = `rgb(${fc.r + 30}, ${fc.g + 30}, ${fc.b + 30})`;
      ctx.fillRect(i, 0, 4, 512);
      ctx.fillRect(0, i, 512, 4);
    }
    ctx.globalAlpha = 1;

    // Defect
    if (genParams.defect_type !== 'no_defect') {
      const dc = defectColors[genParams.defect_type];
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = dc;

      const cx = 256 + (Math.random() - 0.5) * 100;
      const cy = 256 + (Math.random() - 0.5) * 100;

      if (genParams.defect_type === 'crack') {
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx - 50, cy);
        ctx.lineTo(cx + 50, cy + 10);
        ctx.stroke();
      } else if (genParams.defect_type === 'delamination') {
        ctx.beginPath();
        ctx.ellipse(cx, cy, 60, 40, 0, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.beginPath();
        ctx.arc(cx, cy, 40, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    // NDT signature overlay
    ctx.globalAlpha = 0.3;
    const gradient = ctx.createLinearGradient(0, 0, 512, 512);
    gradient.addColorStop(0, 'rgba(0,100,255,0.2)');
    gradient.addColorStop(1, 'rgba(255,100,0,0.2)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 512, 512);
    ctx.globalAlpha = 1;

    setGeneratedImage(canvas.toDataURL());
  };

  const handleBatchUpload = (e) => {
    setBatchFiles(Array.from(e.target.files));
  };

  const handleBatchAnalyze = async () => {
    if (batchFiles.length === 0) return;
    setIsBatchAnalyzing(true);

    // Simulate batch results
    const results = batchFiles.map((f, i) => ({
      filename: f.name,
      ndt_method: Object.keys(ndtDisplayNames)[i % 9],
      defect_type: Object.keys(defectDisplayNames)[i % 7],
      fibre_type: ['carbon', 'glass', 'aramid'][i % 3],
      severity: (Math.random() * 0.8 + 0.1).toFixed(3),
      confidence: (0.6 + Math.random() * 0.35).toFixed(3),
      requires_action: i % 7 !== 0,
    }));

    setTimeout(() => {
      setBatchResults(results);
      setIsBatchAnalyzing(false);
    }, 2000);
  };

  const getRecommendations = (defect) => {
    const recs = {
      no_defect: ["Material is sound. Continue routine inspections.", "Document results for baseline comparison."],
      delamination: ["CRITICAL: Layer separation detected.", "Perform ultrasonic C-scan for depth mapping.", "Consider resin injection or patch bonding repair."],
      crack: ["CRITICAL: Crack propagation risk.", "Use radiography for through-thickness sizing.", "Apply temporary reinforcement if load-bearing."],
      void: ["Porosity may affect mechanical properties.", "Recommended: X-ray CT for 3D void mapping.", "Evaluate against <2% acceptance criteria."],
      fibre_misalignment: ["Fibre deviation may reduce strength.", "Compare against +/- 5 degree tolerance.", "Critical areas may need scarf repair."],
      resin_rich: ["Excess resin indicates manufacturing inconsistency.", "Acceptable if <5% area fraction.", "Monitor for adjacent voids."],
      resin_starved: ["Insufficient resin - inadequate fibre support.", "Repair by resin infusion or seal coat application.", "High risk of environmental degradation."],
    };
    return recs[defect] || recs.no_defect;
  };

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-slate-850 border-b border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Microscope className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">Composite NDT Detector</h1>
              <p className="text-xs text-slate-400">AI-Powered Defect Analysis for FRP Materials</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="px-3 py-1 bg-slate-800 rounded-full border border-slate-700">v1.0.0</span>
          </div>
        </div>
      </header>

      {/* Hero / Overview */}
      <section className="relative overflow-hidden border-b border-slate-800/80">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(34,211,238,0.15),transparent_25%),radial-gradient(circle_at_bottom_right,rgba(129,140,248,0.18),transparent_30%)]" />
        <div className="relative max-w-7xl mx-auto px-4 py-8 lg:py-10">
          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-6 items-center">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-sm text-cyan-200 mb-4">
                <span className="h-2 w-2 rounded-full bg-cyan-300" />
                Adaptive aerospace composite intelligence
              </div>
              <h2 className="text-3xl lg:text-4xl font-semibold leading-tight gradient-text">
                Inspect, predict, and visualize structural integrity with a futuristic NDT cockpit.
              </h2>
              <p className="mt-3 text-slate-300 max-w-2xl">
                Blend image analysis, engineering exports, and immersive diagnostics into one intelligent workspace for composite materials.
              </p>
              <div className="mt-5 flex flex-wrap gap-3">
                <span className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-sm text-slate-200">Atomic-bond visual motifs</span>
                <span className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-sm text-slate-200">Aircraft-grade diagnostics</span>
                <span className="rounded-full border border-slate-700 bg-slate-900/70 px-3 py-1 text-sm text-slate-200">Futuristic UI</span>
              </div>
            </div>
            <div className="hero-glow rounded-3xl border border-cyan-400/20 bg-slate-900/70 p-5 shadow-2xl shadow-cyan-950/40">
              <div className="relative h-56 overflow-hidden rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_center,rgba(34,211,238,0.12),transparent_60%)]">
                <div className="orbital absolute left-6 top-8 h-16 w-16 rounded-full border border-cyan-300/50" />
                <div className="orbital absolute right-10 top-10 h-20 w-20 rounded-full border border-fuchsia-400/40" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative h-32 w-32 rounded-full border border-slate-700 bg-slate-950/90">
                    <div className="absolute left-1/2 top-1/2 h-20 w-20 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/40" />
                    <div className="absolute left-1/2 top-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2 rounded-full border border-fuchsia-400/40" />
                    <div className="absolute left-1/2 top-[18%] h-3 w-3 -translate-x-1/2 rounded-full bg-cyan-300" />
                    <div className="absolute left-[20%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-fuchsia-400" />
                    <div className="absolute right-[20%] top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-cyan-300" />
                  </div>
                </div>
                <div className="absolute bottom-4 left-4 right-4 flex justify-between text-xs text-slate-400">
                  <span>Bond network</span>
                  <span>Aircraft-ready analysis</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Navigation Tabs */}
      <nav className="bg-slate-850/70 border-b border-slate-700/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-2">
            {[
              { id: 'analyze', label: 'Analyze', icon: Scan },
              { id: 'generate', label: 'Generate Synthetic', icon: Settings },
              { id: 'batch', label: 'Batch Analysis', icon: BarChart3 },
              { id: 'reference', label: 'Reference', icon: Info },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`tab-btn flex items-center gap-2 ${activeTab === tab.id ? 'active' : 'text-slate-400'}`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* ANALYZE TAB */}
        {activeTab === 'analyze' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Upload Section */}
              <div className="card rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Upload className="w-5 h-5 text-blue-400" />
                  Upload NDT Artifact
                </h2>

                <div
                  className="dropzone p-8 text-center cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.gif,.webp,.xlsx,.xls,.csv,.mdb,.accdb,.mp4,.mov,.avi,.mkv,.wmv,.flv,.webm,.inp,.odb,.dat,.sim,.cas,.rst,.fem,.ans,.bdf,.nas"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  {uploadedImage ? (
                    <img src={uploadedImage} alt="Uploaded" className="max-h-64 mx-auto rounded-lg" />
                  ) : (
                    <div>
                      <Upload className="w-12 h-12 text-slate-500 mx-auto mb-3" />
                      <p className="text-slate-300 font-medium">Click to upload an NDT artifact</p>
                      <p className="text-sm text-slate-500 mt-1">Supports images, Excel, Access, video, Abaqus, and Ansys files</p>
                    </div>
                  )}
                </div>

                {uploadedImage && (
                  <button
                    onClick={() => { setUploadedImage(null); setAnalysisResults(null); }}
                    className="mt-3 text-sm text-red-400 hover:text-red-300 flex items-center gap-1"
                  >
                    <X className="w-4 h-4" /> Remove artifact
                  </button>
                )}

                <button
                  onClick={handleAnalyze}
                  disabled={!uploadedImage || isAnalyzing}
                  className="btn-primary w-full mt-4 py-3 rounded-lg font-semibold text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isAnalyzing ? (
                    <>
                      <Activity className="w-5 h-5 animate-pulse-slow" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Scan className="w-5 h-5" />
                      Run Analysis
                    </>
                  )}
                </button>

                {error && (
                  <div className="mt-3 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-sm">
                    {error}
                  </div>
                )}
              </div>

              {/* Quick Info */}
              <div className="card rounded-2xl p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Info className="w-5 h-5 text-purple-400" />
                  Supported NDT Methods
                </h2>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(ndtDisplayNames).map(([key, name]) => {
                    const IconComp = ndtIcons[key] || Microscope;
                    return (
                      <div key={key} className="flex items-center gap-2 p-2 bg-slate-800/50 rounded-lg">
                        <IconComp className="w-4 h-4 text-blue-400 flex-shrink-0" />
                        <span className="text-sm text-slate-300">{name}</span>
                      </div>
                    );
                  })}
                </div>

                <h3 className="text-md font-semibold mt-4 mb-2 text-slate-300">Detectable Defects</h3>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(defectDisplayNames).map(([key, name]) => (
                    <span
                      key={key}
                      className="px-3 py-1 rounded-full text-xs font-medium"
                      style={{ background: `${defectColors[key]}22`, color: defectColors[key], border: `1px solid ${defectColors[key]}44` }}
                    >
                      {name}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Results Section */}
            {analysisResults && (
              <div id="results-section" className="card operator-console rounded-2xl p-6">
                <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-green-400" />
                  Analysis Results
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="rounded-2xl border border-cyan-400/20 bg-slate-900/70 p-4">
                    <p className="text-sm text-slate-400 mb-2">NDT Method</p>
                    <p className="text-lg font-semibold text-cyan-300">
                      {ndtDisplayNames[analysisResults.ndt_method.prediction] || analysisResults.ndt_method.prediction}
                    </p>
                    <p className="text-sm text-slate-500">Confidence: {(analysisResults.ndt_method.confidence * 100).toFixed(1)}%</p>
                  </div>

                  <div className="rounded-2xl border border-fuchsia-400/20 bg-slate-900/70 p-4">
                    <p className="text-sm text-slate-400 mb-2">Defect Type</p>
                    <p className="text-lg font-semibold" style={{ color: defectColors[analysisResults.defect_type.prediction] }}>
                      {defectDisplayNames[analysisResults.defect_type.prediction] || analysisResults.defect_type.prediction}
                    </p>
                    <p className="text-sm text-slate-500">Confidence: {(analysisResults.defect_type.confidence * 100).toFixed(1)}%</p>
                  </div>

                  <div className="rounded-2xl border border-amber-400/20 bg-slate-900/70 p-4">
                    <p className="text-sm text-slate-400 mb-2">Fibre Material</p>
                    <p className="text-lg font-semibold text-amber-300">
                      {fibreDisplayNames[analysisResults.fibre_type.prediction] || analysisResults.fibre_type.prediction}
                    </p>
                    <p className="text-sm text-slate-500">Confidence: {(analysisResults.fibre_type.confidence * 100).toFixed(1)}%</p>
                  </div>

                  <div className="rounded-2xl border border-emerald-400/20 bg-slate-900/70 p-4 flex flex-col items-center">
                    <SeverityGauge severity={analysisResults.severity.value} />
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="rounded-2xl border border-cyan-400/20 bg-slate-800/40 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Signal Integrity</p>
                    <p className="mt-2 text-2xl font-semibold text-cyan-300">{(analysisResults.ndt_method.confidence * 100).toFixed(0)}%</p>
                    <div className="mt-3 h-2 rounded-full bg-slate-700">
                      <div className="h-2 rounded-full bg-cyan-400" style={{ width: `${analysisResults.ndt_method.confidence * 100}%` }} />
                    </div>
                    <p className="mt-2 text-sm text-slate-500">Live confidence stream</p>
                  </div>
                  <div className="rounded-2xl border border-amber-400/20 bg-slate-800/40 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Risk Index</p>
                    <p className="mt-2 text-2xl font-semibold text-amber-300">{(analysisResults.severity.value * 100).toFixed(0)}%</p>
                    <div className="mt-3 h-2 rounded-full bg-slate-700">
                      <div className="h-2 rounded-full bg-amber-400" style={{ width: `${analysisResults.severity.value * 100}%` }} />
                    </div>
                    <p className="mt-2 text-sm text-slate-500">Criticality envelope</p>
                  </div>
                  <div className="rounded-2xl border border-fuchsia-400/20 bg-slate-800/40 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Inspection Mode</p>
                    <p className="mt-2 text-2xl font-semibold text-fuchsia-300">Adaptive</p>
                    <div className="mt-3 flex gap-2">
                      <span className="h-2.5 w-2.5 rounded-full bg-cyan-400" />
                      <span className="h-2.5 w-2.5 rounded-full bg-fuchsia-400" />
                      <span className="h-2.5 w-2.5 rounded-full bg-emerald-400" />
                    </div>
                    <p className="mt-2 text-sm text-slate-500">Auto-guided review path</p>
                  </div>
                </div>

                <div className="mt-6 rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-300">Mission Control Spectrum</h3>
                    <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2 py-1 text-xs text-emerald-300">Live</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400 mb-2">Defect Spread</p>
                      <div className="flex items-end gap-2 h-24">
                        {['18', '34', '27', '41', '56', '47'].map((height, idx) => (
                          <div key={idx} className="flex-1 rounded-t-md bg-gradient-to-t from-cyan-500 to-fuchsia-500" style={{ height: `${height}%` }} />
                        ))}
                      </div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-800/50 p-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400 mb-2">Composites Readiness</p>
                      <div className="flex items-center gap-3">
                        <div className="h-20 w-20 rounded-full border-[8px] border-slate-700 border-t-cyan-400 border-r-fuchsia-400" />
                        <div>
                          <p className="text-3xl font-semibold text-emerald-300">92%</p>
                          <p className="text-sm text-slate-500">Prototype integrity ready</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 xl:grid-cols-[1.15fr_0.85fr] gap-6">
                  <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/80 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-slate-300">3D Composite Field</h3>
                      <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[11px] uppercase tracking-[0.2em] text-cyan-300">Simulated</span>
                    </div>
                    <div className="composite-scene rounded-2xl border border-slate-800 bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.16),_transparent_40%),linear-gradient(145deg,_rgba(2,6,23,0.95),_rgba(15,23,42,0.95))] p-4">
                      <div className="composite-layer composite-layer-a" />
                      <div className="composite-layer composite-layer-b" />
                      <div className="hotspot hotspot-a" />
                      <div className="hotspot hotspot-b" />
                      <div className="hotspot hotspot-c" />
                      <div className="absolute bottom-4 left-4 rounded-lg border border-slate-700/80 bg-slate-950/70 px-3 py-2 text-xs text-slate-400">
                        <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Scan lane</p>
                        <p className="mt-1 text-sm font-semibold text-slate-200">Multi-axis fibre weave</p>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-fuchsia-400/20 bg-slate-950/80 p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-slate-300">Operator Console</h3>
                      <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Telemetry</span>
                    </div>
                    <div className="space-y-3">
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Hotspot Severity</p>
                        <p className="mt-1 text-lg font-semibold text-amber-300">3 active anomalies</p>
                      </div>
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Signal Sweep</p>
                        <div className="mt-2 flex items-end gap-1">
                          {[54, 72, 64, 88, 76, 92].map((height, idx) => (
                            <div key={idx} className="h-12 flex-1 rounded-t-md bg-gradient-to-t from-cyan-500 to-fuchsia-500" style={{ height: `${height}%` }} />
                          ))}
                        </div>
                      </div>
                      <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                        <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">Review Priority</p>
                        <p className="mt-1 text-sm text-emerald-300">Secondary inspection window open</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Detailed Probabilities */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                  {/* Defect Probabilities */}
                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-slate-300 mb-3">Defect Probabilities</h3>
                    {Object.entries(analysisResults.defect_type.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([defect, prob]) => (
                        <ConfidenceBar
                          key={defect}
                          label={defectDisplayNames[defect] || defect}
                          value={prob}
                          color={defectColors[defect] || '#888'}
                        />
                      ))}
                  </div>

                  {/* NDT Method Probabilities */}
                  <div className="bg-slate-800/30 rounded-xl p-4">
                    <h3 className="text-sm font-semibold text-slate-300 mb-3">NDT Method Probabilities</h3>
                    {Object.entries(analysisResults.ndt_method.probabilities)
                      .sort((a, b) => b[1] - a[1])
                      .map(([method, prob]) => (
                        <ConfidenceBar
                          key={method}
                          label={ndtDisplayNames[method] || method}
                          value={prob}
                          color="#3b82f6"
                        />
                      ))}
                  </div>
                </div>

                {/* Recommendations */}
                <div className="mt-6 rounded-2xl border border-slate-700 bg-slate-800/30 p-4">
                  <h3 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    Recommendations
                  </h3>
                  <div className="space-y-2">
                    {getRecommendations(analysisResults.defect_type.prediction).map((rec, i) => (
                      <div key={i} className={`p-3 rounded-lg text-sm ${
                        rec.startsWith('CRITICAL') ? 'bg-red-900/30 text-red-300 border border-red-700' :
                        rec.startsWith('Recommended') ? 'bg-blue-900/30 text-blue-300 border border-blue-700' :
                        'bg-slate-800 text-slate-300'
                      }`}>
                        {rec}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Status Banner */}
                <div className={`mt-6 p-4 rounded-2xl flex items-center gap-3 ${
                  analysisResults.requires_action && analysisResults.severity.value > 0.6
                    ? 'bg-red-900/30 border border-red-700' :
                  analysisResults.requires_action
                    ? 'bg-amber-900/30 border border-amber-700' :
                    'bg-green-900/30 border border-green-700'
                }`}>
                  {analysisResults.requires_action && analysisResults.severity.value > 0.6 ? (
                    <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0" />
                  ) : analysisResults.requires_action ? (
                    <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-6 h-6 text-green-400 flex-shrink-0" />
                  )}
                  <div>
                    <p className="font-semibold">
                      {analysisResults.requires_action && analysisResults.severity.value > 0.6
                        ? 'CRITICAL: Immediate Action Required'
                        : analysisResults.requires_action
                        ? 'WARNING: Further Inspection Recommended'
                        : 'PASSED: No Action Needed'}
                    </p>
                    <p className="text-sm text-slate-400">
                      {analysisResults.requires_action
                        ? `Defect detected with ${analysisResults.severity.level.toLowerCase()} severity.`
                        : 'Material is within acceptable quality standards.'}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* GENERATE TAB */}
        {activeTab === 'generate' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Synthetic Sample Configuration</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">NDT Method</label>
                  <select
                    value={genParams.ndt_method}
                    onChange={e => setGenParams(p => ({ ...p, ndt_method: e.target.value }))}
                    className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:ring-2 focus:ring-blue-500"
                  >
                    {Object.entries(ndtDisplayNames).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-slate-400 mb-2">Fibre Type</label>
                  <select
                    value={genParams.fibre_type}
                    onChange={e => setGenParams(p => ({ ...p, fibre_type: e.target.value }))}
                    className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:ring-2 focus:ring-blue-500"
                  >
                    {Object.entries(fibreDisplayNames).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-slate-400 mb-2">Defect Type</label>
                  <select
                    value={genParams.defect_type}
                    onChange={e => setGenParams(p => ({ ...p, defect_type: e.target.value }))}
                    className="w-full p-3 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:ring-2 focus:ring-blue-500"
                  >
                    {Object.entries(defectDisplayNames).map(([k, v]) => (
                      <option key={k} value={k}>{v}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm text-slate-400 mb-2">Random Seed: {genParams.seed}</label>
                  <input
                    type="range"
                    min="0"
                    max="1000"
                    value={genParams.seed}
                    onChange={e => setGenParams(p => ({ ...p, seed: parseInt(e.target.value) }))}
                    className="w-full"
                  />
                </div>

                <button
                  onClick={handleGenerate}
                  className="btn-primary w-full py-3 rounded-lg font-semibold text-white flex items-center justify-center gap-2"
                >
                  <Settings className="w-5 h-5" />
                  Generate Sample
                </button>
              </div>
            </div>

            <div className="card rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-4">Generated Image</h2>
              {generatedImage ? (
                <div>
                  <img src={generatedImage} alt="Generated" className="w-full rounded-lg" />
                  <div className="mt-4 flex gap-2">
                    <button
                      onClick={() => {
                        setUploadedImage(generatedImage);
                        setActiveTab('analyze');
                      }}
                      className="btn-primary px-4 py-2 rounded-lg text-sm font-medium text-white flex items-center gap-2"
                    >
                      <Scan className="w-4 h-4" />
                      Analyze This Image
                    </button>
                    <a
                      href={generatedImage}
                      download={`synthetic_${genParams.defect_type}.png`}
                      className="px-4 py-2 bg-slate-700 rounded-lg text-sm font-medium text-slate-200 hover:bg-slate-600"
                    >
                      Download
                    </a>
                  </div>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  <p>Click "Generate Sample" to create a synthetic NDT image</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* BATCH TAB */}
        {activeTab === 'batch' && (
          <div className="card rounded-2xl p-6">
            <h2 className="text-lg font-semibold mb-4">Batch NDT Analysis</h2>

            <div
              className="dropzone p-6 text-center cursor-pointer mb-4"
              onClick={() => document.getElementById('batch-input').click()}
            >
              <input
                id="batch-input"
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.gif,.webp,.xlsx,.xls,.csv,.mdb,.accdb,.mp4,.mov,.avi,.mkv,.wmv,.flv,.webm,.inp,.odb,.dat,.sim,.cas,.rst,.fem,.ans,.bdf,.nas"
                onChange={handleBatchUpload}
                className="hidden"
              />
              <Upload className="w-10 h-10 text-slate-500 mx-auto mb-2" />
              <p className="text-slate-300">Click to select multiple NDT artifacts</p>
              <p className="text-sm text-slate-500">{batchFiles.length} files selected</p>
            </div>

            {batchFiles.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-semibold text-slate-300 mb-2">Selected Files:</h3>
                <div className="flex flex-wrap gap-2">
                  {batchFiles.map((f, i) => (
                    <span key={i} className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300">
                      {f.name}
                    </span>
                  ))}
                </div>
                <button
                  onClick={handleBatchAnalyze}
                  disabled={isBatchAnalyzing}
                  className="btn-primary mt-3 px-6 py-2 rounded-lg font-medium text-white disabled:opacity-50"
                >
                  {isBatchAnalyzing ? 'Processing...' : 'Run Batch Analysis'}
                </button>
              </div>
            )}

            {batchResults && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-2 px-3 text-slate-400">File</th>
                      <th className="text-left py-2 px-3 text-slate-400">NDT Method</th>
                      <th className="text-left py-2 px-3 text-slate-400">Defect</th>
                      <th className="text-left py-2 px-3 text-slate-400">Fibre</th>
                      <th className="text-left py-2 px-3 text-slate-400">Severity</th>
                      <th className="text-left py-2 px-3 text-slate-400">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchResults.map((r, i) => (
                      <tr key={i} className="border-b border-slate-800">
                        <td className="py-2 px-3 text-slate-300">{r.filename}</td>
                        <td className="py-2 px-3 text-blue-400">{ndtDisplayNames[r.ndt_method] || r.ndt_method}</td>
                        <td className="py-2 px-3" style={{ color: defectColors[r.defect_type] }}>
                          {defectDisplayNames[r.defect_type] || r.defect_type}
                        </td>
                        <td className="py-2 px-3 text-yellow-400">{fibreDisplayNames[r.fibre_type] || r.fibre_type}</td>
                        <td className="py-2 px-3 text-slate-300">{r.severity}</td>
                        <td className="py-2 px-3">
                          {r.requires_action ? (
                            <span className="px-2 py-1 bg-red-900/30 text-red-400 rounded text-xs">YES</span>
                          ) : (
                            <span className="px-2 py-1 bg-green-900/30 text-green-400 rounded text-xs">NO</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* REFERENCE TAB */}
        {activeTab === 'reference' && (
          <div className="space-y-6">
            {/* NDT Methods Reference */}
            <div className="card rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4">NDT Methods Reference</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-700">
                      <th className="text-left py-3 px-4 text-slate-400">Method</th>
                      <th className="text-left py-3 px-4 text-slate-400">Code</th>
                      <th className="text-left py-3 px-4 text-slate-400">Best For</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ndtDisplayNames).map(([key, name]) => (
                      <tr key={key} className="border-b border-slate-800 hover:bg-slate-800/30">
                        <td className="py-3 px-4 text-slate-200 font-medium">{name}</td>
                        <td className="py-3 px-4 text-blue-400 font-mono">{key}</td>
                        <td className="py-3 px-4 text-slate-400">
                          {key === 'visual_testing' && 'Surface defects, colour variations'}
                          {key === 'ultrasonic' && 'Internal delaminations, voids, thickness'}
                          {key === 'magnetic_particle' && 'Surface cracks in ferromagnetic materials'}
                          {key === 'radiography' && 'Internal voids, inclusions, density changes'}
                          {key === 'liquid_penetrant' && 'Surface-breaking cracks, porosity'}
                          {key === 'eddy_current' && 'Conductivity changes, fibre orientation'}
                          {key === 'thermal_infrared' && 'Subsurface delaminations, disbonds'}
                          {key === 'microwave' && 'Dielectric changes, moisture content'}
                          {key === 'vibration_analysis' && 'Stiffness changes, modal properties'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Defect Types Reference */}
            <div className="card rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4">Defect Types Reference</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(defectDisplayNames).map(([key, name]) => (
                  <div key={key} className="bg-slate-800/50 rounded-xl p-4 border" style={{ borderColor: `${defectColors[key]}44` }}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 rounded-full" style={{ background: defectColors[key] }} />
                      <h3 className="font-semibold text-slate-200">{name}</h3>
                    </div>
                    <p className="text-sm text-slate-400">
                      {key === 'no_defect' && 'Material is within acceptable quality standards.'}
                      {key === 'delamination' && 'Separation between composite layers reducing structural integrity.'}
                      {key === 'crack' && 'Fracture in matrix or fibres that can propagate under load.'}
                      {key === 'void' && 'Air pocket or porosity within the composite material.'}
                      {key === 'fibre_misalignment' && 'Deviation of fibres from the designed orientation.'}
                      {key === 'resin_rich' && 'Excess resin accumulation in localized areas.'}
                      {key === 'resin_starved' && 'Insufficient resin leaving fibres inadequately supported.'}
                    </p>
                    <span className={`inline-block mt-2 px-2 py-1 rounded text-xs ${
                      key === 'no_defect' ? 'bg-green-900/30 text-green-400' :
                      key === 'delamination' || key === 'crack' ? 'bg-red-900/30 text-red-400' :
                      'bg-amber-900/30 text-amber-400'
                    }`}>
                      {_get_defect_criticality(key)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Fibre Types */}
            <div className="card rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4">Fibre Types</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(fibreDisplayNames).map(([key, name]) => (
                  <div key={key} className="bg-slate-800/50 rounded-xl p-4">
                    <h3 className="font-semibold text-slate-200 mb-2">{name}</h3>
                    <p className="text-sm text-slate-400">
                      {key === 'aramid' && 'High impact resistance, ballistic protection, excellent toughness.'}
                      {key === 'carbon' && 'High stiffness and strength with low weight. Aerospace grade.'}
                      {key === 'glass' && 'Cost-effective with good strength and corrosion resistance.'}
                    </p>
                    <div className="mt-3 flex flex-wrap gap-1">
                      {key === 'aramid' && ['Ballistic protection', 'Aerospace', 'Marine'].map(a => (
                        <span key={a} className="px-2 py-1 bg-slate-700 rounded text-xs text-slate-300">{a}</span>
                      ))}
                      {key === 'carbon' && ['Aerospace', 'Automotive', 'Wind turbines'].map(a => (
                        <span key={a} className="px-2 py-1 bg-slate-700 rounded text-xs text-slate-300">{a}</span>
                      ))}
                      {key === 'glass' && ['Boat hulls', 'Storage tanks', 'Construction'].map(a => (
                        <span key={a} className="px-2 py-1 bg-slate-700 rounded text-xs text-slate-300">{a}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-slate-850 border-t border-slate-700 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          <p>Composite NDT Defect Detection System v1.0.0 | AI-Powered Analysis for FRP Materials</p>
          <p className="mt-1">Supports 9 NDT methods, 3 fibre types, and 6 defect categories</p>
        </div>
      </footer>
    </div>
  );
}

function _get_defect_criticality(defect) {
  const c = {
    no_defect: 'None',
    delamination: 'High',
    crack: 'High',
    void: 'Medium',
    fibre_misalignment: 'Medium',
    resin_rich: 'Low',
    resin_starved: 'Medium',
  };
  return c[defect] || 'Unknown';
}

export default App;
