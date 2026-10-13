"use client";

import React, { useState, useRef, useEffect } from "react";
import { UploadCloud, Activity, FileText, CheckCircle, AlertTriangle, Info, Clock, Download, X, Layers, Sun, Moon, Sparkles, RefreshCw, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

type AppState = "dashboard" | "upload" | "analyzing" | "results" | "history";

interface PredictionResult {
  predictions: Record<string, number>;
  metadata_attributions: Record<string, number>;
  heatmap: string; // base64
  diagnosis?: string;
  has_findings?: boolean;
  primary_finding?: string;
  primary_confidence?: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024;

const PRESET_CASES = [
  {
    id: "pneumonia",
    title: "Acute Lobar Pneumonia",
    badge: "High Risk",
    badgeColor: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-200 dark:border-rose-900/50",
    imagePath: "/sample_pneumonia.jpg",
    age: "62",
    sex: "male",
    temperature: "39.4",
    heart_rate: "110",
    resp_rate: "28",
    spo2: "89",
    note: "Patient presents with acute 4-day fever (39.4°C), productive cough with rust-colored sputum, and right-sided pleuritic chest pain. Airspace opacities suspected."
  },
  {
    id: "effusion_pneu",
    title: "Pneumonia + Effusion",
    badge: "Multi-Finding",
    badgeColor: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-900/50",
    imagePath: "/sample_effusion_pneumonia.png",
    age: "60",
    sex: "male",
    temperature: "39.1",
    heart_rate: "105",
    resp_rate: "26",
    spo2: "90",
    note: "Worsening shortness of breath, right lower zone dullness to percussion, high-grade fever, and productive purulent cough."
  },
  {
    id: "consolidation",
    title: "Consolidation & Edema",
    badge: "Cardiopulmonary",
    badgeColor: "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-900/50",
    imagePath: "/sample_consolidation.png",
    age: "55",
    sex: "female",
    temperature: "38.6",
    heart_rate: "98",
    resp_rate: "24",
    spo2: "91",
    note: "Bilateral patchy airspace opacities, progressive dyspnea on exertion, and mild bilateral lower extremity edema."
  },
  {
    id: "normal",
    title: "Normal / Healthy Screening",
    badge: "Low Risk",
    badgeColor: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-900/50",
    imagePath: "/sample_normal.png",
    age: "45",
    sex: "female",
    temperature: "36.8",
    heart_rate: "72",
    resp_rate: "16",
    spo2: "99",
    note: "Routine pre-operative health checkup. Patient is asymptomatic and afebrile with clear bilateral breath sounds and normal chest expansion."
  }
];

export default function ClinicalWorkstation() {
  const [appState, setAppState] = useState<AppState>("dashboard");
  const [theme, setTheme] = useState<"light" | "dark">("light");

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [clinicalNote, setClinicalNote] = useState("");
  const [meta, setMeta] = useState({
    age: "50", sex: "male", temperature: "37.0", heart_rate: "75", resp_rate: "16", spo2: "98"
  });

  const [results, setResults] = useState<PredictionResult | null>(null);
  const [caseHistory, setCaseHistory] = useState<(PredictionResult & { id: string, date: Date })[]>([]);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.5);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.className = theme;
  }, [theme]);

  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setErrorMessage("Please select a PNG, JPEG, or other supported image file.");
      return;
    }
    if (selectedFile.size > MAX_IMAGE_SIZE_BYTES) {
      setErrorMessage("The image must be smaller than 15 MB.");
      return;
    }
    setErrorMessage(null);
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const loadPresetCase = async (preset: typeof PRESET_CASES[0]) => {
    setMeta({
      age: preset.age,
      sex: preset.sex,
      temperature: preset.temperature,
      heart_rate: preset.heart_rate,
      resp_rate: preset.resp_rate,
      spo2: preset.spo2,
    });
    setClinicalNote(preset.note);

    try {
      const response = await fetch(preset.imagePath);
      const blob = await response.blob();
      const filename = preset.imagePath.split("/").pop() || "sample.jpg";
      const sampleFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
      setFile(sampleFile);
      setPreviewUrl(preset.imagePath);
    } catch (err) {
      console.error("Failed to load preset image:", err);
    }
  };

  const submitAnalysis = async () => {
    if (!file) return;
    const invalidFields = Object.entries(meta).filter(([key, value]) => {
      if (key === "sex") return false;
      const number = Number(value);
      if (!Number.isFinite(number)) return true;
      if (key === "age") return number < 0 || number > 120;
      if (key === "temperature") return number < 30 || number > 45;
      if (key === "heart_rate") return number < 20 || number > 250;
      if (key === "resp_rate") return number < 4 || number > 80;
      if (key === "spo2") return number < 0 || number > 100;
      return false;
    });
    if (invalidFields.length > 0) {
      setErrorMessage("Please enter valid patient values before starting the analysis.");
      return;
    }
    setErrorMessage(null);
    setAppState("analyzing");

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("clinical_note", clinicalNote.trim() || "No clinical notes provided. Routine chest radiograph.");
      formData.append("metadata", JSON.stringify({
        age: parseFloat(meta.age) || 50,
        sex: meta.sex || "male",
        temperature: parseFloat(meta.temperature) || 37.0,
        heart_rate: parseFloat(meta.heart_rate) || 75,
        resp_rate: parseFloat(meta.resp_rate) || 16,
        spo2: parseFloat(meta.spo2) || 98,
      }));

      const res = await axios.post(`${API_URL}/predict`, formData);
      const newResult = { ...res.data, id: Math.random().toString(36).substring(2, 9), date: new Date() };
      setResults(newResult);
      setCaseHistory(prev => [newResult, ...prev]);
      setAppState("results");
    } catch (err) {
      console.error(err);
      setErrorMessage(`Analysis failed. Confirm the backend is running at ${API_URL}.`);
      setAppState("upload");
    }
  };

  const resetSession = () => {
    setFile(null);
    setPreviewUrl(null);
    setClinicalNote("");
    setMeta({ age: "50", sex: "male", temperature: "37.0", heart_rate: "75", resp_rate: "16", spo2: "98" });
    setResults(null);
    setErrorMessage(null);
    setAppState("upload");
  };

  return (
    <div className="flex min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300">

      {/* Sidebar */}
      <aside className="w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col hidden md:flex">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 text-primary font-semibold text-lg tracking-tight">
            <Activity className="w-6 h-6 text-primary" />
            <span>X-Fusion AI</span>
          </div>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <SidebarItem icon={<FileText />} label="Dashboard" active={appState === "dashboard"} onClick={() => setAppState("dashboard")} />
          <SidebarItem icon={<UploadCloud />} label="New Analysis" active={appState === "upload" || appState === "analyzing" || appState === "results"} onClick={() => setAppState("upload")} />
          <SidebarItem icon={<Clock />} label="Case History" active={appState === "history"} onClick={() => setAppState("history")} />
        </nav>
        <div className="p-4 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400">
          <p className="flex items-center gap-1 font-medium mb-1"><Info className="w-3 h-3" /> Tri-Modal CADx</p>
          <p className="leading-tight">BiomedCLIP (Vision) + CXR-BERT (Text) + 6-D Vitals MLP with attention maps and SHAP.</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between px-6 shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              {appState === "dashboard" && "Radiology Dashboard"}
              {appState === "upload" && "New Chest X-Ray Analysis"}
              {appState === "analyzing" && "Processing Multimodal Streams..."}
              {appState === "results" && "Diagnostic Findings & Explainability"}
              {appState === "history" && "Session Case History"}
            </h1>
            {appState === "upload" && (
              <span className="hidden sm:inline-flex text-xs bg-primary/10 text-primary px-2.5 py-0.5 rounded-full font-medium">
                14-Disease Multi-Label
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            <button onClick={() => setTheme(theme === "light" ? "dark" : "light")} className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors" title="Toggle Theme">
              {theme === "light" ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>
          </div>
        </header>

        {/* Scrollable Area */}
        <div className="flex-1 overflow-auto p-6 md:p-8">
          <div className="max-w-5xl mx-auto w-full">
            <AnimatePresence mode="wait">

              {/* DASHBOARD STATE */}
              {appState === "dashboard" && (() => {
                const totalScans = caseHistory.length;
                const anomalies = caseHistory.filter(c => Object.values(c.predictions || {}).some(prob => prob > 0.3)).length;
                const positivityRate = totalScans > 0 ? ((anomalies / totalScans) * 100).toFixed(1) + "%" : "0%";

                return (
                  <motion.div key="dashboard" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <StatCard title="Total Scans Processed" value={totalScans.toString()} trend="Active session" />
                      <StatCard title="Pathologies Flagged" value={anomalies.toString()} trend={`${positivityRate} abnormal findings`} alert={anomalies > 0} />
                      <StatCard title="Model Inputs" value="3" trend="Image + text + metadata" />
                    </div>

                    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-8 text-center flex flex-col items-center justify-center py-16 shadow-sm">
                      <div className="w-16 h-16 bg-primary/10 text-primary rounded-full flex items-center justify-center mb-4">
                        <UploadCloud className="w-8 h-8" />
                      </div>
                      <h2 className="text-xl font-semibold mb-2 text-slate-900 dark:text-white">Ready for Patient Case</h2>
                      <p className="text-slate-500 dark:text-slate-400 max-w-md mb-6 text-sm">
                        Upload a frontal chest radiograph along with clinical notes and vital signs to perform tri-modal screening and generate an attention map.
                      </p>
                      <button onClick={() => setAppState("upload")} className="bg-primary hover:bg-primary/90 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm flex items-center gap-2">
                        <Zap className="w-4 h-4" /> Start New Analysis
                      </button>
                    </div>
                  </motion.div>
                );
              })()}

              {/* UPLOAD STATE */}
              {appState === "upload" && (
                <motion.div key="upload" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">

                  {/* Preset Clinical Cases Bar */}
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-primary" />
                        <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">Quick-Load Sample Clinical Cases</span>
                      </div>
                      <span className="text-xs text-slate-400">Click any preset to test instantly</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                      {PRESET_CASES.map((preset) => (
                        <button
                          key={preset.id}
                          onClick={() => loadPresetCase(preset)}
                          className="flex flex-col text-left p-3 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-primary dark:hover:border-primary bg-slate-50/70 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all group"
                        >
                          <div className="flex justify-between items-start mb-1.5">
                            <span className="text-xs font-semibold text-slate-800 dark:text-slate-200 group-hover:text-primary transition-colors">{preset.title}</span>
                            <span className={cn("text-[10px] font-medium px-1.5 py-0.5 rounded border", preset.badgeColor)}>{preset.badge}</span>
                          </div>
                          <span className="text-[11px] text-slate-500 dark:text-slate-400 line-clamp-2 leading-snug">
                            {preset.note}
                          </span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Left Column: Image */}
                    <div className="space-y-4">
                      <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 flex items-center justify-between">
                        <span>1. Chest Radiograph (CXR)</span>
                        {file && (
                          <span className="text-xs text-emerald-600 dark:text-emerald-400 font-normal">
                            Image Selected: {file.name}
                          </span>
                        )}
                      </h3>
                      {!previewUrl ? (
                        <div
                          onDragOver={(e) => e.preventDefault()}
                          onDrop={handleFileDrop}
                          onClick={() => fileInputRef.current?.click()}
                          className="border-2 border-dashed border-slate-300 dark:border-slate-700 hover:border-primary dark:hover:border-primary bg-slate-50 dark:bg-slate-800/50 rounded-xl aspect-[4/3] flex flex-col items-center justify-center cursor-pointer transition-colors"
                        >
                          <UploadCloud className="w-10 h-10 text-slate-400 mb-3" />
                          <p className="font-medium text-slate-700 dark:text-slate-300 mb-1">Drag & drop X-ray image</p>
                          <p className="text-sm text-slate-500">or click to browse files (JPEG, PNG)</p>
                          <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={(e) => e.target.files && handleFileSelection(e.target.files[0])} />
                        </div>
                      ) : (
                        <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black aspect-[4/3] group">
                          <img src={previewUrl} alt="X-ray preview" className="w-full h-full object-contain" />
                          <button onClick={() => { setPreviewUrl(null); setFile(null); }} className="absolute top-3 right-3 p-1.5 bg-black/60 hover:bg-black/90 text-white rounded-md backdrop-blur-sm transition-colors">
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Right Column: Metadata */}
                    <div className="space-y-4">
                      <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">2. Clinical Indication & Acute Vitals</h3>
                      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <Input label="Age (Years)" value={meta.age} onChange={e => setMeta({ ...meta, age: e.target.value })} placeholder="50" />
                          <div>
                            <label className="block text-xs font-medium text-slate-500 mb-1.5">Sex</label>
                            <select className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md px-3 py-2 text-sm outline-none focus:border-primary transition-colors" value={meta.sex} onChange={e => setMeta({ ...meta, sex: e.target.value })}>
                              <option value="male">Male</option>
                              <option value="female">Female</option>
                            </select>
                          </div>
                          <Input label="Temperature (°C)" value={meta.temperature} onChange={e => setMeta({ ...meta, temperature: e.target.value })} placeholder="37.0" />
                          <Input label="Heart Rate (bpm)" value={meta.heart_rate} onChange={e => setMeta({ ...meta, heart_rate: e.target.value })} placeholder="75" />
                          <Input label="Respiration Rate (bpm)" value={meta.resp_rate} onChange={e => setMeta({ ...meta, resp_rate: e.target.value })} placeholder="16" />
                          <Input label="SpO2 (%)" value={meta.spo2} onChange={e => setMeta({ ...meta, spo2: e.target.value })} placeholder="98" />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-slate-500 mb-1.5">Clinical Notes / Radiology Indication</label>
                          <textarea
                            className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md px-3 py-2 text-sm outline-none focus:border-primary transition-colors min-h-[80px] resize-y"
                            placeholder="Enter clinical notes (e.g. Patient presents with cough and fever. Rule out consolidation.)"
                            value={clinicalNote}
                            onChange={e => setClinicalNote(e.target.value)}
                          />
                        </div>
                      </div>

                      <div className="pt-2 flex justify-end">
                        <button
                          onClick={submitAnalysis}
                          disabled={!file}
                          className={cn(
                            "px-6 py-2.5 rounded-lg font-medium transition-all shadow-sm flex items-center gap-2",
                            file ? "bg-primary hover:bg-primary/90 text-white cursor-pointer" : "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
                          )}
                        >
                          <Zap className="w-4 h-4" /> Run Tri-Modal Analysis
                        </button>
                      </div>
                    </div>
                  </div>
                  {errorMessage && (
                    <div role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-900/50 dark:bg-rose-950/30 dark:text-rose-300">
                      {errorMessage}
                    </div>
                  )}
                </motion.div>
              )}

              {/* ANALYZING STATE */}
              {appState === "analyzing" && (
                <motion.div key="analyzing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center justify-center py-32">
                  <div className="relative w-24 h-24 mb-8">
                    <div className="absolute inset-0 rounded-full border-4 border-slate-100 dark:border-slate-800"></div>
                    <div className="absolute inset-0 rounded-full border-4 border-primary border-t-transparent animate-spin"></div>
                    <Activity className="absolute inset-0 m-auto w-8 h-8 text-primary animate-pulse" />
                  </div>
                  <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">Analyzing Tri-Modal Streams...</h2>
                  <p className="text-slate-500 max-w-sm text-center text-sm">
                    Extracting visual features via BiomedCLIP ViT-B/16, tokenizing clinical notes with CXR-BERT, and computing an attention map.
                  </p>
                </motion.div>
              )}

              {/* RESULTS STATE */}
              {appState === "results" && results && (
                <motion.div key="results" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="space-y-6">

                  {/* Results Header */}
                  {(() => {
                    const topConditions = Object.entries(results.predictions || {})
                      .sort(([, a], [, b]) => b - a);
                    const primaryCondition = topConditions[0] || ["Unknown", 0];
                    const hasFindings = results.has_findings !== undefined
                      ? results.has_findings
                      : (primaryCondition[1] >= 0.30);

                    const elevatedConditions = topConditions.filter(([, prob]) => prob >= 0.25);

                    return (
                      <>
                        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                          <div>
                            <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-1">
                              {hasFindings ? (
                                <span className="flex items-center gap-2 text-rose-600 dark:text-rose-500">
                                  <AlertTriangle className="w-6 h-6" /> Findings Detected: {primaryCondition[0].replace('_', ' ')} ({(primaryCondition[1] * 100).toFixed(1)}%)
                                </span>
                              ) : (
                                <span className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                                  <CheckCircle className="w-6 h-6" /> No Acute Pathologies Detected
                                </span>
                              )}
                            </h2>
                            <p className="text-sm text-slate-500 flex items-center gap-2">
                              <span className="font-medium text-slate-700 dark:text-slate-300">Diagnosis:</span>
                              {hasFindings ? primaryCondition[0].replace('_', ' ') : "Normal Baseline"}
                              <span className="text-slate-300 dark:text-slate-700">•</span>
                              <span>Inference Confidence: {(primaryCondition[1] * 100).toFixed(1)}%</span>
                            </p>
                          </div>
                          <div className="flex gap-2">
                            <button onClick={() => window.print()} className="px-4 py-2 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 rounded-lg text-sm font-medium flex items-center gap-2 transition-colors">
                              <Download className="w-4 h-4" /> Export Report
                            </button>
                            <button onClick={resetSession} className="px-4 py-2 bg-primary hover:bg-primary/90 text-white rounded-lg text-sm font-medium transition-colors flex items-center gap-2">
                              <RefreshCw className="w-4 h-4" /> New Patient
                            </button>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                          {/* Visual Evidence (XAI) */}
                          <div className="lg:col-span-7 space-y-6">
                            <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden flex flex-col shadow-sm">
                              <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-slate-50 dark:bg-slate-900/50">
                                <h3 className="font-medium flex items-center gap-2 text-slate-800 dark:text-slate-200">
                                  <Layers className="w-4 h-4 text-primary" /> AI Visual Attention Map
                                </h3>
                                <div className="flex items-center gap-3">
                                  <span className="text-xs font-medium text-slate-500">Overlay</span>
                                  <input type="range" min="0" max="1" step="0.05" value={heatmapOpacity} onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))} className="w-24 accent-primary" />
                                </div>
                              </div>
                              <div className="relative flex-1 bg-black aspect-[4/3] flex items-center justify-center overflow-hidden">
                                {previewUrl && <img src={previewUrl} className="absolute inset-0 w-full h-full object-contain" alt="Original X-Ray" />}
                                {results.heatmap && (
                                  <img
                                    src={results.heatmap.startsWith('data:') ? results.heatmap : `data:image/jpeg;base64,${results.heatmap}`}
                                    style={{ opacity: heatmapOpacity }}
                                    className="absolute inset-0 w-full h-full object-contain mix-blend-screen transition-opacity duration-200"
                                    alt="AI Heatmap"
                                  />
                                )}
                              </div>
                              <div className="p-3 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-100 dark:border-slate-800 text-xs text-slate-500 flex justify-between">
                                <span>Red/Warm regions indicate high Transformer attention</span>
                                <span>Attention visualization; not a diagnostic certainty</span>
                              </div>
                            </div>

                            {/* Metadata SHAP Attributions */}
                            {results.metadata_attributions && Object.keys(results.metadata_attributions).length > 0 && (
                              <div className="bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm">
                                <h3 className="font-medium text-slate-800 dark:text-slate-200 border-b border-slate-100 dark:border-slate-800 pb-3 mb-4 flex items-center justify-between">
                                  <span>Clinical Vital Attributions (Kernel SHAP)</span>
                                  <span className="text-xs text-slate-400 font-normal">Relative importance</span>
                                </h3>
                                <div className="space-y-3">
                                  {Object.entries(results.metadata_attributions)
                                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                                    .map(([feature, importance]) => (
                                      <div key={feature} className="space-y-1">
                                        <div className="flex justify-between text-xs">
                                          <span className="font-medium text-slate-700 dark:text-slate-300 capitalize">{feature.replace('_', ' ')}</span>
                                          <span className={cn("font-semibold", importance > 0 ? "text-emerald-600 dark:text-emerald-400" : "text-slate-400")}>
                                            {(Math.abs(importance) * 100).toFixed(1)}% importance
                                          </span>
                                        </div>
                                        <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                          <div className="h-full bg-primary rounded-full transition-all duration-700" style={{ width: `${Math.min(100, Math.abs(importance) * 100)}%` }}></div>
                                        </div>
                                      </div>
                                    ))}
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Detailed Findings */}
                          <div className="lg:col-span-5 space-y-6">
                            <div className="bg-white dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-800 space-y-5 shadow-sm">
                              <h3 className="font-medium text-slate-800 dark:text-slate-200 border-b border-slate-100 dark:border-slate-800 pb-3 flex justify-between items-center">
                                <span>Diagnostic Breakdown (14 Pathologies)</span>
                                <span className="text-xs text-slate-400 font-normal">NIH ChestX-ray14</span>
                              </h3>

                              <div className="space-y-3.5 max-h-[380px] overflow-y-auto pr-2">
                                {topConditions.map(([condition, prob]) => (
                                  <div key={condition} className="space-y-1">
                                    <div className="flex justify-between items-center text-xs">
                                      <span className={cn("font-medium", prob >= 0.25 ? "text-slate-900 dark:text-white font-semibold" : "text-slate-600 dark:text-slate-400")}>
                                        {condition.replace('_', ' ')}
                                      </span>
                                      <span className={cn("font-bold text-xs", prob >= 0.5 ? "text-rose-600 dark:text-rose-400" : prob >= 0.25 ? "text-amber-600 dark:text-amber-400" : "text-slate-400")}>
                                        {(prob * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    <div className="w-full h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                      <div
                                        className={cn("h-full rounded-full transition-all duration-1000", prob >= 0.5 ? "bg-rose-500" : prob >= 0.25 ? "bg-amber-500" : "bg-slate-300 dark:bg-slate-700")}
                                        style={{ width: `${Math.max(2, prob * 100)}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>

                            {/* Automated Clinical Summary */}
                            <div className="bg-slate-50 dark:bg-slate-800/50 p-5 rounded-xl border border-slate-200 dark:border-slate-800 space-y-2.5">
                              <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                                <FileText className="w-4 h-4 text-primary" /> Automated Diagnostic Summary
                              </h4>
                              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                                {hasFindings ? (
                                  <>
                                    The X-Fusion tri-modal network ranked <strong className="text-rose-600 dark:text-rose-400">{primaryCondition[0].replace('_', ' ')} ({(primaryCondition[1] * 100).toFixed(1)}%)</strong> as the highest model score
                                    {elevatedConditions.length > 1 && (
                                      <> alongside secondary indicators of {elevatedConditions.slice(1).map(([c, p]) => `${c.replace('_', ' ')} (${(p * 100).toFixed(1)}%)`).join(', ')}</>
                                    )}. Review the attention visualization and metadata importance alongside the original case information.
                                  </>
                                ) : (
                                  <>
                                    Analysis completed using X-Fusion tri-modal fusion. No finding exceeded the configured screening threshold; this result does not rule out disease and requires clinical review.
                                  </>
                                )}
                              </p>
                              <div className="pt-2 border-t border-slate-200 dark:border-slate-700 text-[11px] text-slate-500">
                                <strong>Next step:</strong> Clinical correlation and professional review are recommended for every result.
                              </div>
                            </div>
                          </div>
                        </div>
                      </>
                    );
                  })()}
                </motion.div>
              )}

              {/* HISTORY STATE */}
              {appState === "history" && (
                <motion.div key="history" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-6">
                  {caseHistory.length === 0 ? (
                    <div className="bg-white dark:bg-slate-900 p-8 rounded-xl border border-slate-200 dark:border-slate-800 text-center flex flex-col items-center justify-center shadow-sm">
                      <Clock className="w-12 h-12 text-slate-300 dark:text-slate-700 mb-4" />
                      <h2 className="text-xl font-medium mb-2 text-slate-900 dark:text-white">Case History</h2>
                      <p className="text-slate-500 dark:text-slate-400 max-w-sm mb-6 text-sm">No previous cases analyzed in this session yet.</p>
                      <button onClick={() => setAppState("upload")} className="bg-primary hover:bg-primary/90 text-white px-6 py-2.5 rounded-lg font-medium transition-colors shadow-sm">
                        Start New Analysis
                      </button>
                    </div>
                  ) : (
                    <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden shadow-sm">
                      <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
                        <h3 className="font-medium text-slate-800 dark:text-slate-200 flex items-center gap-2"><Clock className="w-4 h-4" /> Session Case History ({caseHistory.length})</h3>
                      </div>
                      <div className="divide-y divide-slate-100 dark:divide-slate-800">
                        {caseHistory.map((item, i) => {
                          const topCondition = Object.entries(item.predictions || {}).sort(([, a], [, b]) => b - a)[0];
                          const hasFindings = topCondition && topCondition[1] >= 0.30;
                          return (
                            <div key={item.id} className="p-4 flex items-center justify-between hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                              <div className="flex items-center gap-4">
                                <div className="w-10 h-10 rounded-md bg-slate-100 dark:bg-slate-800 flex items-center justify-center font-medium text-slate-500 text-sm">
                                  #{caseHistory.length - i}
                                </div>
                                <div>
                                  <div className="font-medium text-slate-800 dark:text-slate-200 text-sm">
                                    {hasFindings ? (
                                      <span className="text-rose-500 font-semibold">Finding: {topCondition[0].replace('_', ' ')} ({(topCondition[1] * 100).toFixed(1)}%)</span>
                                    ) : (
                                      <span className="text-emerald-500 font-semibold">Normal Baseline</span>
                                    )}
                                  </div>
                                  <div className="text-xs text-slate-400">
                                    {item.date.toLocaleTimeString()} - Analyzed via X-Fusion
                                  </div>
                                </div>
                              </div>
                              <button
                                onClick={() => { setResults(item); setAppState("results"); }}
                                className="px-3 py-1.5 text-xs bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-md transition-colors font-medium text-slate-700 dark:text-slate-300"
                              >
                                View Results
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </div>
      </main>
    </div>
  );
}

// ----- Helper Components -----

function SidebarItem({ icon, label, active, onClick }: { icon: React.ReactNode; label: string; active?: boolean; onClick?: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-colors",
        active
          ? "bg-primary/10 text-primary font-semibold"
          : "text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100"
      )}
    >
      <div className="[&>svg]:w-5 [&>svg]:h-5">{icon}</div>
      {label}
    </button>
  );
}

function StatCard({ title, value, trend, alert }: { title: string; value: string; trend: string; alert?: boolean }) {
  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-sm">
      <h3 className="text-xs font-medium text-slate-500 mb-1.5 uppercase tracking-wide">{title}</h3>
      <div className="text-3xl font-bold text-slate-900 dark:text-white mb-1">{value}</div>
      <p className={cn("text-xs", alert ? "text-rose-500 font-medium" : "text-emerald-500 font-medium")}>{trend}</p>
    </div>
  );
}

function Input({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (e: React.ChangeEvent<HTMLInputElement>) => void; placeholder?: string }) {
  return (
    <div>
      <label className="block text-xs font-medium text-slate-500 mb-1.5">{label}</label>
      <input
        type="text"
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="w-full bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-md px-3 py-2 text-sm outline-none focus:border-primary transition-colors"
      />
    </div>
  );
}
