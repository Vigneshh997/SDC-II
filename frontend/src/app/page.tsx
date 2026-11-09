"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Activity, Download, FileText, Info, Moon, RefreshCw, Sun, UploadCloud } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
type Field = { unit: string; min?: number; max?: number; categories?: string[] };
type Health = { status: string; reason?: string; device?: string };
type Explanations = {
  status?: Record<string, string>;
  heatmap?: string;
  target?: string;
  visual_method?: string;
  visual_limitation?: string;
  limitation?: string;
  text?: { method: string; limitation: string; tokens: { token: string; attribution: number }[] };
  metadata?: { method: string; expected_value: number; unit: string; limitation: string; values: Record<string, number> };
  cross_modal?: { method: string; limitation: string; links: { patch_row: number; patch_column: number; token: string; weight: number }[] };
};
type Result = {
  predictions: Record<string, number>;
  thresholds: Record<string, number>;
  selected_conditions: string[];
  elapsed_seconds: number;
  explanations: Explanations;
  report: { disclaimer: string; limitations: string[] } & Record<string, unknown>;
};

export default function ClinicalWorkstation() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [health, setHealth] = useState<Health>({ status: "checking" });
  const [fields, setFields] = useState<Record<string, Field>>({});
  const [metadata, setMetadata] = useState<Record<string, string>>({});
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [examinationId, setExaminationId] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);

  useEffect(() => { document.documentElement.className = theme; }, [theme]);
  useEffect(() => {
    return () => { if (preview) URL.revokeObjectURL(preview); };
  }, [preview]);

  async function checkBackend() {
    try {
      const [healthResponse, schemaResponse] = await Promise.all([
        fetch(`${API_URL}/health`), fetch(`${API_URL}/schema`),
      ]);
      if (!healthResponse.ok || !schemaResponse.ok) throw new Error("Backend readiness check failed");
      setHealth(await healthResponse.json());
      const schema = await schemaResponse.json();
      setFields(schema.metadata_fields);
    } catch {
      setHealth({ status: "unavailable", reason: `Cannot reach the backend at ${API_URL}.` });
    }
  }
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([fetch(`${API_URL}/health`, { signal: controller.signal }), fetch(`${API_URL}/schema`, { signal: controller.signal })])
      .then(async ([h, s]) => {
        if (!h.ok || !s.ok) throw new Error("Readiness check failed");
        const [state, schema] = await Promise.all([h.json(), s.json()]);
        setHealth(state); setFields(schema.metadata_fields);
      })
      .catch(() => { if (!controller.signal.aborted) setHealth({ status: "unavailable", reason: `Cannot reach the backend at ${API_URL}.` }); });
    return () => controller.abort();
  }, []);

  function chooseFile(selected?: File) {
    if (!selected) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(selected.type)) {
      setError("Select a JPEG, PNG or WebP image."); return;
    }
    if (selected.size > 15 * 1024 * 1024) { setError("The image must be at most 15 MB."); return; }
    setError(null); setFile(selected); setPreview(URL.createObjectURL(selected)); setResult(null);
  }

  async function analyze(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || !note.trim() || !examinationId.trim() || !confirmed) {
      setError("Provide all three observed inputs, an examination ID and the pairing confirmation."); return;
    }
    const values: Record<string, string | number> = {};
    for (const [name, spec] of Object.entries(fields)) {
      const value = metadata[name];
      if (!value?.trim()) { setError(`Observed ${name} is required.`); return; }
      values[name] = spec.categories ? value : Number(value);
    }
    const form = new FormData();
    form.append("image", file);
    form.append("clinical_note", note.trim());
    form.append("metadata", JSON.stringify(values));
    form.append("examination_id", examinationId.trim());
    form.append("paired_inputs_confirmed", String(confirmed));
    setBusy(true); setError(null); setResult(null);
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: form, signal: AbortSignal.timeout(180000) });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Invalid request; check all examination inputs.");
      setResult(data);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Analysis failed. No result was generated.");
    } finally { setBusy(false); }
  }

  function exportReport() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result.report, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = "xfusion-research-report.json"; anchor.click();
    URL.revokeObjectURL(url);
  }

  const panel = "rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900";
  const input = "w-full rounded-md border border-slate-300 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-950";

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4 dark:border-slate-800 dark:bg-slate-900">
        <h1 className="flex items-center gap-2 text-xl font-semibold"><Activity className="text-primary" /> X-Fusion research workstation</h1>
        <button aria-label="Toggle theme" onClick={() => setTheme(theme === "light" ? "dark" : "light")}>{theme === "light" ? <Moon /> : <Sun />}</button>
      </header>
      <main className="mx-auto max-w-6xl space-y-6 p-6">
        <p className="flex items-center gap-2 text-sm"><Info size={18} /> Research prototype only. Not clinically validated or approved for diagnosis or treatment.</p>
        <section className={panel} aria-live="polite">
          <div className="flex items-center justify-between gap-3"><h2 className="font-semibold">Model status: {health.status.replaceAll("_", " ")}</h2>
            <button onClick={() => void checkBackend()} className="flex items-center gap-2 text-sm text-primary"><RefreshCw size={16} /> Refresh</button></div>
          <p className="mt-2 text-sm text-slate-500">{health.reason ?? `Using ${health.device}. All three modalities are required.`}</p>
          {health.status !== "ok" && <p className="mt-2 text-sm">Predictions remain unavailable until a compatible model has been trained on genuine paired data. You can prepare inputs below.</p>}
        </section>
        <form onSubmit={analyze} className="space-y-6">
          <fieldset disabled={busy} className="grid gap-6 lg:grid-cols-2">
            <section className={panel}>
              <h2 className="mb-4 flex items-center gap-2 font-semibold"><UploadCloud size={18} /> 1. Chest X-ray</h2>
              <label className="mb-4 block text-sm">Upload JPEG, PNG or WebP (maximum 15 MB)
                <input required type="file" accept="image/jpeg,image/png,image/webp" onChange={e => chooseFile(e.target.files?.[0])} className="mt-2 block w-full" /></label>
              {preview && file && <div className="relative h-96 rounded-lg bg-black"><Image src={preview} alt="Selected chest X-ray" fill unoptimized className="object-contain" /></div>}
              <label className="mt-4 block text-sm">De-identified examination ID
                <input className={input} value={examinationId} maxLength={128} required onChange={e => setExaminationId(e.target.value)} /></label>
            </section>
            <section className={`${panel} space-y-4`}>
              <h2 className="font-semibold">2. Observed clinical notes</h2>
              <label className="block text-sm">Clinical context available at prediction time
                <textarea className={`${input} min-h-32`} required maxLength={20000} value={note} onChange={e => setNote(e.target.value)} placeholder="Enter the actual examination notes. Preserve negations." /></label>
              <h2 className="font-semibold">3. Observed patient attributes</h2>
              <div className="grid grid-cols-2 gap-4">{Object.entries(fields).map(([name, spec]) => (
                <label className="block text-sm" key={name}>{name.replaceAll("_", " ")} ({spec.unit})
                  {spec.categories ? <select className={input} required value={metadata[name] ?? ""} onChange={e => setMetadata({ ...metadata, [name]: e.target.value })}>
                    <option value="">Select observed category</option>{spec.categories.map(category => <option key={category} value={category}>{category}</option>)}
                  </select> : <input className={input} required type="number" step="any" min={spec.min} max={spec.max} value={metadata[name] ?? ""} onChange={e => setMetadata({ ...metadata, [name]: e.target.value })} />}
                </label>
              ))}</div>
              <label className="flex items-start gap-2 text-sm"><input type="checkbox" required checked={confirmed} onChange={e => setConfirmed(e.target.checked)} /> I confirm that the image, clinical note and attributes are observed inputs from this same examination.</label>
              <p className="text-xs text-slate-500">Inputs and reports are held in session memory. Downloading a report creates a local copy containing the supplied examination data.</p>
            </section>
          </fieldset>
          <button type="submit" disabled={busy || health.status !== "ok" || !file || !Object.keys(fields).length} className="rounded-lg bg-primary px-6 py-3 font-medium text-white disabled:cursor-not-allowed disabled:opacity-50">
            {busy ? "Running tri-modal inference and explanations…" : "Run tri-modal analysis"}
          </button>
          {error && <p role="alert" className="rounded-lg bg-rose-50 p-4 text-rose-700">{error}</p>}
        </form>
        {result && <section className="space-y-6" aria-live="polite">
          <div className={`${panel} flex items-center justify-between`}><div><h2 className="font-semibold">Research predictions</h2><p className="text-sm">{result.selected_conditions.length ? `Above validation thresholds: ${result.selected_conditions.join(", ")}` : "No condition exceeds its validation threshold. This does not establish normality."}</p><p className="text-xs text-slate-500">Elapsed: {result.elapsed_seconds.toFixed(2)} seconds</p></div>
            <button onClick={exportReport} className="flex items-center gap-2 text-primary"><Download size={18} /> Export report</button></div>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className={`${panel} space-y-3`}><h3 className="font-semibold">14 independent disease scores</h3>{Object.entries(result.predictions).sort(([, a], [, b]) => b - a).map(([label, probability]) => (
              <div key={label}><div className="flex justify-between text-sm"><span>{label.replaceAll("_", " ")}</span><span>{(probability * 100).toFixed(1)}% · threshold {(result.thresholds[label] * 100).toFixed(0)}%</span></div><progress value={probability} max={1} className="w-full accent-blue-600" /></div>
            ))}</div>
            <div className={`${panel} space-y-3`}><h3 className="font-semibold">{result.explanations.visual_method ?? "Visual explanation unavailable"}</h3><p className="text-sm">Target: {result.explanations.target ?? "unavailable"}</p>
              {result.explanations.heatmap && <div className="relative h-96 bg-black"><Image src={result.explanations.heatmap} fill unoptimized className="object-contain" alt="Class-specific LayerCAM overlay aligned with the uploaded X-ray" /></div>}
              <p className="text-xs text-slate-500">{result.explanations.visual_limitation ?? result.explanations.limitation}</p></div>
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className={`${panel} space-y-3`}><h3 className="font-semibold">Structured feature attribution</h3>{result.explanations.metadata && <><p className="text-sm">{result.explanations.metadata.method}, baseline probability {result.explanations.metadata.expected_value.toFixed(4)}</p>
              {Object.entries(result.explanations.metadata.values).map(([name, value]) => <div key={name}><p className="flex justify-between text-sm"><span>{name}</span><span>{value.toFixed(5)} probability change</span></p><progress className="w-full" value={Math.abs(value)} max={1} /></div>)}<p className="text-xs text-slate-500">{result.explanations.metadata.limitation}</p></>}</div>
            <div className={`${panel} space-y-3`}><h3 className="font-semibold">Clinical text evidence</h3>{result.explanations.text && <><p className="text-sm">{result.explanations.text.method}</p><div className="flex max-h-64 flex-wrap gap-2 overflow-auto">{result.explanations.text.tokens.map((token, index) => <span key={index} title={`Signed local logit attribution: ${token.attribution.toFixed(5)}`} className={`rounded px-2 py-1 text-xs ${token.attribution >= 0 ? "bg-blue-100 text-blue-900" : "bg-rose-100 text-rose-900"}`}>{token.token} ({token.attribution.toFixed(3)})</span>)}</div><p className="text-xs text-slate-500">{result.explanations.text.limitation}</p></>}</div>
          </div>
          <div className={`${panel} space-y-3`}><h3 className="font-semibold">Cross-modal attention evidence</h3>{result.explanations.cross_modal && <><p className="text-sm">{result.explanations.cross_modal.method}</p><div className="max-h-48 overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>Patch (row, column)</th><th>Text token</th><th>Attention weight</th></tr></thead><tbody>{result.explanations.cross_modal.links.map((link, index) => <tr key={index}><td>{link.patch_row}, {link.patch_column}</td><td>{link.token}</td><td>{link.weight.toFixed(5)}</td></tr>)}</tbody></table></div><p className="text-xs text-slate-500">{result.explanations.cross_modal.limitation}</p></>}
            {Object.entries(result.explanations.status ?? {}).map(([name, status]) => <p className="text-sm" key={name}>{name}: {status}</p>)}</div>
          <div className={`${panel} space-y-2`}><h3 className="flex items-center gap-2 font-semibold"><FileText size={18} /> Report limitations</h3>{result.report.limitations.map(item => <p className="text-sm" key={item}>{item}</p>)}<p className="text-sm font-semibold">{result.report.disclaimer}</p></div>
        </section>}
      </main>
    </div>
  );
}
