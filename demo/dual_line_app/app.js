const $ = (id) => document.getElementById(id);
let selectedFile = null;
let selectedDataUrl = null;

const fmt = (value, digits = 4) => value == null ? "-" : Number(value).toFixed(digits);
const esc = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));

async function checkStatus() {
  try {
    const data = await fetch("/api/status").then(r => r.json());
    $("status").classList.add("ready");
    $("status").querySelector("span").textContent = `${data.backbone} · ${data.device} · label-free`;
  } catch {
    $("status").querySelector("span").textContent = "모델 연결 실패";
  }
}

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    selectedDataUrl = reader.result;
    $("preview").src = selectedDataUrl;
    $("dropzone").classList.add("has-image");
    $("filename").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $("runButton").disabled = false;
  };
  reader.readAsDataURL(file);
}

function fillRows(data) {
  const g = data.gate;
  const rows = [
    ["REVIEW 요청", g.review_requested ? "YES" : "NO"],
    ["SWITCH 승인", g.switched ? "YES" : "NO"],
    ["Risk", `${fmt(g.risk_score)} / threshold ${fmt(g.risk_threshold)}`],
    ["Branch risk", `direct ${fmt(g.direct_risk)} · parent ${fmt(g.parent_risk)} · fine ${fmt(g.fine_risk)}`],
    ["Normality risk", fmt(g.normality_risk)],
    ["Validity", `base ${fmt(g.base_validity)} · candidate ${fmt(g.candidate_validity)}`],
    ["Utility", `${fmt(g.utility)} / threshold ${fmt(g.utility_threshold)}`],
    ["Chosen view", g.chosen_view],
  ];
  $("gateTable").innerHTML = rows.map(([a,b]) => `<tr><th>${esc(a)}</th><td>${esc(b)}</td></tr>`).join("");
  $("candidateTable").innerHTML = data.candidates.length
    ? data.candidates.map(row => `<tr class="${row.selected ? "selected" : ""}"><td>${esc(row.view)}</td><td>${esc(row.parent)}</td><td>${esc(row.fine)}</td><td>${fmt(row.score)}</td><td>${fmt(row.joint_validity)}</td><td>${row.selected ? "선택" : ""}</td></tr>`).join("")
    : `<tr><td colspan="6">REVIEW가 요청되지 않아 후보를 생성하지 않았습니다.</td></tr>`;
  const top = (items) => items.map(x => `<li><b>${esc(x.class)}</b> <span>${(x.probability * 100).toFixed(2)}%</span></li>`).join("");
  $("parentTop3").innerHTML = top(data.prediction.parent_top3);
  $("fineTop3").innerHTML = top(data.prediction.fine_top3);
  $("rawJson").textContent = JSON.stringify(data, null, 2);
}

function showResult(data) {
  const p = data.prediction;
  const g = data.gate;
  $("parentPrediction").textContent = p.parent;
  $("finePrediction").textContent = p.fine;
  // Keep compound decisions readable inside the narrow status column.
  $("decisionBadge").textContent = String(g.decision).replaceAll("_", "\n");
  $("decisionBadge").className = `decision-badge ${g.switched ? "switch" : g.review_requested ? "review" : ""}`;
  $("riskScore").textContent = fmt(g.risk_score);
  $("riskThreshold").textContent = `threshold ${fmt(g.risk_threshold)}`;
  $("baseValidity").textContent = fmt(g.base_validity);
  $("candidateValidity").textContent = fmt(g.candidate_validity);
  $("utility").textContent = fmt(g.utility);
  $("utilityThreshold").textContent = `threshold ${fmt(g.utility_threshold)}`;
  $("baseDecision").textContent = `${p.base_parent} / ${p.base_fine}`;
  $("gateDecision").textContent = g.decision;
  $("chosenView").textContent = g.chosen_view;
  $("overlayImage").src = data.visuals.overlay;
  if (data.visuals.crop) {
    $("cropFigure").hidden = false;
    $("cropImage").src = data.visuals.crop;
  } else {
    $("cropFigure").hidden = true;
  }
  $("runtimeText").textContent = `${data.runtime_ms.toLocaleString()} ms · label-free inference`;
  fillRows(data);
  $("resultEmpty").hidden = true;
  $("resultContent").hidden = false;
}

async function run() {
  if (!selectedFile || !selectedDataUrl) return;
  $("runButton").disabled = true;
  $("resultEmpty").hidden = true;
  $("resultContent").hidden = true;
  $("loading").hidden = false;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({filename: selectedFile.name, image: selectedDataUrl}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "분석에 실패했습니다.");
    showResult(data);
  } catch (error) {
    $("resultEmpty").textContent = error.message;
    $("resultEmpty").hidden = false;
  } finally {
    $("loading").hidden = true;
    $("runButton").disabled = false;
  }
}

$("fileInput").addEventListener("change", e => setFile(e.target.files[0]));
$("runButton").addEventListener("click", run);
for (const name of ["dragenter", "dragover"]) $("dropzone").addEventListener(name, e => { e.preventDefault(); $("dropzone").classList.add("drag"); });
for (const name of ["dragleave", "drop"]) $("dropzone").addEventListener(name, e => { e.preventDefault(); $("dropzone").classList.remove("drag"); });
$("dropzone").addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach(x => x.classList.remove("active"));
  button.classList.add("active");
  $(`${button.dataset.tab}Panel`).classList.add("active");
}));
checkStatus();
