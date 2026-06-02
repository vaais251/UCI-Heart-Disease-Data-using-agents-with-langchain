/* CardioTriage AI — Machine Learning Phase deck (light theme). */
const pptxgen = require("pptxgenjs");
const path = require("path");

const IMG = (f) => path.join(__dirname, "images", f);

// ---- Palette (light, cardiac) --------------------------------------------
const WHITE = "FFFFFF";
const SOFT = "EEF3F8";   // soft slate for title/closing backgrounds
const INK = "0F2C3F";    // deep slate (titles)
const BODY = "33475B";   // body text
const MUTED = "70819A";  // captions
const RED = "C0392B";    // cardiac accent
const TEAL = "1C7293";   // data secondary
const GREEN = "2E7D5B";  // positive
const CARD = "F4F7FA";   // card fill
const LINE = "D9E2EC";   // borders

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "CardioTriage AI";
pres.title = "CardioTriage AI — Machine Learning Phase";

const PW = 13.33, PH = 7.5, M = 0.5, CW = PW - 2 * M;
const mkShadow = () => ({ type: "outer", color: "8AA0B5", blur: 8, offset: 3, angle: 90, opacity: 0.25 });

// ---- Helpers --------------------------------------------------------------
function header(slide, code, title, sub) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 0.42, w: 1.55, h: 0.42, fill: { color: RED }, rectRadius: 0.08, line: { type: "none" } });
  slide.addText(code, { x: M, y: 0.42, w: 1.55, h: 0.42, align: "center", valign: "middle", color: WHITE, bold: true, fontSize: 13, fontFace: "Consolas", margin: 0 });
  slide.addText(title, { x: M, y: 0.95, w: CW, h: 0.62, color: INK, bold: true, fontSize: 28, fontFace: "Georgia", margin: 0 });
  if (sub) slide.addText(sub, { x: M, y: 1.55, w: CW, h: 0.4, color: MUTED, italic: true, fontSize: 14, margin: 0 });
}

function footer(slide, n) {
  slide.addText("CardioTriage AI  ·  Machine Learning Phase", { x: M, y: PH - 0.42, w: 8, h: 0.3, color: MUTED, fontSize: 9, margin: 0 });
  slide.addText(String(n), { x: PW - 1.0, y: PH - 0.42, w: 0.5, h: 0.3, color: MUTED, fontSize: 9, align: "right", margin: 0 });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || CARD }, line: { color: LINE, width: 1 }, rectRadius: 0.08, shadow: mkShadow() });
}

function stat(slide, x, y, w, h, num, label, color, fs) {
  card(slide, x, y, w, h, WHITE);
  slide.addText(num, { x: x + 0.1, y: y + 0.12, w: w - 0.2, h: h * 0.55, align: "center", valign: "middle", color: color, bold: true, fontSize: fs || 30, fontFace: "Georgia", margin: 0 });
  slide.addText(label, { x: x + 0.1, y: y + h * 0.6, w: w - 0.2, h: h * 0.34, align: "center", valign: "top", color: BODY, fontSize: 11, margin: 0 });
}

// place image inside a box preserving aspect ratio (ratio = w/h)
function img(slide, file, ratio, bx, by, bw, bh) {
  let w = bw, h = w / ratio;
  if (h > bh) { h = bh; w = h * ratio; }
  slide.addImage({ path: IMG(file), x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h });
}

function bullets(slide, items, x, y, w, h, fs) {
  slide.addText(items.map((t, i) => ({ text: t, options: { bullet: { code: "2022", indent: 14 }, breakLine: true, paraSpaceAfter: 6, color: BODY, fontSize: fs || 14 } })), { x, y, w, h, valign: "top", margin: 0 });
}

const TH = (t) => ({ text: t, options: { fill: { color: INK }, color: WHITE, bold: true, fontSize: 11, align: "center", valign: "middle" } });
const TD = (t, opts) => ({ text: t, options: Object.assign({ fontSize: 11, color: BODY, align: "center", valign: "middle" }, opts || {}) });

// =========================================================================
// SLIDE 1 — Title
// =========================================================================
let s = pres.addSlide();
s.background = { color: SOFT };
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.35, w: 0.22, h: 2.0, fill: { color: RED }, rectRadius: 0.05, line: { type: "none" } });
s.addText("CardioTriage AI", { x: 0.95, y: 1.4, w: 11.5, h: 1.1, color: INK, bold: true, fontSize: 56, fontFace: "Georgia", margin: 0 });
s.addText("Machine Learning Phase — Detailed Walkthrough", { x: 0.97, y: 2.55, w: 11.5, h: 0.6, color: TEAL, bold: true, fontSize: 24, margin: 0 });
s.addText("From raw cardiac records to a deployed risk-and-phenotype engine (Steps A1–A8)", { x: 0.97, y: 3.15, w: 11.5, h: 0.5, color: MUTED, italic: true, fontSize: 15, margin: 0 });
const chips = [["920", "patients"], ["3", "models compared"], ["0.91", "ROC-AUC"], ["0.96", "recall (tuned)"]];
chips.forEach((c, i) => stat(s, 0.95 + i * 2.65, 4.35, 2.4, 1.35, c[0], c[1], i % 2 ? TEAL : RED));
s.addText("BS Artificial Intelligence  ·  Combined Assessment (Intro to AI + Machine Learning)", { x: 0.95, y: 6.6, w: 11.5, h: 0.4, color: MUTED, fontSize: 12, margin: 0 });

// =========================================================================
// SLIDE 2 — Where ML fits
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "CONTEXT", "Where the Machine Learning Phase Fits");
const layers = [
  ["1 · Data & ML", "ML risk engine — predicts per-patient risk probability + phenotype", true],
  ["2 · Classical AI core", "A* search · CSP allocator · forward-chaining knowledge base", false],
  ["3 · Orchestration", "Gemini + LangChain agent — decides which tool to call & explains", false],
  ["4 · Presentation", "Interactive web triage board (ranking, paths, allocations)", false],
];
let ly = 1.95;
layers.forEach(([t, d, hl]) => {
  card(s, M, ly, 8.4, 1.05, hl ? "FCEDEB" : WHITE);
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: ly, w: 0.12, h: 1.05, fill: { color: hl ? RED : LINE }, line: { type: "none" } });
  s.addText(t, { x: M + 0.3, y: ly + 0.12, w: 3.0, h: 0.8, color: hl ? RED : INK, bold: true, fontSize: 15, valign: "middle", margin: 0 });
  s.addText(d, { x: M + 3.2, y: ly + 0.12, w: 5.0, h: 0.8, color: BODY, fontSize: 12.5, valign: "middle", margin: 0 });
  ly += 1.2;
});
card(s, 9.2, 1.95, 3.6, 4.5, INK);
s.addText("This deck covers Layer 1", { x: 9.4, y: 2.2, w: 3.2, h: 0.8, color: WHITE, bold: true, fontSize: 18, fontFace: "Georgia", margin: 0 });
s.addText([
  { text: "Phase A builds the ML risk engine — Steps A1–A8.", options: { breakLine: true, paraSpaceAfter: 10, fontSize: 13, color: "CADCFC" } },
  { text: "Its outputs (risk probability + phenotype) become the inputs every layer above consumes.", options: { breakLine: true, paraSpaceAfter: 10, fontSize: 13, color: "CADCFC" } },
  { text: "The LLM never makes a medical call alone — it coordinates deterministic tools.", options: { fontSize: 13, color: "CADCFC" } },
], { x: 9.4, y: 3.1, w: 3.2, h: 3.1, valign: "top", margin: 0 });
footer(s, 2);

// =========================================================================
// SLIDE 3 — Dataset
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "DATA", "The Dataset — UCI Heart Disease (combined)");
bullets(s, [
  "920 patients × 16 clinical columns.",
  "Pooled from four hospitals (multi-site).",
  "Target 'num' (0–4) → binarized: 0 = no disease, 1–4 = disease.",
  "Contains real missing values — handled honestly, not dropped.",
], M, 2.0, 6.6, 2.2, 15);
const hosp = [["Cleveland", "304"], ["Hungary", "293"], ["VA Long Beach", "200"], ["Switzerland", "123"]];
hosp.forEach((hh, i) => {
  const x = M + (i % 2) * 3.25, y = 4.35 + Math.floor(i / 2) * 1.1;
  stat(s, x, y, 3.0, 0.95, hh[1], hh[0] + " records", TEAL, 24);
});
card(s, 7.6, 1.95, 5.2, 4.6, WHITE);
s.addText("Outcome balance", { x: 7.8, y: 2.1, w: 4.8, h: 0.4, color: INK, bold: true, fontSize: 14, margin: 0 });
s.addChart(pres.charts.DOUGHNUT, [{ name: "Outcome", labels: ["No disease (411)", "Disease (509)"], values: [411, 509] }], {
  x: 7.7, y: 2.5, w: 5.0, h: 3.9, chartColors: ["94A3B8", RED], holeSize: 58, showLegend: true, legendPos: "b", legendColor: BODY, legendFontSize: 11, showValue: false, showPercent: true, dataLabelColor: WHITE, dataLabelFontSize: 12,
});
footer(s, 3);

// =========================================================================
// SLIDE 4 — Roadmap
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "ROADMAP", "Eight Steps, One Pipeline");
const steps = [
  ["A1", "Inspect", "shape, types, missingness"],
  ["A2", "Preprocess", "clean · encode · scale · split"],
  ["A3", "Explore", "EDA: what separates sick"],
  ["A4", "Baseline", "Logistic Regression"],
  ["A5", "Compare", "RF & Gradient Boosting"],
  ["A6", "Tune", "bias-variance · reg · threshold"],
  ["A7", "Cluster", "K-means risk phenotypes"],
  ["A8", "Save", "persist + bridge to AI"],
];
steps.forEach((st, i) => {
  const x = M + (i % 4) * 3.13, y = 2.2 + Math.floor(i / 4) * 2.15;
  card(s, x, y, 2.9, 1.9, WHITE);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.7, h: 0.7, fill: { color: i < 4 ? TEAL : RED }, line: { type: "none" } });
  s.addText(st[0], { x: x + 0.25, y: y + 0.25, w: 0.7, h: 0.7, align: "center", valign: "middle", color: WHITE, bold: true, fontSize: 15, fontFace: "Consolas", margin: 0 });
  s.addText(st[1], { x: x + 1.1, y: y + 0.28, w: 1.7, h: 0.6, color: INK, bold: true, fontSize: 16, valign: "middle", margin: 0 });
  s.addText(st[2], { x: x + 0.25, y: y + 1.05, w: 2.45, h: 0.7, color: BODY, fontSize: 11.5, valign: "top", margin: 0 });
});
footer(s, 4);

// =========================================================================
// SLIDE 5 — A1 Inspect
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A1", "Load & Inspect the Data");
bullets(s, [
  "Read the CSV into a pandas DataFrame — change nothing yet.",
  "Confirmed 920 rows × 16 columns; identified each column's meaning & type.",
  "Target distribution: 411 healthy vs 509 diseased (after binarizing).",
  "Surfaced the missing-value map (right) — the central challenge.",
], M, 2.0, 6.2, 2.1, 14);
card(s, M, 4.3, 6.2, 2.5, "FCEDEB");
s.addText("Key finding", { x: M + 0.25, y: 4.45, w: 5.7, h: 0.4, color: RED, bold: true, fontSize: 14, margin: 0 });
s.addText("ca (66%) and thal (53%) are missing for over half of patients — too valuable to drop, too sparse to ignore. This drove the A2 strategy.", { x: M + 0.25, y: 4.85, w: 5.7, h: 1.8, color: BODY, fontSize: 13, valign: "top", margin: 0 });
card(s, 7.1, 1.95, 5.7, 4.85, WHITE);
s.addText("Missing values by column (%)", { x: 7.3, y: 2.1, w: 5.3, h: 0.4, color: INK, bold: true, fontSize: 13, margin: 0 });
s.addChart(pres.charts.BAR, [{ name: "% missing", labels: ["ca", "thal", "slope", "fbs", "oldpeak", "trestbps", "exang", "thalch", "chol"], values: [66.4, 52.8, 33.6, 9.8, 6.7, 6.4, 6.0, 6.0, 3.3] }], {
  x: 7.15, y: 2.5, w: 5.55, h: 4.2, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: BODY, dataLabelFontSize: 9, dataLabelFormatCode: "0.0", catAxisLabelColor: BODY, catAxisLabelFontSize: 10, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false,
});
footer(s, 5);

// =========================================================================
// SLIDE 6 — A2 Preprocess
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A2", "Preprocessing — Leakage-Safe Pipeline");
bullets(s, [
  "Binarize target: num 1–4 → 1 (disease present).",
  "Drop id (row number) and dataset (hospital — a confound).",
  "Fix hidden missingness: chol = 0 is impossible → 172 cases set to missing.",
  "Add ca_missing / thal_missing flags so the model knows a value was estimated.",
  "Median/mode impute · one-hot encode · standardize — all inside one Pipeline.",
  "Fit on TRAIN only, apply to TEST → no information leaks across the split.",
], M, 2.0, 7.2, 3.2, 14);
const a2 = [["736 / 184", "stratified train / test", 28], ["0.553 / 0.554", "disease rate · train vs test", 18], ["15 → 27", "features after encoding", 28], ["0", "missing values remaining", 28]];
a2.forEach((c, i) => {
  const x = 8.05 + (i % 2) * 2.45, y = 2.05 + Math.floor(i / 2) * 1.55;
  stat(s, x, y, 2.25, 1.35, c[0], c[1], i % 2 ? TEAL : RED, c[2]);
});
card(s, 8.05, 5.25, 4.75, 1.45, INK);
s.addText("Why leakage matters", { x: 8.25, y: 5.4, w: 4.4, h: 0.4, color: WHITE, bold: true, fontSize: 13, margin: 0 });
s.addText("If fill-values or scales were learned from the whole dataset, the test score would be inflated and dishonest.", { x: 8.25, y: 5.8, w: 4.4, h: 0.85, color: "CADCFC", fontSize: 12, valign: "top", margin: 0 });
footer(s, 6);

// =========================================================================
// SLIDE 7 — A3 EDA
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A3", "Exploratory Analysis — What Separates Sick from Healthy");
s.addText("Mean value by outcome (training set)", { x: M, y: 1.95, w: 6, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
s.addTable([
  [TH("Feature"), TH("Disease"), TH("Healthy"), TH("Signal")],
  [TD("thalch (max HR)", { align: "left" }), TD("128.2"), TD("149.1"), TD("strong", { color: RED, bold: true })],
  [TD("oldpeak", { align: "left" }), TD("1.3"), TD("0.4"), TD("strong", { color: RED, bold: true })],
  [TD("ca (vessels)", { align: "left" }), TD("1.1"), TD("0.2"), TD("strong", { color: RED, bold: true })],
  [TD("age", { align: "left" }), TD("56.2"), TD("50.4"), TD("moderate", { color: TEAL })],
  [TD("chol", { align: "left" }), TD("254"), TD("240"), TD("weak", { color: MUTED })],
  [TD("trestbps", { align: "left" }), TD("134"), TD("130"), TD("weak", { color: MUTED })],
], { x: M, y: 2.35, w: 6.1, colW: [2.5, 1.2, 1.2, 1.2], rowH: 0.4, border: { pt: 0.5, color: LINE }, fill: { color: WHITE }, valign: "middle" });
card(s, M, 5.55, 6.1, 1.25, "FCEDEB");
s.addText("Silent danger", { x: M + 0.2, y: 5.66, w: 5.7, h: 0.35, color: RED, bold: true, fontSize: 13, margin: 0 });
s.addText("Asymptomatic chest pain shows the HIGHEST disease rate (78%) — the 'silent' group screening can miss.", { x: M + 0.2, y: 6.0, w: 5.7, h: 0.75, color: BODY, fontSize: 12.5, valign: "top", margin: 0 });
card(s, 6.95, 1.95, 5.85, 4.9, WHITE);
img(s, "a3_categorical_rates.png", 1.444, 7.05, 2.05, 5.65, 4.7);
footer(s, 7);

// =========================================================================
// SLIDE 8 — A3 correlation
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A3", "Feature Correlation with Disease");
card(s, M, 1.95, 6.6, 4.9, WHITE);
img(s, "a3_correlation_heatmap.png", 1.333, M + 0.1, 2.05, 6.4, 4.7);
s.addText("Strength of link with disease", { x: 7.4, y: 2.0, w: 5.4, h: 0.4, color: INK, bold: true, fontSize: 14, margin: 0 });
s.addChart(pres.charts.BAR, [{ name: "corr", labels: ["ca", "thalch", "oldpeak", "age", "chol", "trestbps"], values: [0.48, 0.40, 0.38, 0.31, 0.12, 0.12] }], {
  x: 7.3, y: 2.4, w: 5.5, h: 2.8, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: BODY, dataLabelFontSize: 10, dataLabelFormatCode: "0.00", catAxisLabelColor: BODY, catAxisLabelFontSize: 11, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false,
});
card(s, 7.3, 5.4, 5.5, 1.45, CARD);
bullets(s, [
  "ca, thalch, oldpeak, age are the strongest signals.",
  "chol & BP correlate weakly — famous ≠ predictive.",
  "No pair exceeds 0.38 → low multicollinearity (good for the baseline).",
], 7.5, 5.5, 5.1, 1.25, 11.5);
footer(s, 8);

// =========================================================================
// SLIDE 9 — A4 Baseline
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A4", "Baseline Model — Logistic Regression");
const a4 = [["0.799", "Accuracy", TEAL], ["0.804", "Precision", TEAL], ["0.843", "Recall", RED], ["0.744", "Specificity", TEAL], ["0.823", "F1", TEAL], ["0.897", "ROC-AUC", RED]];
a4.forEach((c, i) => {
  const x = M + (i % 2) * 2.45, y = 2.0 + Math.floor(i / 2) * 1.35;
  stat(s, x, y, 2.25, 1.2, c[0], c[1], c[2]);
});
card(s, 5.55, 5.4, 7.25, 1.45, INK);
s.addText("Read it clinically", { x: 5.75, y: 5.52, w: 6.8, h: 0.35, color: WHITE, bold: true, fontSize: 13, margin: 0 });
s.addText("Caught 86 of 102 sick patients; 16 missed cases (false negatives) are the target for tuning. Train 0.836 vs test 0.799 → small gap, a healthy baseline.", { x: 5.75, y: 5.86, w: 6.8, h: 0.9, color: "CADCFC", fontSize: 12.5, valign: "top", margin: 0 });
card(s, 5.55, 1.95, 7.25, 3.3, WHITE);
img(s, "a4_confusion_matrix.png", 1.25, 5.65, 2.05, 7.05, 3.1);
footer(s, 9);

// =========================================================================
// SLIDE 10 — A5 Compare
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A5", "Three Models Compared");
s.addTable([
  [TH("Model"), TH("Acc"), TH("Recall"), TH("Spec"), TH("F1"), TH("AUC"), TH("Train")],
  [TD("Logistic Regression", { align: "left" }), TD("0.799"), TD("0.843"), TD("0.744"), TD("0.823"), TD("0.897"), TD("0.836")],
  [TD("Random Forest", { align: "left" }), TD("0.810"), TD("0.863"), TD("0.744"), TD("0.834"), TD("0.907"), TD("1.000", { color: RED, bold: true })],
  [{ text: "Gradient Boosting  ✓", options: { align: "left", bold: true, color: INK, fontSize: 11, fill: { color: "E7F0E9" } } }, TD("0.821", { fill: { color: "E7F0E9" } }), TD("0.892", { bold: true, color: GREEN, fill: { color: "E7F0E9" } }), TD("0.732", { fill: { color: "E7F0E9" } }), TD("0.847", { bold: true, color: GREEN, fill: { color: "E7F0E9" } }), TD("0.905", { fill: { color: "E7F0E9" } }), TD("0.916", { fill: { color: "E7F0E9" } })],
], { x: M, y: 2.0, w: 7.0, colW: [2.5, 0.75, 0.85, 0.75, 0.75, 0.75, 0.85], rowH: 0.5, border: { pt: 0.5, color: LINE }, valign: "middle" });
card(s, M, 4.5, 7.0, 2.3, CARD);
bullets(s, [
  "Both ensembles beat the baseline — real non-linear interactions exist.",
  "Random Forest hit a perfect 1.000 on training → clear overfitting.",
  "Winner: Gradient Boosting — best recall (0.892), F1 and accuracy; AUC tied for top.",
], M + 0.2, 4.62, 6.6, 2.1, 13);
card(s, 7.75, 1.95, 5.05, 4.85, WHITE);
s.addText("ROC curves", { x: 7.95, y: 2.05, w: 4.6, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
img(s, "a5_roc_comparison.png", 1.2, 7.85, 2.45, 4.85, 4.3);
footer(s, 10);

// =========================================================================
// SLIDE 11 — A6a learning curves
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A6a", "Bias–Variance Diagnosis (Learning Curves)");
card(s, M, 1.95, CW, 3.15, WHITE);
img(s, "a6a_learning_curves.png", 3.2, M + 0.1, 2.05, CW - 0.2, 2.95);
const lc = [["Logistic Regression", "gap 0.021", "Mild bias, low variance — curves meet; more data won't help.", TEAL], ["Random Forest", "gap 0.181", "Severe overfitting — train pinned at 1.000 while CV lags.", RED], ["Gradient Boosting", "gap 0.127", "Moderate, still converging — regularization should help.", "B8860B"]];
lc.forEach((c, i) => {
  const x = M + i * 4.18;
  card(s, x, 5.3, 3.95, 1.55, CARD);
  s.addText(c[0], { x: x + 0.2, y: 5.4, w: 3.6, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
  s.addText(c[1], { x: x + 0.2, y: 5.72, w: 3.6, h: 0.3, color: c[3], bold: true, fontSize: 12, fontFace: "Consolas", margin: 0 });
  s.addText(c[2], { x: x + 0.2, y: 6.04, w: 3.6, h: 0.75, color: BODY, fontSize: 11.5, valign: "top", margin: 0 });
});
footer(s, 11);

// =========================================================================
// SLIDE 12 — A6b regularization
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A6b", "Regularization — Shrinking the Variance Gap");
card(s, M, 1.95, 6.5, 4.9, WHITE);
s.addText("Train–CV accuracy gap: before vs after", { x: M + 0.2, y: 2.1, w: 6.1, h: 0.4, color: INK, bold: true, fontSize: 13, margin: 0 });
s.addChart(pres.charts.BAR, [
  { name: "default", labels: ["Random Forest", "Gradient Boosting"], values: [0.189, 0.124] },
  { name: "regularized", labels: ["Random Forest", "Gradient Boosting"], values: [0.040, 0.074] },
], { x: M + 0.1, y: 2.5, w: 6.3, h: 4.1, barDir: "col", chartColors: [RED, GREEN], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: BODY, dataLabelFontSize: 11, dataLabelFormatCode: "0.000", catAxisLabelColor: BODY, catAxisLabelFontSize: 12, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendColor: BODY });
bullets(s, [
  "Random Forest: gap crushed 0.189 → 0.040 (depth & leaf limits).",
  "Gradient Boosting: gap halved 0.124 → 0.074 — and CV accuracy rose.",
  "Logistic Regression is bias-limited: regularization couldn't help it.",
  "L1 (lasso) drove 11 of 27 coefficients to exactly zero — automatic feature selection.",
], 7.2, 2.2, 5.6, 3.5, 14);
card(s, 7.2, 5.5, 5.6, 1.3, "FCEDEB");
s.addText("The lesson", { x: 7.4, y: 5.6, w: 5.2, h: 0.35, color: RED, bold: true, fontSize: 13, margin: 0 });
s.addText("The cure depends on the diagnosis: high variance → constrain complexity; high bias → add capacity.", { x: 7.4, y: 5.95, w: 5.2, h: 0.8, color: BODY, fontSize: 12.5, valign: "top", margin: 0 });
footer(s, 12);

// =========================================================================
// SLIDE 13 — A6c threshold
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A6c", "Threshold Tuning — Prioritizing Recall");
s.addText("Default 0.50  vs  tuned 0.35", { x: M, y: 1.95, w: 6, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
s.addTable([
  [TH("Metric"), TH("0.50"), TH("0.35")],
  [TD("Missed cases (FN)", { align: "left" }), TD("10"), TD("4", { bold: true, color: GREEN })],
  [TD("Recall", { align: "left" }), TD("0.902"), TD("0.961", { bold: true, color: GREEN })],
  [TD("Precision", { align: "left" }), TD("0.821"), TD("0.778")],
  [TD("Specificity", { align: "left" }), TD("0.756"), TD("0.659")],
], { x: M, y: 2.35, w: 5.6, colW: [2.8, 1.4, 1.4], rowH: 0.46, border: { pt: 0.5, color: LINE }, valign: "middle" });
stat(s, M, 4.75, 2.7, 1.5, "−60%", "missed cases", RED);
card(s, 3.4, 4.75, 2.7, 1.5, INK);
s.addText("Why lower?", { x: 3.55, y: 4.85, w: 2.4, h: 0.35, color: WHITE, bold: true, fontSize: 12, margin: 0 });
s.addText("A missed heart disease ≫ a false alarm. We accept 8 extra alarms to catch 6 more real cases.", { x: 3.55, y: 5.18, w: 2.4, h: 1.0, color: "CADCFC", fontSize: 11, valign: "top", margin: 0 });
card(s, 6.45, 1.95, 6.35, 4.9, WHITE);
img(s, "a6c_threshold_sweep.png", 1.6, 6.55, 2.1, 6.15, 4.6);
footer(s, 13);

// =========================================================================
// SLIDE 14 — A7 clustering
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A7", "Risk Phenotypes — K-means (k = 4)");
s.addTable([
  [TH("Pri"), TH("n"), TH("Disease"), TH("Phenotype")],
  [TD("1", { bold: true, color: RED }), TD("74"), TD("0.84"), TD("Advanced multi-vessel disease", { align: "left" })],
  [TD("2", { bold: true, color: RED }), TD("211"), TD("0.74"), TD("Hypertensive-hypercholesterolemic", { align: "left" })],
  [TD("3", { bold: true, color: TEAL }), TD("292"), TD("0.66"), TD("Reduced-exertion / low max-HR", { align: "left" })],
  [TD("4", { bold: true, color: TEAL }), TD("343"), TD("0.28"), TD("Younger lower-risk", { align: "left" })],
], { x: M, y: 2.05, w: 6.7, colW: [0.7, 0.8, 1.1, 4.1], rowH: 0.5, border: { pt: 0.5, color: LINE }, valign: "middle" });
card(s, M, 4.85, 6.7, 2.0, CARD);
bullets(s, [
  "Unsupervised: target hidden during fitting, used only to interpret.",
  "Chose k=4 for clinical granularity (silhouette was flat) — a transparent judgment call.",
  "Each phenotype gives the AI agent a priority hint beyond raw probability.",
], M + 0.2, 4.97, 6.3, 1.8, 12.5);
card(s, 7.35, 1.95, 5.45, 4.9, WHITE);
s.addText("Clusters in 2D (PCA view)", { x: 7.55, y: 2.05, w: 5.0, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
img(s, "a7_clusters_pca.png", 1.273, 7.45, 2.45, 5.25, 4.3);
footer(s, 14);

// =========================================================================
// SLIDE 15 — A8 save & bridge
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "STEP A8", "Persist & Bridge to the AI Phase");
s.addText("Saved artifacts (models/)", { x: M, y: 1.95, w: 6, h: 0.35, color: INK, bold: true, fontSize: 14, margin: 0 });
bullets(s, [
  "risk_model.joblib — regularized Gradient Boosting pipeline (refit on all 920).",
  "phenotype_model.joblib — K-means impute+scale+cluster.",
  "metadata.json — threshold 0.35, feature list, phenotype map, test metrics.",
], M, 2.35, 6.2, 1.9, 13);
const a8 = [["0.961", "Recall", RED], ["0.908", "ROC-AUC", RED], ["0.826", "Accuracy", TEAL]];
a8.forEach((c, i) => stat(s, M + i * 2.13, 4.35, 1.95, 1.2, c[0], c[1], c[2]));
s.addText("Held-out test metrics @ threshold 0.35", { x: M, y: 5.6, w: 6.2, h: 0.3, color: MUTED, italic: true, fontSize: 11, margin: 0 });
card(s, 7.0, 1.95, 5.8, 4.9, INK);
s.addText("model_api.assess(patient)", { x: 7.2, y: 2.1, w: 5.4, h: 0.4, color: WHITE, bold: true, fontSize: 14, fontFace: "Consolas", margin: 0 });
s.addText([
  { text: "HIGH-risk patient", options: { color: "FF9B8A", bold: true, breakLine: true, fontSize: 12 } },
  { text: "  probability : 0.976  → high-risk", options: { color: "CADCFC", breakLine: true, fontSize: 12 } },
  { text: "  phenotype   : Advanced multi-vessel (priority 1)", options: { color: "CADCFC", breakLine: true, fontSize: 12 } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "LOW-risk patient", options: { color: "9BE3C4", bold: true, breakLine: true, fontSize: 12 } },
  { text: "  probability : 0.013  → cleared", options: { color: "CADCFC", breakLine: true, fontSize: 12 } },
  { text: "  phenotype   : Younger lower-risk (priority 4)", options: { color: "CADCFC", fontSize: 12 } },
], { x: 7.2, y: 2.65, w: 5.4, h: 3.0, fontFace: "Consolas", valign: "top", margin: 0 });
s.addText("One clean call returns risk + phenotype — the launchpad for Phase B.", { x: 7.2, y: 6.0, w: 5.4, h: 0.7, color: "9FB3C8", italic: true, fontSize: 12, valign: "top", margin: 0 });
footer(s, 15);

// =========================================================================
// SLIDE 16 — Headline results
// =========================================================================
s = pres.addSlide(); s.background = { color: WHITE };
header(s, "RESULTS", "Phase A — Headline Results");
const hl = [["0.908", "ROC-AUC", RED], ["0.961", "Recall (tuned)", RED], ["4 / 102", "Missed cases", GREEN], ["4", "Risk phenotypes", TEAL]];
hl.forEach((c, i) => stat(s, M + i * 3.13, 2.0, 2.9, 1.5, c[0], c[1], c[2]));
card(s, M, 3.85, 6.2, 2.95, WHITE);
s.addText("Recall across models", { x: M + 0.2, y: 3.98, w: 5.8, h: 0.35, color: INK, bold: true, fontSize: 13, margin: 0 });
s.addChart(pres.charts.BAR, [{ name: "recall", labels: ["LogReg", "Rand. Forest", "Grad. Boost", "GB @ 0.35"], values: [0.843, 0.863, 0.892, 0.961] }], {
  x: M + 0.1, y: 4.35, w: 6.0, h: 2.35, barDir: "col", chartColors: [TEAL, TEAL, TEAL, RED], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: BODY, dataLabelFontSize: 11, dataLabelFormatCode: "0.000", catAxisLabelColor: BODY, catAxisLabelFontSize: 11, valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false,
});
card(s, 6.95, 3.85, 5.85, 2.95, CARD);
s.addText("What we delivered", { x: 7.15, y: 3.98, w: 5.5, h: 0.4, color: INK, bold: true, fontSize: 14, margin: 0 });
bullets(s, [
  "Best model: regularized Gradient Boosting.",
  "Full leakage-safe pipeline, reused across every step.",
  "Tuned for safety: only 4 missed cases out of 102.",
  "Four interpretable phenotypes to guide triage.",
  "One-call interface bridging ML → AI.",
], 7.15, 4.4, 5.5, 2.3, 13);
footer(s, 16);

// =========================================================================
// SLIDE 17 — Closing
// =========================================================================
s = pres.addSlide(); s.background = { color: SOFT };
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.3, w: 0.22, h: 1.4, fill: { color: GREEN }, rectRadius: 0.05, line: { type: "none" } });
s.addText("Machine Learning Phase: Complete", { x: 0.95, y: 1.3, w: 11.5, h: 0.9, color: INK, bold: true, fontSize: 40, fontFace: "Georgia", margin: 0 });
s.addText("A trained, evaluated, deployable cardiac-risk engine.", { x: 0.97, y: 2.25, w: 11.5, h: 0.5, color: TEAL, fontSize: 18, margin: 0 });
bullets(s, [
  "Inspected, cleaned, and encoded 920 records — leakage-safe.",
  "Compared three models; chose Gradient Boosting on recall & AUC.",
  "Diagnosed bias/variance, regularized, and tuned the threshold for safety.",
  "Discovered four clinical risk phenotypes with K-means.",
  "Saved the model and exposed assess(patient) for the AI phase.",
], 0.97, 3.05, 8.2, 2.7, 15);
card(s, 9.3, 3.0, 3.5, 2.9, INK);
s.addText("Next", { x: 9.55, y: 3.2, w: 3.0, h: 0.4, color: WHITE, bold: true, fontSize: 18, fontFace: "Georgia", margin: 0 });
s.addText("Say “start the AI phase” to build A* search, the CSP allocator, the knowledge base, and the Gemini + LangChain agent — all on top of assess().", { x: 9.55, y: 3.7, w: 3.0, h: 2.0, color: "CADCFC", fontSize: 13, valign: "top", margin: 0 });
s.addText("CardioTriage AI  ·  BSAI Combined Assessment", { x: 0.95, y: 6.7, w: 11.5, h: 0.4, color: MUTED, fontSize: 12, margin: 0 });

pres.writeFile({ fileName: "CardioTriage_ML_Phase.pptx" }).then((f) => console.log("Saved:", f));
