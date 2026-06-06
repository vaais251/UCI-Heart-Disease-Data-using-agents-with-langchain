// CardioTriage AI — Reasoning Console front-end.
// Stage (b): load preset patients, populate the selector + summary chip,
// and enable the Run button. The reasoning run itself arrives in stage (c).

const els = {
  select: document.getElementById("patient-select"),
  chip: document.getElementById("patient-chip"),
  runBtn: document.getElementById("btn-run"),
};

// Patients keyed by id, cached after the initial fetch so selecting one is
// instant (no extra round-trip just to update the chip).
const patients = {};
let selectedId = null;

// On page load, ask the backend for the preset cases + their ML summaries.
// `fetch` returns a Promise; `await` pauses until the JSON arrives.
init();

async function init() {
  try {
    const res = await fetch("/ai/patients");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    populateSelect(data.patients);
  } catch (err) {
    els.chip.querySelector(".chip-label").textContent = "Failed to load patients";
    els.chip.querySelector(".chip-meta").textContent = err.message;
    console.error(err);
  }
}

function populateSelect(list) {
  els.select.innerHTML = '<option value="" disabled selected>Select patient…</option>';
  list.forEach((p) => {
    patients[p.id] = p;
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.label;
    els.select.appendChild(opt);
  });
  els.select.disabled = false;
}

// When the clinician picks a patient: update the chip and enable "Run".
els.select.addEventListener("change", () => {
  selectedId = els.select.value;
  const p = patients[selectedId];
  if (!p) return;

  const pct = (p.risk_probability * 100).toFixed(1);
  const level = p.is_high_risk ? "high" : "low";
  els.chip.classList.remove("chip-high", "chip-low");
  els.chip.classList.add(`chip-${level}`);
  els.chip.querySelector(".chip-label").textContent =
    p.is_high_risk ? "High risk" : "Low risk";
  els.chip.querySelector(".chip-meta").innerHTML =
    `ML risk ${pct}%&nbsp;·&nbsp;${p.phenotype}`;

  els.runBtn.disabled = false;
});

// ===== Stage (c): run the A* search and animate the path =====

els.runBtn.addEventListener("click", runReasoning);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const ZONES = ["zone-search", "zone-trace", "zone-briefing"];

async function runReasoning() {
  if (!selectedId) return;
  setRunning(true);
  resetZones();
  try {
    // Zone 01 — A* search path.
    setActiveZone("zone-search");
    const search = await getJSON(`/ai/search/${selectedId}`);
    await renderSearchPath(search);

    // Zone 02 — forward-chaining trace (runs after the path is drawn).
    setActiveZone("zone-trace");
    const trace = await getJSON(`/ai/trace/${selectedId}`);
    await renderTrace(trace);

    // Zone 03 — Gemini briefing (typed out after the trace completes).
    setActiveZone("zone-briefing");
    setBriefingPending();
    const brief = await getJSON(`/ai/briefing/${selectedId}`);
    await renderBriefing(brief);
  } catch (err) {
    zoneError("zone-trace", err.message);
    console.error(err);
  } finally {
    setActiveZone(null);
    setRunning(false);
  }
}

// Clear all three zones to a "waiting" state so a re-run never shows stale
// content from the previous patient.
function resetZones() {
  zonePending("zone-search", "Searching for the optimal pathway");
  zonePending("zone-trace", "Awaiting inference");
  zonePending("zone-briefing", "Awaiting briefing");
}

function zonePending(zoneId, label) {
  const body = document.querySelector(`#${zoneId} .panel-body`);
  body.innerHTML =
    `<div class="zone-waiting"><span class="dots"><i></i><i></i><i></i></span>${label}</div>`;
}

// Glow the panel currently being computed; clear the others.
function setActiveZone(zoneId) {
  ZONES.forEach((z) => {
    document.getElementById(z).classList.toggle("panel-active", z === zoneId);
  });
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
  return res.json();
}

function setRunning(isRunning) {
  els.runBtn.disabled = isRunning;
  els.select.disabled = isRunning;
  els.runBtn.classList.toggle("running", isRunning);
  els.runBtn.lastChild.textContent = isRunning ? " Reasoning…" : " Run reasoning";
}

// Render the path node-by-node: each node is appended, then we pause briefly
// before the next so the chain visibly "grows" toward the goal.
async function renderSearchPath(data) {
  const body = document.querySelector("#zone-search .panel-body");
  body.innerHTML = '<div class="search-chain" id="search-chain"></div>';
  const chain = document.getElementById("search-chain");

  for (const node of data.nodes) {
    const el = buildNode(node);
    chain.appendChild(el);
    // Mark just-added node as "active" (soft pulse); clear the previous one.
    chain.querySelectorAll(".node.active").forEach((n) => n.classList.remove("active"));
    el.classList.add("active");
    await sleep(520);
  }
  chain.querySelectorAll(".node.active").forEach((n) => n.classList.remove("active"));

  const meta = document.createElement("div");
  meta.className = "search-meta fade-in";
  meta.innerHTML =
    `<span>plan cost <b>${data.total_cost}</b></span>` +
    `<span>nodes expanded <b>${data.nodes_expanded}</b></span>` +
    `<div class="search-decision">${data.decision}</div>`;
  body.appendChild(meta);
}

function buildNode(node) {
  const el = document.createElement("div");
  el.className = `node node-${node.tone}`;
  const cost = node.cost != null ? `<span class="node-cost">cost ${node.cost}</span>` : "";
  el.innerHTML = `
    <span class="node-dot"></span>
    <div class="node-body">
      <div class="node-kicker">${node.label}</div>
      <div class="node-title">${node.title}${cost}</div>
      ${node.subtitle ? `<div class="node-sub">${node.subtitle}</div>` : ""}
    </div>`;
  return el;
}

// ===== Zone 02 — forward-chaining trace (streamed line by line) =====
async function renderTrace(data) {
  const body = document.querySelector("#zone-trace .panel-body");
  body.innerHTML = '<div class="log" id="trace-log"></div>';
  const log = document.getElementById("trace-log");

  if (!data.lines.length) {
    log.appendChild(textLine("(no rules fired)", "log-dim"));
  }

  let lastPass = 0;
  for (const line of data.lines) {
    if (line.pass !== lastPass) {
      lastPass = line.pass;
      log.appendChild(textLine(`— pass ${line.pass} —`, "log-pass-head"));
      log.scrollTop = log.scrollHeight;
      await sleep(220);
    }
    log.appendChild(buildLogLine(line));
    log.scrollTop = log.scrollHeight;
    await sleep(380);
  }

  const done = textLine("no new facts — inference complete", "log-done");
  log.appendChild(done);
  log.scrollTop = log.scrollHeight;

  if (data.override_count > 0) {
    const warn = textLine(
      `⚠ ${data.override_count} safety override(s) fired`, "log-override-note");
    log.appendChild(warn);
  }
}

function buildLogLine(line) {
  const el = document.createElement("div");
  el.className = "log-line" + (line.override ? " log-line-override" : "");

  const facts = line.facts
    .map((f) => {
      const cls = f.category === "override" ? "fact fact-override" : "fact";
      return `<span class="${cls}" title="${f.meaning}">${f.key}</span>`;
    })
    .join(" ");

  const tag = line.override ? `<span class="ov-tag">SAFETY OVERRIDE</span>` : "";
  el.innerHTML = `
    <div class="log-main">
      <span class="log-rule">${line.rule}</span>
      <span class="log-arrow">▸</span>
      ${facts}${tag}
    </div>
    <div class="log-why">IF ${line.if_text}</div>`;
  return el;
}

function textLine(text, cls) {
  const el = document.createElement("div");
  el.className = cls + " fade-in";
  el.textContent = text;
  return el;
}

// ===== Zone 03 — Gemini briefing (typewriter) + safety-override box =====
function setBriefingPending() {
  const body = document.querySelector("#zone-briefing .panel-body");
  body.innerHTML = `<p class="briefing-pending">Composing briefing…</p>`;
}

async function renderBriefing(data) {
  const body = document.querySelector("#zone-briefing .panel-body");
  const sourceTag = data.source === "gemini"
    ? `<span class="src src-live">● live Gemini</span>`
    : `<span class="src src-fallback">○ deterministic</span>`;

  body.innerHTML = `
    <div class="briefing">
      <div class="briefing-bar">${sourceTag}</div>
      <p class="briefing-text" id="briefing-text"><span class="caret" id="caret"></span></p>
    </div>`;

  const textEl = document.getElementById("briefing-text");
  const caret = document.getElementById("caret");
  await typewriter(textEl, caret, data.text);
  caret.remove();

  // Prominent amber safety-override box when the KB vetoed anything.
  if (data.overrides && data.overrides.length) {
    body.querySelector(".briefing").appendChild(buildSafetyBox(data));
  }
}

// Insert characters before the blinking caret one at a time.
async function typewriter(el, caret, text, speed = 12) {
  for (const ch of text) {
    caret.insertAdjacentText("beforebegin", ch);
    await sleep(ch === " " ? speed / 2 : speed);
  }
}

function buildSafetyBox(data) {
  const box = document.createElement("div");
  box.className = "safety-box";
  const items = data.overrides.map((o) => `<li>${o}</li>`).join("");
  const mods = (data.safety_modifications || []).map((m) => `<li>${m}</li>`).join("");
  box.innerHTML = `
    <div class="safety-head">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
      <span>Safety override — knowledge base vetoes</span>
    </div>
    <ul class="safety-list">${items}</ul>
    ${mods ? `<div class="safety-sub">Plan modifications:</div><ul class="safety-list">${mods}</ul>` : ""}`;
  return box;
}

function zoneError(zoneId, message) {
  const body = document.querySelector(`#${zoneId} .panel-body`);
  body.innerHTML = `<p class="zone-error">⚠️ ${message}</p>`;
}

console.log("CardioTriage AI — Reasoning Console (stage c) loaded.");
