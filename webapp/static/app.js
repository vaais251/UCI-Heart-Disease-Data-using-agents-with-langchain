// CardioTriage AI — front-end logic.
// Stage (d): render the polished prediction panel (badge, progress bar,
// side-by-side model comparison, phenotype) with a smooth fade-in.

const form = document.getElementById("patient-form");
const resultPanel = document.getElementById("result-panel");
const submitBtn = form.querySelector(".btn-predict");

// ---- Friendly validation messages ----
// Replace the browser's terse default ("Please fill out this field") with
// warmer, field-specific text. 'invalid' fires per field when the form is
// submitted with bad/empty input; 'input' clears the message once corrected.
form.querySelectorAll("input, select").forEach((el) => {
  const label = (el.closest(".field")?.querySelector("label")?.textContent || "this field")
    .replace(/\s+/g, " ").trim();
  el.addEventListener("invalid", () => {
    if (el.validity.valueMissing) {
      el.setCustomValidity(`Please enter ${labelPhrase(label)}.`);
    } else if (el.validity.rangeUnderflow || el.validity.rangeOverflow) {
      el.setCustomValidity(`Please enter a realistic value for ${labelPhrase(label)} (${el.min}–${el.max}).`);
    } else {
      el.setCustomValidity("");
    }
  });
  el.addEventListener("input", () => el.setCustomValidity(""));
  el.addEventListener("change", () => el.setCustomValidity(""));
});

function labelPhrase(label) {
  // "Age years" -> "age"; strip trailing unit words for a natural sentence.
  return label.replace(/\s*(years|mm Hg|mg\/dl|bpm|0–3)\s*$/i, "").toLowerCase();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Native HTML5 validation gate — shows our friendly bubbles if anything's off.
  if (!form.reportValidity()) return;

  // Collect the form into an object: { age: "54", sex: "Male", ... }.
  // The server (Pydantic) coerces strings -> numbers/booleans and validates.
  const patient = Object.fromEntries(new FormData(form).entries());

  setLoading(true);
  renderLoading();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patient),
    });

    if (!response.ok) {
      // 422 = validation error from FastAPI; anything else = server error.
      throw new Error(`Prediction failed (HTTP ${response.status})`);
    }

    const data = await response.json();
    console.log("Prediction:", data);
    renderResult(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.textContent = isLoading ? "Assessing…" : "Predict risk";
}

// ---- Rendering helpers ----

function pct(value) {
  return value === null || value === undefined ? null : value * 100;
}

function renderLoading() {
  resultPanel.innerHTML = `
    <h2>Prediction result</h2>
    <p class="placeholder">Assessing…</p>`;
}

function renderError(message) {
  resultPanel.innerHTML = `
    <h2>Prediction result</h2>
    <p class="error-msg">⚠️ ${message}</p>`;
}

function renderResult(data) {
  const high = data.is_high_risk;
  const main = pct(data.probability);          // deployed Gradient Boosting
  const gb = pct(data.gradient_boosting);
  const rf = pct(data.random_forest);
  const thresholdPct = pct(data.threshold);
  const level = high ? "high" : "low";
  const ph = data.phenotype;
  const diseaseRate = pct(ph.disease_rate);

  resultPanel.innerHTML = `
    <h2>Prediction result</h2>
    <div class="result fade-in">

      <div class="risk-headline">
        <div class="risk-figure ${level}">
          <span class="risk-pct">${main.toFixed(1)}<small>%</small></span>
          <span class="risk-caption">estimated risk of heart disease</span>
        </div>
        <span class="badge ${level}">${high ? "High risk" : "Low risk"}</span>
      </div>

      <div class="risk-bar"
           role="img" aria-label="Risk ${main.toFixed(1)} percent">
        <div class="risk-bar-fill ${level}" style="width:${main.toFixed(1)}%"></div>
        <div class="risk-bar-threshold" style="left:${thresholdPct.toFixed(1)}%"
             title="Decision threshold ${thresholdPct.toFixed(0)}%"></div>
      </div>
      <p class="bar-note">Decision threshold
        <strong>${thresholdPct.toFixed(0)}%</strong> — at or above this the
        patient is flagged high risk.</p>

      <h3 class="section-title">Model comparison</h3>
      <div class="model-grid">
        ${modelCard("Gradient Boosting", gb, "deployed", true)}
        ${modelCard("Random Forest", rf, "comparison", false)}
      </div>

      <h3 class="section-title">Risk phenotype</h3>
      <div class="phenotype">
        <div class="phenotype-row">
          <span class="phenotype-name">${ph.name}</span>
          <span class="priority-tag p${ph.priority}">Triage priority ${ph.priority}</span>
        </div>
        ${diseaseRate !== null
          ? `<p class="phenotype-desc">This patient profile historically has a
               <strong>${diseaseRate.toFixed(0)}%</strong> disease rate
               (cluster of ${ph.size} similar patients).</p>`
          : ""}
      </div>

    </div>`;
}

function modelCard(name, value, tag, primary) {
  if (value === null) {
    return `
      <div class="model-card">
        <div class="model-head"><span>${name}</span></div>
        <div class="model-pct muted">n/a</div>
      </div>`;
  }
  return `
    <div class="model-card${primary ? " primary" : ""}">
      <div class="model-head">
        <span>${name}</span><em class="model-tag">${tag}</em>
      </div>
      <div class="model-pct">${value.toFixed(1)}<small>%</small></div>
      <div class="mini-bar"><div class="mini-bar-fill" style="width:${value.toFixed(1)}%"></div></div>
    </div>`;
}

console.log("CardioTriage AI UI loaded.");
