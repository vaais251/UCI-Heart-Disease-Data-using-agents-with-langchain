"""
CardioTriage AI — ML web UI backend (FastAPI).

Stage (a): serve the single-page UI and a health check. The /predict endpoint
(stage c) will be added later.

Run with:  uv run uvicorn webapp.server:app --reload --port 8000
Then open:  http://127.0.0.1:8000
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Make the trained-model interface (ml/model_api.py) and this package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastapi import HTTPException  # noqa: E402
from model_api import full_assessment  # noqa: E402
from ai_service import (  # noqa: E402
    briefing,
    get_patient,
    list_patients,
    search_path,
    trace_log,
)

app = FastAPI(title="CardioTriage AI - Risk Assessment")

STATIC_DIR = Path(__file__).parent / "static"


class PatientIn(BaseModel):
    """Validated patient input. FastAPI rejects out-of-range/missing values
    automatically (e.g. negative age) and returns a 422 with details.
    Pydantic also coerces the form's strings ("54", "true") to int/bool."""
    age: int = Field(ge=18, le=110)
    sex: str
    cp: str
    trestbps: float = Field(ge=60, le=260)
    chol: float = Field(ge=0, le=700)
    fbs: bool
    restecg: str
    thalch: float = Field(ge=60, le=220)
    exang: bool
    oldpeak: float = Field(ge=0, le=10)
    slope: str
    ca: int = Field(ge=0, le=3)
    thal: str

# Serve CSS/JS/assets under /static (e.g. /static/style.css).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/ai")
def ai_console() -> FileResponse:
    """Serve the AI reasoning console (separate showcase page)."""
    return FileResponse(STATIC_DIR / "console.html")


@app.get("/ai/patients")
def ai_patients() -> dict:
    """List the preset showcase patients with their ML risk summaries."""
    return {"patients": list_patients()}


@app.get("/ai/search/{pid}")
def ai_search(pid: str) -> dict:
    """Run the A* planner for a preset patient and return the path nodes."""
    patient = get_patient(pid)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{pid}'")
    return search_path(patient)


@app.get("/ai/trace/{pid}")
def ai_trace(pid: str) -> dict:
    """Run the forward-chaining KB for a preset patient and return the trace."""
    patient = get_patient(pid)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{pid}'")
    return trace_log(patient)


@app.get("/ai/briefing/{pid}")
def ai_briefing(pid: str) -> dict:
    """Return the Gemini (or fallback) triage briefing + KB safety overrides."""
    patient = get_patient(pid)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient '{pid}'")
    return briefing(patient)


@app.get("/health")
def health() -> dict:
    """Simple liveness check."""
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientIn) -> dict:
    """Validate the patient, run both models + phenotype, return JSON."""
    return full_assessment(patient.model_dump())
