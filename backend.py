import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from ml_engine.qc_engine import WeatherQCEngine

app = FastAPI(
    title="SkyGuard AI — AWS Quality Control & Anomaly Engine",
    description="Intelligent anomaly detection engine detecting spikes, frozen flatlines, progressive drift, physical violations, and genuine regional extreme weather."
)

FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://skyguardapp.streamlit.app")

ALLOWED_ORIGINS = [
    "https://skyguardapp.streamlit.app",
    "https://skyguardapp.streamlit.app/",
    FRONTEND_URL,
    FRONTEND_URL.rstrip("/"),
    "http://localhost:8501",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint providing service metadata and links."""
    return {
        "status": "online",
        "service": "SkyGuard AI — AWS Quality Control & Anomaly Engine",
        "version": "1.0.0",
        "frontend": FRONTEND_URL,
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "detect": "/api/v1/detect",
            "latest": "/api/v1/latest",
            "stations": "/api/v1/stations",
            "scenario": "/api/v1/scenario"
        }
    }


@app.get("/health")
def health():
    """Health check for uptime monitors and deployment verification."""
    return {
        "status": "healthy",
        "service": "skyguard-backend",
        "qc_engine": "operational",
        "active_scenario": active_scenario.get("scenario", "nominal"),
        "stations_monitored": len(station_histories),
        "frontend_url": FRONTEND_URL
    }



class SensorInput(BaseModel):
    station_id: str
    temperature: float
    humidity: float
    pressure: float
    is_extreme_scenario: Optional[bool] = False
    temp_history: Optional[List[float]] = None
    peer_readings: Optional[Dict[str, float]] = None


class ScenarioInput(BaseModel):
    scenario: str  # "nominal", "spike", "frozen", "drift", "regional_heatwave", "regional_squall"
    target_station_id: Optional[str] = "AWS-IND-001"
    drift_magnitude: Optional[float] = 4.5
    anomaly_parameter: Optional[str] = "Temperature"


# Initialize QC Engine
qc_engine = WeatherQCEngine()

# In-memory station buffers and state
station_histories: Dict[str, Dict[str, List[float]]] = {}
stations_state: Dict[str, Dict[str, Any]] = {}
active_scenario: Dict[str, Any] = {"scenario": "nominal"}
latest_telemetry = {}
latest_result = {}

# Default station seeds to ensure spatial buddy checks work out of the box
DEFAULT_PEERS = {
    "AWS-IND-001": 28.5,
    "AWS-IND-002": 28.1,
    "AWS-IND-003": 28.8,
    "AWS-IND-004": 28.4,
    "AWS-IND-005": 28.6,
    "AWS-IND-006": 30.2,
    "AWS-IND-007": 24.5,
}

for sid, t_val in DEFAULT_PEERS.items():
    station_histories[sid] = {
        "temperature": [t_val + round(np.sin(i / 5.0) * 1.2, 2) for i in range(15)],
        "humidity": [60.0 for _ in range(15)],
        "pressure": [1011.0 for _ in range(15)],
    }


@app.post("/api/v1/detect")
def detect_anomaly(data: SensorInput):
    """
    Main detection endpoint executing hybrid AI/Physics/Temporal/Spatial QC.
    Distinguishes sensor faults (spikes, frozen, drift, bounds) from genuine regional extreme weather.
    """
    sid = data.station_id

    # Initialize history for new station if not present
    if sid not in station_histories:
        station_histories[sid] = {"temperature": [], "humidity": [], "pressure": []}

    # Record reading
    hist = station_histories[sid]["temperature"]
    hist.append(data.temperature)
    if len(hist) > 30:
        hist.pop(0)

    # Gather peer readings (temperature across active stations in same regional cluster)
    peer_temps = {}
    if data.peer_readings:
        peer_temps = data.peer_readings
    else:
        for p_id, p_h in station_histories.items():
            if p_id != sid and p_h["temperature"]:
                peer_temps[p_id] = p_h["temperature"][-1]

    # Evaluate via QC engine
    result = qc_engine.evaluate(
        station_id=sid,
        temperature=data.temperature,
        humidity=data.humidity,
        pressure=data.pressure,
        temp_history=data.temp_history or hist,
        peer_temp_readings=peer_temps,
        is_extreme_scenario=data.is_extreme_scenario or (active_scenario.get("scenario") == "regional_heatwave")
    )

    global latest_telemetry, latest_result
    latest_telemetry = data.model_dump()
    latest_result = result
    stations_state[sid] = {
        "telemetry": latest_telemetry,
        "result": result
    }

    return result


@app.post("/api/v1/scenario")
def set_scenario(config: ScenarioInput):
    """
    Set active simulation scenario across the fleet for testing and live demonstrations:
    - nominal: All stations reporting clean nominal data
    - spike: Single station experiences abrupt thermal/ADC surge
    - frozen: Sensor output flatlines at fixed reading
    - drift: Sensor progressively drifts away from true value
    - regional_heatwave: Multi-station regional heatwave where all NCR stations report 47-49°C
    """
    global active_scenario
    active_scenario = config.model_dump()

    target_id = config.target_station_id or "AWS-IND-001"
    scenario = config.scenario

    if scenario == "regional_heatwave":
        # All NCR regional stations experience extreme heatwave
        ncr_stations = ["AWS-IND-001", "AWS-IND-002", "AWS-IND-003", "AWS-IND-004", "AWS-IND-005"]
        for st_id in ncr_stations:
            station_histories[st_id]["temperature"] = [46.0, 47.2, 48.0, 48.5]
    elif scenario == "nominal":
        for st_id, base_t in DEFAULT_PEERS.items():
            station_histories[st_id]["temperature"] = [base_t for _ in range(15)]

    return {
        "status": "SCENARIO_SET",
        "active_scenario": active_scenario,
        "message": f"Simulation scenario '{scenario}' configured."
    }


@app.get("/api/v1/latest")
def get_latest():
    """Get the latest ingested telemetry and QC assessment."""
    if not latest_result:
        return {
            "status": "NO_DATA",
            "message": "Waiting for sensor telemetry..."
        }

    return {
        "telemetry": latest_telemetry,
        "result": latest_result,
        "active_scenario": active_scenario
    }


@app.get("/api/v1/stations")
def get_stations():
    """Get the current operational status and consensus across all fleet stations."""
    return {
        "active_stations": list(station_histories.keys()),
        "stations_state": stations_state,
        "active_scenario": active_scenario
    }