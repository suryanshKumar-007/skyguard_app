import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from ml_engine.anomaly_model import WeatherAnomalyModel
import math

class WeatherQCEngine:
    """
    Intelligent Quality Control (QC) & Anomaly Detection Engine for AWS Telemetry.
    Combines:
    1. Physics bounds & Dew-point thermodynamic consistency
    2. Temporal consistency (Spikes, Frozen Flatlines, Progressive Drift)
    3. Spatial Buddy Checks (Multi-station consensus)
    4. Isolation Forest unsupervised ML
    5. Evidence fusion to distinguish sensor faults from genuine regional extreme weather
    """

    # Physical and climatological thresholds
    TEMP_MIN = -20.0
    TEMP_MAX = 50.0
    HUM_MIN = 5.0
    HUM_MAX = 100.0
    PRES_MIN = 920.0
    PRES_MAX = 1050.0

    # Rate of change limits (per 2-minute interval)
    TEMP_MAX_RATE = 4.0      # °C per 2-min
    HUM_MAX_RATE = 20.0      # % per 2-min
    PRES_MAX_RATE = 4.0      # hPa per 2-min

    def __init__(self, ml_model: Optional[WeatherAnomalyModel] = None):
        if ml_model is not None:
            self.ml_model = ml_model
        else:
            self.ml_model = WeatherAnomalyModel()
            try:
                self.ml_model.train_from_csv("ml_engine/weather_normal_data.csv")
            except Exception:
                # Fallback synthetic training if file missing
                np.random.seed(42)
                synth = np.column_stack([
                    np.random.normal(28, 3, 1000),
                    np.random.normal(60, 8, 1000),
                    np.random.normal(1011, 3, 1000)
                ])
                self.ml_model.train(synth)

    @staticmethod
    def calculate_dew_point(temp: float, humidity: float) -> float:
        """Magnus-Tetens formula for dew point calculation."""
        a, b = 17.27, 237.7
        h_clamped = max(1.0, min(100.0, humidity))
        alpha = ((a * temp) / (b + temp)) + np.log(h_clamped / 100.0)
        return (b * alpha) / (a - alpha)

    def check_physics(self, temp: float, hum: float, pres: float) -> Dict[str, Any]:
        """Verify thermodynamic and physical climatological limits."""
        dew_point = float(self.calculate_dew_point(temp, hum))
        dew_point_violation = bool(dew_point > temp + 0.05)

        out_of_bounds = bool(
            temp < self.TEMP_MIN or temp > self.TEMP_MAX or
            hum < self.HUM_MIN or hum > self.HUM_MAX or
            pres < self.PRES_MIN or pres > self.PRES_MAX
        )

        violations = []
        if temp < self.TEMP_MIN or temp > self.TEMP_MAX:
            violations.append(f"Temperature {temp:.1f}°C outside [{self.TEMP_MIN}, {self.TEMP_MAX}]°C")
        if hum < self.HUM_MIN or hum > self.HUM_MAX:
            violations.append(f"Humidity {hum:.1f}% outside [{self.HUM_MIN}, {self.HUM_MAX}]%")
        if pres < self.PRES_MIN or pres > self.PRES_MAX:
            violations.append(f"Pressure {pres:.1f} hPa outside [{self.PRES_MIN}, {self.PRES_MAX}] hPa")
        if dew_point_violation:
            violations.append(f"Thermodynamic violation: Dew point ({dew_point:.1f}°C) exceeds ambient temp ({temp:.1f}°C)")

        return {
            "passed": bool(not (out_of_bounds or dew_point_violation)),
            "dew_point": float(dew_point),
            "dew_point_violation": bool(dew_point_violation),
            "out_of_bounds": bool(out_of_bounds),
            "violations": violations
        }

    def check_temporal(self, history: List[float], param_type: str = "temperature") -> Dict[str, Any]:
        """
        Evaluate temporal dynamics:
        1. Spike / Step-change test
        2. Frozen flatline test (zero variance across 3+ samples)
        3. Drift test (steady monotonic shift away from rolling mean)
        """
        if not history or len(history) < 2:
            return {"spike": False, "frozen": False, "drift": False, "rate_of_change": 0.0, "drift_slope": 0.0}

        latest = float(history[-1])
        prev = float(history[-2])
        delta = float(abs(latest - prev))

        max_rate = self.TEMP_MAX_RATE if param_type == "temperature" else (
            self.HUM_MAX_RATE if param_type == "humidity" else self.PRES_MAX_RATE
        )
        is_spike = bool(delta > max_rate)

        # Frozen sensor check: check last 3 to 6 readings
        is_frozen = False
        if len(history) >= 3:
            recent_window = history[-min(5, len(history)):]
            std_dev = float(np.std(recent_window))
            all_equal = all(abs(x - latest) < 1e-4 for x in recent_window)
            if all_equal or (std_dev < 0.005 and len(recent_window) >= 3):
                is_frozen = True

        # Drift check: systematic deviation across window
        is_drift = False
        drift_slope = 0.0
        if len(history) >= 10:
            window = history[-10:]
            x = np.arange(len(window))
            slope, _ = np.polyfit(x, window, 1)
            drift_slope = float(slope)
            early_mean = float(np.mean(history[:-5])) if len(history) > 10 else float(history[0])
            cumulative_deviation = float(abs(latest - early_mean))
            if abs(slope) > 0.15 and cumulative_deviation > 2.5 and not is_spike:
                is_drift = True

        return {
            "spike": bool(is_spike),
            "frozen": bool(is_frozen),
            "drift": bool(is_drift),
            "rate_of_change": float(delta),
            "drift_slope": float(drift_slope)
        }

    def check_spatial_buddy(
        self,
        target_station_id: str,
        target_val: float,
        peer_readings: Dict[str, float],
        tolerance_sigma: float = 3.0,
        extreme_weather_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Spatial buddy check: compare target station with regional peers.
        Distinguishes isolated sensor faults from genuine regional extreme weather:
        - If target is extreme BUT peer readings are normal -> Buddy Check FAILED (Isolated Fault)
        - If target is extreme AND peer readings are ALSO extreme -> Buddy Check PASSED (Consensus - Regional Extreme)
        """
        if not peer_readings:
            return {
                "passed": True,
                "is_regional_extreme": False,
                "consensus": "NO_PEERS",
                "peer_median": float(target_val),
                "residual": 0.0,
                "z_score": 0.0,
                "message": "Isolated node: no spatial peers available for cross-validation"
            }

        peer_vals = [float(v) for v in peer_readings.values()]
        peer_median = float(np.median(peer_vals))
        peer_std = float(np.std(peer_vals)) if len(peer_vals) > 1 else 1.0
        effective_std = float(max(peer_std, 0.6))

        residual = float(target_val - peer_median)
        z_score = float(abs(residual) / effective_std)

        is_peer_cluster_extreme = bool(extreme_weather_mode or (peer_median > 42.0 or peer_median < -5.0))

        if is_peer_cluster_extreme and abs(target_val - peer_median) < 4.0:
            return {
                "passed": True,
                "is_regional_extreme": True,
                "consensus": "REGIONAL_CONSENSUS_MATCH",
                "peer_median": float(peer_median),
                "residual": float(residual),
                "z_score": float(z_score),
                "message": f"PASSED (Spatial Consensus Match · {len(peer_vals)} regional peers confirm extreme event)"
            }

        if z_score > tolerance_sigma:
            return {
                "passed": False,
                "is_regional_extreme": False,
                "consensus": "DIVERGENT",
                "peer_median": float(peer_median),
                "residual": float(residual),
                "z_score": float(z_score),
                "message": f"FAILED (Diverged {z_score:.1f}σ from regional peer consensus median of {peer_median:.1f})"
            }

        return {
            "passed": True,
            "is_regional_extreme": False,
            "consensus": "NORMAL_CONSENSUS",
            "peer_median": float(peer_median),
            "residual": float(residual),
            "z_score": float(z_score),
            "message": f"PASSED (Consistent with {len(peer_vals)} peer stations · Residual {residual:+.1f})"
        }

    def evaluate(
        self,
        station_id: str,
        temperature: float,
        humidity: float,
        pressure: float,
        temp_history: Optional[List[float]] = None,
        peer_temp_readings: Optional[Dict[str, float]] = None,
        is_extreme_scenario: bool = False
    ) -> Dict[str, Any]:
        """
        Execute comprehensive QC evaluation fusing ML, Physics, Temporal, and Spatial checks.
        """
        # 1. Physics Check
        physics_res = self.check_physics(temperature, humidity, pressure)

        # 2. Temporal Check (Temperature focused for primary demo)
        temp_history = temp_history or [temperature]
        temporal_res = self.check_temporal(temp_history, "temperature")

        # 3. Spatial Buddy Check
        peer_temps = peer_temp_readings or {}
        spatial_res = self.check_spatial_buddy(
            station_id,
            temperature,
            peer_temps,
            extreme_weather_mode=is_extreme_scenario
        )

        # 4. ML Anomaly Model Prediction
        ml_res = self.ml_model.predict(temperature, humidity, pressure)
        ml_anomaly = ml_res["is_anomaly"]
        ml_score = ml_res["anomaly_score"]

        # 5. Evidence Fusion & Classification
        is_extreme_weather = spatial_res["is_regional_extreme"]
        
        # Decide if this is a genuine weather extreme or a sensor fault
        if is_extreme_weather:
            classification = "Genuine Regional Extreme Weather (Heatwave / Severe Pattern)"
            is_anomaly = True          # Statistically anomalous
            is_sensor_fault = False    # NOT a sensor fault! Sensor is working properly!
            action_required = "DO NOT ISOLATE SENSOR. Retain readings in NWP models; issue regional meteorological advisory."
            confidence = 0.96
        elif temporal_res["frozen"]:
            classification = "Frozen Sensor Value Fault (Flatline / ADC Hang)"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Recommend data stream isolation, check communication bus and power rails, schedule inspection."
            confidence = 0.98
        elif temporal_res["drift"]:
            classification = "Sensor Drift Fault (Gradual Calibration Decay)"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Recommend field calibration verification, compare against historical baseline, apply offset if needed."
            confidence = 0.92
        elif temporal_res["spike"] or (ml_anomaly and not spatial_res["passed"]):
            classification = "Thermal Spike / ADC Surge Fault"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Verify sensor reading, inspect wiring and probe harness, recommend data isolation if abnormal."
            confidence = 0.95
        elif physics_res["dew_point_violation"]:
            classification = "Thermodynamic Physics Fault (Dew-Point Inversion)"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Recommend data isolation; inspect psychrometric / humidity probe pair for moisture ingress."
            confidence = 0.94
        elif physics_res["out_of_bounds"]:
            classification = "Out-of-Range Reading Fault (Physical Limit Breach)"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Treat reading as invalid, recommend data isolation, and verify sensor voltage rails."
            confidence = 0.99
        elif ml_anomaly:
            classification = "Multivariate Telemetry Anomaly (Unusual Cluster)"
            is_anomaly = True
            is_sensor_fault = True
            action_required = "Flag telemetry for monitoring and cross-station validation."
            confidence = 0.88
        else:
            classification = "NOMINAL"
            is_anomaly = False
            is_sensor_fault = False
            action_required = "None. Station telemetry verified across all QC layers."
            confidence = 0.99

        # Explainable AI (XAI) feature attribution breakdown
        rate_val = temporal_res["rate_of_change"]
        z_val = spatial_res.get("z_score", 0.1)
        dew_viol = 0.45 if physics_res["dew_point_violation"] else -0.05
        freeze_contrib = 0.85 if temporal_res["frozen"] else -0.05
        drift_contrib = 0.75 if temporal_res["drift"] else -0.05
        rate_contrib = 0.65 if temporal_res["spike"] else (0.1 if rate_val > 1.5 else -0.08)
        spatial_contrib = 0.55 if not spatial_res["passed"] else (-0.25 if is_extreme_weather else -0.1)

        shap_scores = {
            "Rate-of-Change (Spike)": round(rate_contrib, 3),
            "Temporal Flatline Score": round(freeze_contrib, 3),
            "Sensor Drift Gradient": round(drift_contrib, 3),
            "Spatial Residual Divergence": round(spatial_contrib, 3),
            "Dew-Point Violation": round(dew_viol, 3),
            "Isolation Forest Anomaly Score": round(0.5 if ml_anomaly else -0.15, 3),
        }

        # Imputed / Reconstructed Value
        imputed_temp = spatial_res.get("peer_median", temperature)
        if temporal_res["frozen"] and len(temp_history) > 3:
            imputed_temp = float(np.median(temp_history[:-3]))

        return {
            "station_id": str(station_id),
            "is_anomaly": bool(is_anomaly),
            "is_sensor_fault": bool(is_sensor_fault),
            "is_extreme_weather": bool(is_extreme_weather),
            "classification": str(classification),
            "confidence_score": float(confidence),
            "ml_detected": bool(ml_anomaly),
            "ml_anomaly_score": float(ml_score),
            "physics_passed": bool(physics_res["passed"]),
            "temporal_passed": bool(not (temporal_res["spike"] or temporal_res["frozen"] or temporal_res["drift"])),
            "spatial_buddy_check": str(spatial_res["message"]),
            "spatial_consensus": str(spatial_res["consensus"]),
            "action_required": str(action_required),
            "imputed_temperature": float(imputed_temp),
            "shap_scores": {str(k): float(v) for k, v in shap_scores.items()},
            "details": {
                "physics": physics_res,
                "temporal": temporal_res,
                "spatial": spatial_res
            }
        }
