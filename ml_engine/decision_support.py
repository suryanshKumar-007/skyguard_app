from typing import Dict, List, Any, Optional
from datetime import datetime

class OperatorDecisionSupport:
    """
    Deterministic Operator Decision-Support Service for SkyGuard AI.
    Maps detected anomalies to standardized IMD/WMO AWS operational response procedures.
    IMPORTANT: Provides recommendations and human workflows; does not simulate physical automated repairs.
    """

    @staticmethod
    def get_response_profile(
        anomaly_type: str,
        parameter: str,
        station_id: str,
        station_name: str,
        observed_val: float,
        expected_val: float,
        nearby_ref_val: float,
        confidence_pct: int,
        is_extreme_weather: bool = False
    ) -> Dict[str, Any]:
        """
        Return deterministic decision-support package tailored to anomaly type and severity.
        """
        deviation = observed_val - expected_val
        unit = "°C" if parameter == "Temperature" else ("%" if parameter == "Humidity" else "hPa")

        # 1. Genuine Regional Extreme Weather
        if is_extreme_weather:
            return {
                "anomaly_title": "Genuine Regional Extreme Weather Pattern",
                "severity": "HIGH",
                "severity_class": "extreme",
                "severity_desc": "Severe Meteorological Event · Multi-Station Consensus Verified",
                "detection_engine": "Spatial Consensus Check (4 Peer Nodes Confirmed)",
                "next_action": f"DO NOT isolate {parameter.lower()} sensor. Retain telemetry in NWP models and broadcast IMD weather advisory.",
                "why_this_matters": (
                    f"Observed {parameter.lower()} of {observed_val:.1f} {unit} exceeds climatological thresholds, "
                    f"but surrounding regional AWS stations also confirm matching conditions ({nearby_ref_val:.1f} {unit}). "
                    f"This is a verified natural weather hazard, not an isolated sensor fault."
                ),
                "why_this_action": (
                    "Quarantining sensors during severe weather blind-spots numerical weather prediction (NWP) "
                    "and emergency disaster response models. Data must be preserved and disseminated to MoES/IMD forecasters."
                ),
                "maintenance_type": "No Sensor Maintenance Required (Hardware Verified Operational)",
                "operational_advice": "Maintain active telemetry ingestion. Forward readings to regional forecast office.",
                "checklist": [
                    "Confirm spatial consensus match across surrounding regional AWS nodes",
                    "Verify Doppler weather radar and INSAT-3DR satellite cloud imagery",
                    "Maintain sensor in active NWP data assimilation pipeline",
                    "Notify IMD Regional Meteorological Forecasting Center",
                    "Broadcast public meteorological advisory (Heatwave / Squall warning)"
                ],
                "recommended_action_summary": "Retain sensor in active ingestion; broadcast meteorological hazard alert."
            }

        # 2. Sensor Spike / Sudden Step Change
        if "Spike" in anomaly_type or "ADC Surge" in anomaly_type:
            return {
                "anomaly_title": f"{parameter} Spike (Step-Change Outlier)",
                "severity": "HIGH",
                "severity_class": "critical",
                "severity_desc": "Urgent Operator Verification Required · Rapid Anomaly",
                "detection_engine": "Step Check — Tier 1 & Isolation Forest",
                "next_action": f"Verify {station_name} sensor reading and inspect {parameter.lower()} sensor wiring and environment.",
                "why_this_matters": (
                    f"Observation ({observed_val:.1f} {unit}) jumped {deviation:+.1f} {unit} within a 2-minute cycle, "
                    f"while nearby AWS stations report nominal conditions (~{nearby_ref_val:.1f} {unit})."
                ),
                "why_this_action": (
                    "A sudden step-change typically indicates an analog-to-digital converter surge, loose harness terminal, "
                    "or sensor shield degradation. Verification is required before returning the sensor to operational models."
                ),
                "maintenance_type": f"{parameter} Probe Inspection & Wiring Continuity Check",
                "operational_advice": "Recommend data stream isolation pending on-site sensor inspection.",
                "checklist": [
                    "Verify reading against on-site station logger console",
                    "Compare observation against surrounding regional AWS nodes",
                    "Inspect probe wiring harness, connector pins, and terminal block",
                    "Inspect radiation shield for debris, obstruction, or physical damage",
                    "If abnormal reading persists, recommend sensor data isolation",
                    "Schedule on-site sensor verification or replacement"
                ],
                "recommended_action_summary": f"Verify sensor reading, inspect {parameter.lower()} wiring, and isolate data if abnormal."
            }

        # 3. Frozen Sensor / Persistence
        if "Frozen" in anomaly_type or "Flatline" in anomaly_type:
            return {
                "anomaly_title": f"Frozen {parameter} Sensor (Persistence Failure)",
                "severity": "HIGH",
                "severity_class": "critical",
                "severity_desc": "High Severity · Sensor Output Stagnated",
                "detection_engine": "Persistence Check — Tier 1 Variance Test (σ < 0.005)",
                "next_action": f"Check {station_name} sensor communication, power supply, and probe wiring.",
                "why_this_matters": (
                    f"Sensor reading has remained completely static ({observed_val:.1f} {unit}) across consecutive observation intervals "
                    f"without natural atmospheric micro-fluctuations."
                ),
                "why_this_action": (
                    "A constant flatlined value across repeated cycles indicates an ADC hang, RS-485 bus freeze, "
                    "or data logger communication breakdown rather than true atmospheric stagnation."
                ),
                "maintenance_type": "Interface Module Diagnostic & Power Cycle / Bus Reset",
                "operational_advice": "Recommend flagging sensor output as frozen; test communications before returning to online status.",
                "checklist": [
                    "Verify whether reading remains unchanged across next observation cycle",
                    "Check RS-485 / Modbus serial communication loop with AWS data logger",
                    "Inspect DC power supply voltage rails to the sensor interface",
                    "Check wiring continuity for intermittent breaks or pin oxidation",
                    "Perform remote soft reboot / power cycle of sensor interface module",
                    "If persistence continues, schedule physical module inspection/replacement"
                ],
                "recommended_action_summary": "Check sensor communication, verify power supply rails, and reboot interface module."
            }

        # 4. Gradual Drift
        if "Drift" in anomaly_type:
            return {
                "anomaly_title": f"Gradual {parameter} Sensor Drift",
                "severity": "MEDIUM",
                "severity_class": "warning",
                "severity_desc": "Medium Severity · Calibration Degradation Detected",
                "detection_engine": "Temporal Trend Analysis & Spatial Residual Drift",
                "next_action": f"Verify {station_name} calibration against reference standard and compare against historical baseline.",
                "why_this_matters": (
                    f"Sensor has progressively drifted by {deviation:+.1f} {unit} away from baseline over recent cycles, "
                    f"diverging systematically from regional peers ({nearby_ref_val:.1f} {unit})."
                ),
                "why_this_action": (
                    "Gradual deviation over time indicates sensor transducer aging, optical/membrane fouling, "
                    "or calibration drift. Timely verification prevents slow corruption of climatological datasets."
                ),
                "maintenance_type": f"Zero/Span Recalibration & Sensor Cleaning",
                "operational_advice": "Continue monitoring with caution; apply software bias offset pending field calibration.",
                "checklist": [
                    "Compare observed trend against 7-day historical station baseline",
                    "Cross-reference with neighboring AWS nodes to check regional gradient",
                    "Inspect sensor head for dust accumulation, chemical fouling, or aging",
                    "Perform two-point calibration check using portable field standard",
                    "Apply temporary software bias correction offset if drift is linear",
                    "Schedule sensor recalibration or recalibrated spare replacement"
                ],
                "recommended_action_summary": "Compare against historical baseline and schedule field calibration verification."
            }

        # 5. Physical Range Violation
        if "Physical" in anomaly_type or "Range" in anomaly_type or "Out-of-Range" in anomaly_type or "Physics" in anomaly_type or "Thermodynamic" in anomaly_type:
            return {
                "anomaly_title": f"Physical Range Violation ({parameter} Out-of-Bounds)",
                "severity": "CRITICAL",
                "severity_class": "critical",
                "severity_desc": "Critical Severity · Thermodynamic / Climatological Limit Breached",
                "detection_engine": "Physical Range & Thermodynamic Dew-Point Constraints",
                "next_action": f"Treat {station_name} reading as potentially invalid, recommend data isolation, and inspect sensor hardware.",
                "why_this_matters": (
                    f"Observed value ({observed_val:.1f} {unit}) breaches physical climatological bounds or thermodynamic limits "
                    f"for the station's geographic elevation."
                ),
                "why_this_action": (
                    "Readings violating thermodynamic laws (e.g., dew point exceeding ambient temperature) or physical limits "
                    "indicate electronic component failure, water ingress, or power rail short."
                ),
                "maintenance_type": "Hardware Electronics & Surge Protector Diagnostic",
                "operational_advice": "Immediate data isolation recommended; physical sensor inspection required.",
                "checklist": [
                    "Treat reading as potentially invalid for weather forecasting",
                    "Verify sensor output analog voltage against manufacturer specifications",
                    "Inspect data cable for moisture ingress, pinching, or lightning surge",
                    "Cross-check psychrometric dew-point consistency with relative humidity",
                    "Recommend isolating reading from downstream operational ingestion",
                    "Inspect or replace sensor transducer if hardware fault persists"
                ],
                "recommended_action_summary": "Treat reading as invalid, recommend data isolation, and inspect sensor hardware."
            }

        # 6. Default / General Anomaly
        return {
            "anomaly_title": f"Unusual {parameter} Telemetry Cluster",
            "severity": "MEDIUM",
            "severity_class": "warning",
            "severity_desc": "Medium Severity · Multi-Variate Anomaly Flagged",
            "detection_engine": "Isolation Forest Unsupervised Detection",
            "next_action": f"Verify {station_name} observation against neighboring AWS nodes.",
            "why_this_matters": "Multi-variate statistical correlation indicates atypical relationship among weather variables.",
            "why_this_action": "Statistically unusual observations should be cross-verified before assimilation.",
            "maintenance_type": "Telemetry Monitoring & Cross-Validation",
            "operational_advice": "Operator inspection recommended; compare with nearby AWS.",
            "checklist": [
                "Verify reading on telemetry dashboard",
                "Compare with nearby AWS station records",
                "Inspect sensor housing and connection",
                "Monitor next observation cycle for persistence"
            ],
            "recommended_action_summary": "Monitor reading and verify against neighboring stations."
        }
