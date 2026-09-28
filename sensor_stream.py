import random
import time
import os
import requests

API_URL = os.environ.get("SKYGUARD_API_URL", "http://127.0.0.1:8000/api/v1/detect")

print(f"📡 Starting SkyGuard AWS Telemetry Stream Simulator to {API_URL}...")
print("Modes simulated sequentially: Nominal -> Thermal Spike -> Flatline Frozen -> Sensor Drift -> Regional Heatwave")

stations = ["AWS-IND-001", "AWS-IND-002", "AWS-IND-003"]
step = 0
drift_accumulator = 0.0

while True:
    step += 1
    cycle = (step // 6) % 5  # Switch scenario every 6 iterations (18 seconds)

    # 0 = Nominal, 1 = Spike, 2 = Frozen, 3 = Drift, 4 = Regional Heatwave
    for sid in stations:
        if cycle == 0:
            # Nominal
            temp = round(28.0 + random.uniform(-0.8, 0.8), 2)
            hum = round(62.0 + random.uniform(-1.5, 1.5), 2)
            pres = round(1011.0 + random.uniform(-0.4, 0.4), 2)
            mode_desc = "NOMINAL"
        elif cycle == 1:
            # Single-station Thermal Spike on AWS-IND-001
            temp = 58.4 if sid == "AWS-IND-001" else round(28.0 + random.uniform(-0.8, 0.8), 2)
            hum = 60.0
            pres = 1011.0
            mode_desc = "THERMAL SPIKE"
        elif cycle == 2:
            # Flatline frozen reading on AWS-IND-001
            temp = 29.4 if sid == "AWS-IND-001" else round(28.0 + random.uniform(-0.8, 0.8), 2)
            hum = 60.0
            pres = 1011.0
            mode_desc = "FROZEN VALUE"
        elif cycle == 3:
            # Sensor Drift on AWS-IND-001
            drift_accumulator += 0.4
            temp = round(28.0 + drift_accumulator, 2) if sid == "AWS-IND-001" else round(28.0 + random.uniform(-0.8, 0.8), 2)
            hum = 60.0
            pres = 1011.0
            mode_desc = f"SENSOR DRIFT (+{drift_accumulator:.1f}°C)"
        else:
            # Genuine Regional Extreme Weather (Heatwave 48°C across ALL stations)
            temp = round(48.2 + random.uniform(-0.5, 0.5), 2)
            hum = round(20.0 + random.uniform(-1.0, 1.0), 2)
            pres = round(1003.0 + random.uniform(-0.5, 0.5), 2)
            mode_desc = "REGIONAL HEATWAVE (GENUINE EXTREME)"
            drift_accumulator = 0.0

        payload = {
            "station_id": sid,
            "temperature": temp,
            "humidity": hum,
            "pressure": pres,
            "is_extreme_scenario": (cycle == 4)
        }

        try:
            res = requests.post(API_URL, json=payload, timeout=2)
            if res.status_code == 200:
                body = res.json()
                print(f"[{mode_desc}] {sid} -> {temp}°C | QC: {body.get('classification')} | Quarantined: {body.get('is_sensor_fault')}")
            else:
                print(f"[{mode_desc}] {sid} -> HTTP {res.status_code}")
        except Exception as e:
            print(f"[{mode_desc}] {sid} connection error: {e}")

    time.sleep(3)