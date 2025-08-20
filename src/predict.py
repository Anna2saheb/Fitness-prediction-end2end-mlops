import requests

url = "http://127.0.0.1:8080/predict"

payload = {
    "age": 30,
    "height_cm": 158,
    "weight_kg": 72,
    "heart_rate": 70,
    "blood_pressure": 120,
    "sleep_hours": 7,
    "nutrition_quality": 5,
    "activity_index": 5,
    "smokes": 0,
    "gender_M": 1
}

res = requests.post(url, json=payload)
print(res.json())
