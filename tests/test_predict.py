import requests
import json

def test_prediction():
    url = "http://127.0.0.1:8080/predict"
    
    test_cases = [
        {
            "name": "Fit Male Example",
            "data": {
                "age": 45,
                "height_cm": 175,
                "weight_kg": 80,
                "heart_rate": 72,
                "blood_pressure": 120,
                "sleep_hours": 7.5,
                "nutrition_quality": 6.5,
                "activity_index": 3.5,
                "smokes": "no",
                "gender": "M"
            },
            "expected_fit": True  # Change based on your expectations
        },
        {
            "name": "Unfit Female Example",
            "data": {
                "age": 60,
                "height_cm": 160,
                "weight_kg": 90,
                "heart_rate": 85,
                "blood_pressure": 140,
                "sleep_hours": 5,
                "nutrition_quality": 4,
                "activity_index": 1.5,
                "smokes": "yes",
                "gender": "F"
            },
            "expected_fit": False
        }
    ]

    for test in test_cases:
        print(f"\nRunning test: {test['name']}")
        print("Sending data:", json.dumps(test['data'], indent=2))
        
        try:
            response = requests.post(url, json=test['data'])
            print("Status Code:", response.status_code)
            
            if response.status_code == 200:
                result = response.json()
                print("Response:", result)
                if result['is_fit'] == test['expected_fit']:
                    print("✅ Test passed")
                else:
                    print(f"❌ Test failed. Expected {test['expected_fit']} but got {result['is_fit']}")
            else:
                print("Error Response:", response.text)
                
        except Exception as e:
            print("Test failed with exception:", str(e))

if __name__ == "__main__":
    test_prediction()