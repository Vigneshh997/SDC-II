import requests
import json

test_cases = [
    {
        'name': 'Case 1: Acute Lobar Pneumonia',
        'img': 'frontend/public/sample_pneumonia.jpg',
        'note': 'Patient presents with acute 4-day fever (39.4°C), productive cough with rust-colored sputum, and right-sided pleuritic chest pain. Airspace opacities suspected.',
        'meta': {'age': 62, 'sex': 'male', 'temperature': 39.4, 'heart_rate': 110, 'resp_rate': 28, 'spo2': 89}
    },
    {
        'name': 'Case 2: Pneumonia + Effusion',
        'img': 'frontend/public/sample_effusion_pneumonia.png',
        'note': 'Worsening shortness of breath, right lower zone dullness to percussion, high-grade fever, and productive purulent cough.',
        'meta': {'age': 60, 'sex': 'male', 'temperature': 39.1, 'heart_rate': 105, 'resp_rate': 26, 'spo2': 90}
    },
    {
        'name': 'Case 3: Consolidation & Edema',
        'img': 'frontend/public/sample_consolidation.png',
        'note': 'Bilateral patchy airspace opacities, progressive dyspnea on exertion, and mild bilateral lower extremity edema.',
        'meta': {'age': 55, 'sex': 'female', 'temperature': 38.6, 'heart_rate': 98, 'resp_rate': 24, 'spo2': 91}
    },
    {
        'name': 'Case 4: Normal / Clear Screening',
        'img': 'frontend/public/sample_normal.png',
        'note': 'Routine pre-operative health checkup. Patient is asymptomatic and afebrile with clear bilateral breath sounds and normal chest expansion.',
        'meta': {'age': 45, 'sex': 'female', 'temperature': 36.8, 'heart_rate': 72, 'resp_rate': 16, 'spo2': 99}
    }
]

url = 'http://127.0.0.1:8000/predict'
for tc in test_cases:
    with open(tc['img'], 'rb') as f:
        files = {'image': f}
        data = {'clinical_note': tc['note'], 'metadata': json.dumps(tc['meta'])}
        r = requests.post(url, files=files, data=data, timeout=30)
        res = r.json()
        print(f"\n=== {tc['name']} ===")
        print("Diagnosis:", res.get("diagnosis"))
        print("Has Findings:", res.get("has_findings"))
        preds = list(res.get("predictions", {}).items())[:4]
        print("Top 4 Predictions:", preds)
