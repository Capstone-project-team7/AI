import requests
from datetime import datetime

SPRING_BOOT_URL = "http://localhost:8080/api/anomaly/notify"

def build_anomaly_metadata(video_url: str, anomaly_type: str, timestamp: datetime, user_id: int) -> dict:
    return {
        "videoUrl": video_url,
        "anomalyType": anomaly_type,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
        "userId": user_id
    }

def post_anomaly_metadata(metadata: dict):
    try:
        response = requests.post(SPRING_BOOT_URL, json=metadata)
        response.raise_for_status()
        print(f"✅ POST 성공: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"❌ POST 실패: {e}")
