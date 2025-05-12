import torch
import numpy as np
import cv2
import os
import collections
import logging
import time
from ultralytics import YOLO
import torch.nn.functional as F
from model import LSTMPoseClassifier

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "Fall", "Damage", "Fire", "Smoke", "Abandon", "Theft", "Assault", "Normal"
]

_pose_model = None
_lstm_classifier = None

def load_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo11n-pose.pt')
            _pose_model = YOLO(yolo_path)
            logger.info("YOLO11 모델 로드됨")
        except Exception as e:
            logger.error(f"YOLOv8 모델 로드 실패: {e}")
            return None
    return _pose_model

def load_classifier_model(device=None):
    global _lstm_classifier
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _lstm_classifier is not None:
        return _lstm_classifier, device
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo11_buy_best_model.pth')
        model = LSTMPoseClassifier(input_size=34, hidden_size=128, num_layers=2, num_classes=8)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        _lstm_classifier = model
        return model, device
    except Exception as e:
        logger.error(f"LSTM 모델 로드 실패: {e}")
        return None, device

def extract_keypoints_from_frame(frame):
    model = load_pose_model()
    if model is None or frame is None:
        return None
    try:
        results = model(frame)
        if len(results[0].keypoints) == 0:
            return None
        keypoints = results[0].keypoints.xy[0].cpu().numpy().flatten()
        if keypoints.shape != (34,):
            return None
        h, w = frame.shape[:2]
        for i in range(0, 34, 2):
            keypoints[i] /= w
            keypoints[i+1] /= h
        return keypoints
    except Exception as e:
        logger.error(f"키포인트 추출 오류: {e}")
        return None

def prepare_keypoints_sequence(buffer, min_seq_length=45):
    valid = [k for k in buffer if k is not None and isinstance(k, np.ndarray) and k.shape == (34,)]
    if len(valid) < min_seq_length * 0.6:
        return None
    if len(valid) > min_seq_length:
        valid = valid[-min_seq_length:]
    return np.array(valid)

# === 연속 이상 감지용 상태 변수 ===
anomaly_counter = 0
ANOMALY_STREAK_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.6

def classify_behavior(sequence, threshold=CONFIDENCE_THRESHOLD):
    if sequence is None:
        return False, 0.0, None

    model, device = load_classifier_model()
    if model is None:
        return False, 0.0, None

    try:
        x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        max_prob, idx = torch.max(probs, dim=1)
        max_prob, idx = max_prob.item(), idx.item()
        behavior = CLASS_NAMES[idx]
        is_anomaly = behavior != "Normal"

        if max_prob >= threshold:
            return is_anomaly, max_prob, behavior
        else:
            return False, max_prob, None
    except Exception as e:
        logger.error(f"분류 오류: {e}")
        return False, 0.0, None

def advanced_theft_detection_model(frame, buffer):
    global anomaly_counter

    keypoints = extract_keypoints_from_frame(frame)
    if keypoints is not None:
        buffer.append(keypoints)
    elif buffer and np.random.rand() < 0.3:
        buffer.append(next((k for k in reversed(buffer) if k is not None), None))

    if len(buffer) >= 45:
        seq = prepare_keypoints_sequence(buffer)
        is_anomaly, conf, behavior = classify_behavior(seq)

        if is_anomaly:
            anomaly_counter += 1
            if anomaly_counter >= ANOMALY_STREAK_THRESHOLD:
                return True, conf, behavior
        else:
            anomaly_counter = 0

        return False, conf, behavior
    return False, 0.0, None

def theft_detection_model(frame, buffer):
    is_anomaly, conf, behavior = advanced_theft_detection_model(frame, buffer)

    if behavior is None:
        return False, 0.0, None

    if is_anomaly:
        msg = {
            "Theft": "절도 행위 감지",
            "Assault": "폭행 행위 감지",
            "Damage": "기물 파손 감지",
            "Fall": "넘어짐 감지",
            "Fire": "화재 감지",
            "Smoke": "흡연 감지",
            "Abandon": "유기 감지",
        }.get(behavior, f"{behavior} 감지")
        return True, conf, msg

    return False, conf, "정상 행동 감지됨"