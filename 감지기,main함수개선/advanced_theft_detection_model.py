import os
import time
import logging
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from model import LSTMPoseClassifier

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 전역 상수
current_dir = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH    = os.path.join(current_dir, 'yolo11s-pose.pt')
CLASSIFIER_PATH    = os.path.join(current_dir, 'yolo11s_model.pth')
# 원래 클래스 리스트로 복원 (모델 가중치와 일치시키기 위해)
CLASS_NAMES        = ["Fall", "Damage", "Fire", "Smoke", "Abandon", "Theft", "Assault", "Normal"]

# 무시할 클래스 (이 클래스들은 모델에 필요하지만 탐지 결과에서는 제외)
IGNORE_CLASSES = {"Abandon", "Smoke"}

# 클래스별 임계값 설정 
THRESHOLDS = {
    'Fall':    0.8,
    'Damage':  0.5,
    'Fire':    1.0,
    'Smoke':   1.0,  # 
    'Abandon': 1.0,  # 
    'Theft':   0.5,
    'Assault': 0.85,
}

DEFAULT_THRESHOLD = 0.5  # mapping에 없는 클래스용 기본값

# 싱글톤 패턴으로 모델 캐싱
enabled_yolo     = None
classifier       = None
classifier_device = None

def load_pose_model():
    global enabled_yolo
    if enabled_yolo is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error(f"YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL_PATH}")
            return None
        enabled_yolo = YOLO(YOLO_MODEL_PATH)
        logger.info(f"YOLO 포즈 모델 로드: {YOLO_MODEL_PATH}")
    return enabled_yolo

def load_classifier():
    global classifier, classifier_device
    if classifier is not None:
        return classifier, classifier_device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(CLASSIFIER_PATH):
        logger.error(f"분류기 가중치를 찾을 수 없습니다: {CLASSIFIER_PATH}")
        return None, device

    classifier = LSTMPoseClassifier(input_size=34,
                                    hidden_size=128,
                                    num_layers=2,
                                    num_classes=len(CLASS_NAMES))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier = classifier.to(device).eval()
    classifier_device = device
    logger.info(f"LSTM 분류기 로드: {CLASSIFIER_PATH} ({device})")
    return classifier, device

def extract_keypoints_from_frame(frame):
    pose = load_pose_model()
    if pose is None:
        return None
    try:
        res = pose(frame)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            return None
        kp = res.keypoints.xy[0].cpu().numpy().flatten()
        h, w = frame.shape[:2]
        kp[0::2] /= w
        kp[1::2] /= h
        if kp.shape[0] != 34:
            return None
        return kp
    except Exception as e:
        logger.error(f"키포인트 추출 오류: {e}")
        return None

def classify_sequence(seq):
    """시퀀스 단위로 행동 분류"""
    model, device = load_classifier()
    if model is None:
        return None, 0.0

    with torch.no_grad():
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)  # [1, L, 34]
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        return idx, conf

def advanced_theft_detection_model(frame, buffer, min_length=45):
    start = time.time()
    kp = extract_keypoints_from_frame(frame)
    if kp is not None:
        buffer.append(kp)
    else:
        if buffer:
            buffer.append(buffer[-1])
        return False, 0.0, None

    if len(buffer) < min_length:
        return False, 0.0, None

    seq = np.array(list(buffer)[-min_length:])
    idx, conf = classify_sequence(seq)
    elapsed = time.time() - start

    behavior = CLASS_NAMES[idx]

    # 무시 대상이면 로그 없이 정상으로 처리
    if behavior in IGNORE_CLASSES:
        return False, 0.0, None

    threshold = THRESHOLDS.get(behavior, DEFAULT_THRESHOLD)
    normal_idx = CLASS_NAMES.index("Normal")
    logger.debug(f"처리시간: {elapsed:.3f}s, 클래스: {behavior}, 신뢰도: {conf:.3f}, 임계값: {threshold:.2f}")

    if idx != normal_idx and conf >= threshold:
        return True, conf, behavior
    return False, conf, None

def theft_detection_model(frame, buffer):
    return advanced_theft_detection_model(frame, buffer)

def get_detection_with_message(frame, buffer, min_length=45):
    """고급 행동 감지 모델을 실행하고 감지된 행동에 맞는 메시지를 반환합니다."""
    is_anomaly, confidence, behavior_type = advanced_theft_detection_model(frame, buffer, min_length)
    
    if is_anomaly:
        # 행동 유형에 따른 메시지 조정
        if behavior_type == "Theft":
            message = "절도"
        elif behavior_type == "Assault":
            message = "폭행"
        elif behavior_type == "Damage":
            message = "파손"
        elif behavior_type == "Fire":
            message = "방화"
        elif behavior_type == "Fall":
            message = "전도"
        # Abandon과 Smoke 관련 코드는 무시 대상에 있어 실행되지 않지만, 코드 완성성을 위해 유지
        elif behavior_type == "Smoke":
            message = "흡연"
        elif behavior_type == "Abandon":
            message = "유기"
        else:
            message = f"{behavior_type} 행동 감지"
        return True, confidence, message
    
    return False, 0.0, None



