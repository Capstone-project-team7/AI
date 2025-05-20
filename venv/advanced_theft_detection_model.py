import torch
import numpy as np
import cv2
import os
import collections
import logging
import time
from ultralytics import YOLO
import torch.nn.functional as F
import sys

# 프로젝트 AI 모듈 경로 추가 
current_dir = os.path.dirname(os.path.abspath(__file__))

# 새로 개발된 모델 임포트 
from model import LSTMPoseClassifier

logger = logging.getLogger(__name__)

# 클래스 이름 정의
CLASS_NAMES = [
    "Fall", "Damage", "Fire", "Smoke", "Abandon", "Theft", "Assault"
]

# === 모델 로드 함수 ===

# YOLOv8 포즈 모델 로드 (싱글톤)
_pose_model = None
def load_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            # 현재 디렉토리에서 YOLOv8 모델 찾기
            yolo_path = os.path.join(current_dir, 'yolov8n-pose.pt')
            if not os.path.exists(yolo_path):
                logger.warning(f"YOLOv8 모델 파일을 찾을 수 없음: {yolo_path}")
                yolo_path = 'yolov8n-pose.pt'  # 기본 경로 시도
                
            _pose_model = YOLO(yolo_path)
            logger.info(f"YOLOv8 포즈 모델이 로드되었습니다: {yolo_path}")
        except Exception as e:
            logger.error(f"YOLOv8 포즈 모델 로드 실패: {e}")
            return None
    return _pose_model

# LSTM 분류 모델 로드
_lstm_classifier = None
def load_classifier_model(device=None):
    global _lstm_classifier
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 이미 로드되었는지 확인
    if _lstm_classifier is not None:
        return _lstm_classifier, device
    
    try:
        # 모델 경로 설정 (현재 디렉토리에서 모델 파일 찾기)
        model_path = os.path.join(current_dir, 'final_model_20250504_033924.pth')
        if not os.path.exists(model_path):
            logger.error(f"모델 파일을 찾을 수 없음: {model_path}")
            return None, device
            
        # 모델 인스턴스 생성
        _lstm_classifier = LSTMPoseClassifier(input_size=34, hidden_size=128, num_layers=2, num_classes=7)
        
        # 모델 가중치 로드
        _lstm_classifier.load_state_dict(torch.load(model_path, map_location=device))
        _lstm_classifier = _lstm_classifier.to(device)
        _lstm_classifier.eval()  # 평가 모드로 설정
        
        logger.info(f"LSTM 분류 모델이 {device}에 로드되었습니다: {model_path}")
        return _lstm_classifier, device
    
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None, device

# === 키포인트 처리 함수 ===

def extract_keypoints_from_frame(frame):
    """프레임에서 YOLOv8을 사용하여 키포인트 추출"""
    if frame is None:
        return None
    
    # YOLOv8 모델 로드 확인
    pose_model = load_pose_model()
    if pose_model is None:
        return None
    
    try:
        # YOLOv8로 관절점 탐지
        results = pose_model(frame)
        
        # 사람이 탐지되지 않은 경우
        if len(results[0].keypoints) == 0:
            logger.debug("프레임에서 사람이 감지되지 않았습니다.")
            return None
            
        # 첫번째 사람의 관절점 (17개 관절점, x, y 좌표)
        keypoints = results[0].keypoints.xy[0].cpu().numpy().flatten()
        
        # 관절점이 제대로 추출되었는지 검증
        if keypoints.shape != (34,):
            logger.warning(f"추출된 키포인트의 형태가 비정상: {keypoints.shape}, 이 프레임은 건너뜁니다.")
            return None
            
        # 정규화 - 프레임 크기로 나누어 0-1 범위로
        height, width = frame.shape[:2]
        for i in range(0, len(keypoints), 2):
            keypoints[i] /= width      # x 좌표
            keypoints[i+1] /= height   # y 좌표
            
        return keypoints
        
    except Exception as e:
        logger.error(f"키포인트 추출 중 오류: {e}")
        return None

# 키포인트 시퀀스 정규화 및 준비
def prepare_keypoints_sequence(keypoints_buffer, min_seq_length=45):
    """키포인트 버퍼를 처리하여 모델 입력용 시퀀스로 준비"""
    if not keypoints_buffer or len(keypoints_buffer) < min_seq_length:
        logger.debug(f"키포인트 시퀀스가 너무 짧음: {len(keypoints_buffer) if keypoints_buffer else 0} < {min_seq_length}")
        return None
    
    # 유효한 키포인트만 선택 (None 값 제외)
    valid_keypoints = [kp for kp in keypoints_buffer if kp is not None and isinstance(kp, np.ndarray) and kp.shape == (34,)]
    
    # 유효한 키포인트가 충분한지 확인 (최소 60% 이상)
    if len(valid_keypoints) < len(keypoints_buffer) * 0.6:
        logger.warning(f"유효한 키포인트 비율이 낮음: {len(valid_keypoints)}/{len(keypoints_buffer)}")
        return None
    
    # 시퀀스 길이 조정 (모델 입력 길이에 맞게)
    if len(valid_keypoints) > min_seq_length:
        # 가장 최근 프레임 기준으로 시퀀스 자르기
        valid_keypoints = valid_keypoints[-min_seq_length:]
    
    # numpy 배열로 변환
    sequence_array = np.array(valid_keypoints)
    
    return sequence_array

# === 행동 분류 함수 ===

def classify_behavior(keypoints_sequence, threshold=0.1):
    """LSTM 분류 모델을 사용하여 행동 유형 분류"""
    if keypoints_sequence is None:
        return False, 0.0, None
    
    # 모델 및 장치 로드
    model, device = load_classifier_model()
    if model is None:
        return False, 0.0, None
    
    with torch.no_grad():
        try:
            # 입력 텐서 준비 [1, sequence_length, feature_dim]
            input_tensor = torch.FloatTensor(keypoints_sequence).unsqueeze(0).to(device)
            
            # 모델 예측
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
            # 가장 높은 확률의 클래스 및 확률
            max_prob, predicted_idx = torch.max(probs, dim=1)
            max_prob = max_prob.item()
            predicted_idx = predicted_idx.item()
            
            # 모든 클래스 확률 로깅 (디버깅용)
            class_probs = {CLASS_NAMES[i]: probs[0][i].item() for i in range(len(CLASS_NAMES))}
            logger.debug(f"클래스별 확률: {class_probs}")
            
            # 임계값 기반 이상 행동 감지
            if max_prob >= threshold:
                behavior_type = CLASS_NAMES[predicted_idx]
                logger.info(f"행동 분류: {behavior_type}, 확률: {max_prob:.4f}")
                
                # 절도 또는 폭행인 경우 이상 행동으로 간주
                is_anomaly = behavior_type in ["Theft", "Assault", "Damage"]
                return is_anomaly, max_prob, behavior_type
            else:
                logger.debug(f"낮은 신뢰도로 행동 분류 무시: {CLASS_NAMES[predicted_idx]}, 확률: {max_prob:.4f}")
                return False, max_prob, None
                
        except Exception as e:
            logger.error(f"행동 분류 중 오류: {e}")
            return False, 0.0, None

# === 메인 감지 함수 ===

def advanced_theft_detection_model(frame, keypoints_buffer):
    """
    프레임에서 이상 행동을 감지하는 통합 함수 (새로운 AI 모델 사용)
    
    Args:
        frame: 현재 비디오 프레임
        keypoints_buffer: 최근 키포인트 시퀀스를 저장하는 버퍼 (collections.deque)
        
    Returns:
        tuple: (이상 감지 여부, 신뢰도 점수, 이상 행동 유형 문자열)
    """
    start_time = time.time()
    
    # 현재 프레임에서 키포인트 추출
    current_keypoints = extract_keypoints_from_frame(frame)
    
    # 키포인트가 추출되지 않은 경우 처리
    if current_keypoints is not None:
        # 키포인트 시퀀스 버퍼에 추가
        keypoints_buffer.append(current_keypoints)
    else:
        # 연속성을 위해 이전 프레임의 키포인트 보간 (30% 확률)
        if keypoints_buffer and len(keypoints_buffer) > 0:
            if np.random.random() < 0.3:
                recent_valid_keypoints = next((kp for kp in reversed(keypoints_buffer) if kp is not None), None)
                if recent_valid_keypoints is not None:
                    logger.debug("이전 프레임의 키포인트로 보간합니다.")
                    keypoints_buffer.append(recent_valid_keypoints)
        return False, 0.0, None
        
    # 버퍼가 충분히 채워졌을 때만 행동 분류 수행 (최소 45프레임)
    if len(keypoints_buffer) >= 45:
        # 키포인트 시퀀스 준비
        sequence_array = prepare_keypoints_sequence(keypoints_buffer)
        if sequence_array is None:
            return False, 0.0, None
            
        # 행동 분류 (threshold 0.1로 설정)
        is_anomaly, confidence, behavior_type = classify_behavior(sequence_array, threshold=0.1)
        
        # 처리 시간 로깅
        process_time = time.time() - start_time
        logger.debug(f"행동 감지 처리 시간: {process_time:.4f}초")
        
        if is_anomaly:
            return True, confidence, behavior_type
    
    # 이상이 감지되지 않은 경우
    return False, 0.0, None 

# 호환성을 위한 래퍼 함수 (기존 theft_detection_model과 동일한 인터페이스 유지)
def theft_detection_model(frame, keypoints_buffer):
    """
    기존 theft_detection_model과 호환되는 인터페이스 제공
    """
    is_anomaly, confidence, behavior_type = advanced_theft_detection_model(frame, keypoints_buffer)
    
    # 행동 유형에 따른 메시지 조정
    if is_anomaly:
        if behavior_type == "Theft":
            message = "절도 행위 감지"
        elif behavior_type == "Assault":
            message = "폭행 행위 감지"
        elif behavior_type == "Damage":
            message = "기물 파손 감지"
        else:
            message = f"{behavior_type} 행동 감지"
        return True, confidence, message
    
    return False, 0.0, None 