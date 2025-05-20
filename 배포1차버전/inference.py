import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from model import LSTMPoseClassifier  # 모델 정의가 들어있는 파일로부터 import
import argparse

CLASS_NAMES = [
    "Fall", "Damage", "Fire", "Smoke", "Abandon", "Theft", "Assault", "Normal"
]


# 📌 추론 함수
def run_inference(video_path, model_path, sequence_length=45):
    # 1. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = LSTMPoseClassifier(input_size=34, hidden_size=128, num_layers=2, num_classes=8)
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.to(device).eval()

    # 2. YOLO 로드
    pose_model = YOLO("yolo11s-pose.pt")

    # 3. 비디오 열기
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frames = []
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # YOLO 추론
        result = pose_model(frame)[0]
        keypoints = result.keypoints
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0].cpu().numpy().flatten()
            keypoints_list.append(xy if len(xy) == 34 else np.zeros(34))
        else:
            keypoints_list.append(np.zeros(34))

    cap.release()
    keypoints_array = np.array(keypoints_list)

    # 4. 시퀀스 단위로 추론
    for i in range(0, len(keypoints_array) - sequence_length + 1):
        seq = keypoints_array[i:i + sequence_length]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = lstm_model(seq_tensor)
            pred_class = pred.argmax(dim=1).item()
        results.append(pred_class)

    # 5. 결과 요약 출력
    from collections import Counter
    counter = Counter(results)
    print("\n🎯 추론 결과 (시퀀스 단위)")
    for class_idx, count in counter.most_common():
        print(f"  - {CLASS_NAMES[class_idx]}: {count}회")

    # 6. 프레임별로 클래스 이름 출력 (선택)
    print("\n🎬 예측된 클래스 시퀀스:")
    print([CLASS_NAMES[r] for r in results])

# 📌 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="예측할 영상 경로 (.mp4)")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로 (.pth)")
    args = parser.parse_args()

    run_inference(args.video, args.model)
