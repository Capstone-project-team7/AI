# ✅ 기존 코드에서 Dropout 추가 + threshold 자동 비교 코드

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 📌 LSTM 기반 Pose Classifier + Dropout
class LSTMPoseClassifier(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# 📌 데이터셋 클래스
class AbnormalPoseDataset(Dataset):
    def __init__(self, root_dir, sequence_length=30, stride=5, threshold=0.3):
        self.sequence_length = sequence_length
        self.stride = stride
        self.threshold = threshold
        self.model = YOLO("yolov8n-pose.pt")
        self.video_dir = os.path.join(root_dir, "01.원천데이터", "TS_03.이상행동_07.전도")
        self.label_dir = os.path.join(root_dir, "02.라벨링데이터", "TL_03.이상행동_07.전도")
        self.samples = self._make_dataset()

    def _parse_fall_range(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        start_frame, end_frame = -1, -1
        for track in root.findall("track"):
            label = track.attrib.get("label", "")
            if label == "fall_start":
                for box in track.findall("box"):
                    start_frame = int(box.attrib["frame"])
            elif label == "fall_end":
                for box in track.findall("box"):
                    end_frame = int(box.attrib["frame"])
        return start_frame, end_frame

    def _make_dataset(self):
        samples = []
        filenames = sorted(f for f in os.listdir(self.video_dir) if f.endswith(".mp4"))
        for file in filenames:
            video_path = os.path.join(self.video_dir, file)
            label_path = os.path.join(self.label_dir, file.replace(".mp4", ".xml"))
            if not os.path.exists(label_path):
                continue
            fall_start, fall_end = self._parse_fall_range(label_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            keypoints_list = []
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                result = self.model(frame, verbose=False)[0]
                keypoints = result.keypoints
                if keypoints is not None and len(keypoints.xy) > 0:
                    keypoint_xy = keypoints.xy[0].cpu().numpy().flatten()
                    keypoints_list.append(keypoint_xy if len(keypoint_xy) == 34 else np.zeros(34))
                else:
                    keypoints_list.append(np.zeros(34))
            cap.release()

            keypoints_array = np.array(keypoints_list)
            for i in range(0, len(keypoints_array) - self.sequence_length, self.stride):
                seq = keypoints_array[i:i + self.sequence_length]
                overlap = max(0, min(fall_end, i + self.sequence_length) - max(fall_start, i))
                abnormal_ratio = overlap / self.sequence_length
                label = 1 if abnormal_ratio > self.threshold else 0
                samples.append((torch.tensor(seq, dtype=torch.float32), label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# 📌 threshold 자동 실험
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
root_dir = "/Users/seunggi/Desktop/Capstone/local_test"

results = []
for thresh in thresholds:
    print(f"\n🚀 Testing threshold={thresh}")
    dataset = AbnormalPoseDataset(root_dir, sequence_length=30, stride=5, threshold=thresh)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = LSTMPoseClassifier(dropout=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_preds, all_labels = [], []

    for epoch in range(5):
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            class_weights = torch.tensor([1.0, 3.0]).to(device)
            loss = F.cross_entropy(pred, y, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_preds += pred.argmax(dim=1).cpu().tolist()
            all_labels += y.cpu().tolist()

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    print(f"Threshold {thresh:.1f} | Acc: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    results.append({"threshold": thresh, "accuracy": acc, "precision": precision, "recall": recall})

# 📌 결과 저장
pd.DataFrame(results).to_csv("threshold_comparison.csv", index=False)
print("\n✅ 모든 threshold 실험 완료! 결과 저장됨: threshold_comparison.csv")
