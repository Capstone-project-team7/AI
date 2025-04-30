import torch
import torch.nn as nn

# ✅ 모델 구조 정의 (LSTMPoseClassifier)
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

# ✅ 모델 로드 함수
def load_model(pth_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMPoseClassifier()
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ✅ 추론 함수
def predict(model, keypoints_sequence):
    """
    keypoints_sequence: numpy array or torch tensor
        (sequence_length, 34) 형태여야 함
    """
    if isinstance(keypoints_sequence, np.ndarray):
        keypoints_sequence = torch.tensor(keypoints_sequence, dtype=torch.float32)

    keypoints_sequence = keypoints_sequence.unsqueeze(0)  # (1, seq_len, 34)
    device = next(model.parameters()).device
    keypoints_sequence = keypoints_sequence.to(device)

    with torch.no_grad():
        output = model(keypoints_sequence)
        pred = torch.argmax(output, dim=1).item()

    return pred  # 0 (정상) or 1 (이상행동)

# ✅ 예시 실행 코드
if __name__ == "__main__":
    import numpy as np

    # 모델 로드
    model_path = "model_checkpoint_20250416_143900.pth"  
    model = load_model(model_path)

    # 테스트용 더미 데이터 (sequence_length=30, keypoints 34개)
    dummy_input = np.random.rand(30, 34)

    # 추론
    result = predict(model, dummy_input)

    print("예측 결과:", "정상 (0)" if result == 0 else "이상행동 (1)")
