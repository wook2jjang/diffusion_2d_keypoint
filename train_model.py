import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt

# Residual Block 정의 (선택 사항)
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.residual(x)

# UNet 기반 모델 정의 (BatchNorm과 Dropout 추가)
class UNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_embed_dim, dropout_prob=0.3):
        super(UNet, self).__init__()
        # 시간 임베딩 레이어
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # 인코더 레이어 with Residual Blocks
        self.encoder1 = ResidualBlock(input_dim + time_embed_dim, hidden_dim, dropout_prob)
        self.encoder2 = ResidualBlock(hidden_dim, hidden_dim * 2, dropout_prob)
        self.encoder3 = ResidualBlock(hidden_dim * 2, hidden_dim * 4, dropout_prob)
        
        # 디코더 레이어 with Residual Blocks
        self.decoder1 = ResidualBlock(hidden_dim * 4, hidden_dim * 2, dropout_prob)
        self.decoder2 = ResidualBlock(hidden_dim * 2, hidden_dim, dropout_prob)
        self.decoder3 = ResidualBlock(hidden_dim, input_dim, dropout_prob)

    def forward(self, x, t):
        """
        x: [batch_size, input_dim]
        t: [batch_size]
        """
        t = t.unsqueeze(-1)  # [batch_size, 1]
        t_embed = self.time_embedding(t)  # [batch_size, time_embed_dim]
        x = torch.cat([x, t_embed], dim=-1)  # [batch_size, input_dim + time_embed_dim]
        z = self.encoder1(x)  # [batch_size, hidden_dim]
        z = self.encoder2(z)  # [batch_size, hidden_dim * 2]
        z = self.encoder3(z)  # [batch_size, hidden_dim * 4]
        z = self.decoder1(z)  # [batch_size, hidden_dim * 2]
        z = self.decoder2(z)  # [batch_size, hidden_dim]
        out = self.decoder3(z)  # [batch_size, input_dim]
        return out

# Custom Dataset 정의
class DiffusionDataset(Dataset):
    def __init__(self, keypoints):
        """
        keypoints: numpy array [num_samples, input_dim]
        """
        self.keypoints = keypoints

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return self.keypoints[idx]

# 손실 함수 정의 (예측된 노이즈와 실제 노이즈 간의 MSE)
def diffusion_loss(model, x, t, noise):
    """
    x: [batch_size, input_dim] (noisy_input)
    t: [batch_size] (time step)
    noise: [batch_size, input_dim] (actual noise)
    """
    pred_noise = model(x, t)  # [batch_size, input_dim]
    loss = nn.MSELoss()(pred_noise, noise)
    return loss

# 키포인트 데이터 전처리 및 정렬 함수
def compute_relative_keypoints(keypoints, reference_point_idx=0):
    """
    키포인트 데이터를 특정 기준점에 대해 상대적 위치로 변환합니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
        reference_point_idx (int): 기준점의 인덱스 (예: 허리 키포인트).
    
    Returns:
        numpy.ndarray: 상대적 위치로 변환된 키포인트 데이터.
    """
    relative_keypoints = keypoints.copy()
    for i in range(len(keypoints)):
        ref_x = keypoints[i, reference_point_idx * 3]
        ref_y = keypoints[i, reference_point_idx * 3 + 1]
        # 기준점의 위치로 모든 키포인트 이동
        relative_keypoints[i, ::3] -= ref_x
        relative_keypoints[i, 1::3] -= ref_y
    return relative_keypoints

def preprocess_keypoints(data, reference_point_idx=0):
    """
    키포인트 데이터를 전처리합니다. Z-Score 정규화 및 상대적 위치로 변환.
    
    Args:
        data (list of dict): 키포인트 데이터 리스트.
        reference_point_idx (int): 기준점의 인덱스 (예: 허리 키포인트).
    
    Returns:
        numpy.ndarray: 전처리된 키포인트 데이터 [num_samples, 78].
        float: x 축 정규화에 사용된 평균.
        float: x 축 정규화에 사용된 표준편차.
        float: y 축 정규화에 사용된 평균.
        float: y 축 정규화에 사용된 표준편차.
    """
    keypoints = []
    for frame in data:
        keypoints.append(frame['keypoints'])
    keypoints = np.array(keypoints)  # [num_samples, 78]
    
    # Reshape to [num_samples * 26, 3] assuming each keypoint has (x, y, c)
    keypoints = keypoints.reshape(-1, 3)  # [num_samples * 26, 3]
    
    # Separate x, y, c
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    c = keypoints[:, 2]
    
    # Z-Score 정규화
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    
    x_normalized = (x - x_mean) / x_std
    y_normalized = (y - y_mean) / y_std
    
    # 재결합
    keypoints[:, 0] = x_normalized
    keypoints[:, 1] = y_normalized
    
    # 원래 형태로 재배열
    keypoints = keypoints.flatten().reshape(-1, 78)  # [num_samples, 78]
    
    # 상대적 키포인트 계산
    aligned_keypoints = compute_relative_keypoints(keypoints, reference_point_idx=reference_point_idx)
    
    return aligned_keypoints, x_mean, x_std, y_mean, y_std

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(input_path, reference_point_idx=0):
    """
    키포인트 데이터를 로드하고 전처리합니다.
    
    Args:
        input_path (str): 입력 JSON 파일 경로.
        reference_point_idx (int): 기준점의 인덱스 (예: 허리 키포인트).
    
    Returns:
        numpy.ndarray: 전처리된 키포인트 데이터 [num_samples, 78].
        dict: 정규화에 사용된 평균과 표준편차.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    aligned_keypoints, x_mean, x_std, y_mean, y_std = preprocess_keypoints(data, reference_point_idx=reference_point_idx)
    
    normalization_params = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std
    }
    
    return aligned_keypoints, normalization_params

# 데이터 증강 함수
def augment_keypoints(keypoints):
    """
    키포인트 데이터에 다양한 증강 기법을 적용합니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
    
    Returns:
        list of numpy.ndarray: 증강된 키포인트 데이터 리스트.
    """
    augmented_data = []
    
    # 회전
    angles = [-15, -10, -5, 5, 10, 15]
    for angle in angles:
        rotated = rotate_keypoints(keypoints, angle)
        augmented_data.append(rotated)
    
    # 스케일링
    scale_factors = [0.9, 1.1]
    for scale in scale_factors:
        scaled = scale_keypoints(keypoints, scale)
        augmented_data.append(scaled)
    
    # 이동
    translations = [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1)]
    for tx, ty in translations:
        translated = translate_keypoints(keypoints, tx, ty)
        augmented_data.append(translated)
    
    # 반전
    mirrored = mirror_keypoints(keypoints)
    augmented_data.append(mirrored)
    
    # 노이즈 추가
    noisy = add_noise(keypoints, noise_level=0.02)
    augmented_data.append(noisy)
    
    return augmented_data

def rotate_keypoints(keypoints, angle):
    """
    키포인트를 특정 각도만큼 회전시킵니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
        angle (float): 회전 각도 (도 단위).
    
    Returns:
        numpy.ndarray: 회전된 키포인트 데이터.
    """
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians),  np.cos(radians)]
    ])
    
    rotated_keypoints = keypoints.copy()
    for i in range(len(keypoints)):
        for j in range(0, 78, 3):
            x, y = keypoints[i, j], keypoints[i, j+1]
            rotated = rotation_matrix @ np.array([x, y])
            rotated_keypoints[i, j] = rotated[0]
            rotated_keypoints[i, j+1] = rotated[1]
    return rotated_keypoints

def scale_keypoints(keypoints, scale_factor):
    """
    키포인트의 크기를 조정합니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
        scale_factor (float): 스케일링 팩터.
    
    Returns:
        numpy.ndarray: 스케일링된 키포인트 데이터.
    """
    scaled_keypoints = keypoints.copy()
    scaled_keypoints[:, ::3] *= scale_factor
    scaled_keypoints[:, 1::3] *= scale_factor
    return scaled_keypoints

def translate_keypoints(keypoints, tx, ty):
    """
    키포인트를 특정 거리만큼 평행 이동시킵니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
        tx (float): x축 이동 거리.
        ty (float): y축 이동 거리.
    
    Returns:
        numpy.ndarray: 이동된 키포인트 데이터.
    """
    translated_keypoints = keypoints.copy()
    translated_keypoints[:, ::3] += tx
    translated_keypoints[:, 1::3] += ty
    return translated_keypoints

def mirror_keypoints(keypoints):
    """
    키포인트를 좌우 반전시킵니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
    
    Returns:
        numpy.ndarray: 반전된 키포인트 데이터.
    """
    mirrored_keypoints = keypoints.copy()
    mirrored_keypoints[:, ::3] *= -1  # x축 반전
    return mirrored_keypoints

def add_noise(keypoints, noise_level=0.01):
    """
    키포인트에 Gaussian 노이즈를 추가합니다.
    
    Args:
        keypoints (numpy.ndarray): [num_samples, 78] 형태의 키포인트 데이터.
        noise_level (float): 노이즈의 표준편차.
    
    Returns:
        numpy.ndarray: 노이즈가 추가된 키포인트 데이터.
    """
    noisy_keypoints = keypoints + np.random.normal(0, noise_level, keypoints.shape)
    return noisy_keypoints

# 데이터 로드 및 전처리
input_json_path = '/home/wook/diff/error_keypoints.json'  # 오류 키포인트 데이터 경로
reference_point_idx = 0  # 허리 키포인트의 인덱스로 설정 (데이터셋에 따라 조정 필요)

# 전처리된 키포인트 데이터 및 정규화 파라미터 로드
aligned_error_keypoints, normalization_params = load_and_preprocess_data(input_json_path, reference_point_idx=reference_point_idx)

# 데이터 증강 (옵션)
# Uncomment the following lines if you want to include augmented data in training
# augmented_keypoints = augment_keypoints(aligned_error_keypoints)
# all_keypoints = np.vstack([aligned_error_keypoints] + augmented_keypoints)
# print(f"Augmented Keypoints shape: {all_keypoints.shape}")

# 현재는 원본 데이터만 사용
all_keypoints = aligned_error_keypoints

# Torch 텐서로 변환
all_keypoints_tensor = torch.tensor(all_keypoints, dtype=torch.float32)

# 데이터 정보 출력
print(f"Keypoints shape: {all_keypoints_tensor.shape}")  # [num_samples, 78]
print(f"Sample keypoints (first 5 samples):\n{all_keypoints_tensor[:5]}")

# 모델 파라미터 설정
input_dim = all_keypoints_tensor.shape[1]  # 78
hidden_dim = 256  # Increased hidden_dim
time_embed_dim = 32  # Increased time_embed_dim
timesteps = 100
dropout_prob = 0.3  # Dropout 확률 설정

# 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(input_dim, hidden_dim, time_embed_dim, dropout_prob=dropout_prob).to(device)

# 가중치 초기화 (Kaiming Normal)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)

# Optimizer 및 Scheduler 설정
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # AdamW 사용
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Cosine Annealing Scheduler

# Dataset 및 DataLoader 생성
dataset = DiffusionDataset(all_keypoints_tensor.numpy())
batch_size = 64
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Forward Diffusion 과정의 알파 누적 곱 계산
alphas = 1 - np.linspace(0.0001, 0.02, timesteps)
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32).to(device)

# 학습 루프
epochs = 1000  # 에포크 수 증가
start_time = time.time()
best_val_loss = float('inf')
best_model_state = copy.deepcopy(model.state_dict())
prev_loss = float('inf')

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch = batch.to(device)  # [batch_size, input_dim]
        batch_size_current = batch.size(0)
        
        # 각 샘플마다 랜덤 타임스텝 샘플링
        t = torch.randint(0, timesteps, (batch_size_current,), device=device).long()  # [batch_size]
        t_float = t.float()  # [batch_size]
        
        # 노이즈 추가
        alpha_t = alphas_cumprod[t].unsqueeze(-1)  # [batch_size, 1]
        sqrt_alpha_t = torch.sqrt(alpha_t)  # [batch_size, 1]
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)  # [batch_size, 1]
        noise = torch.randn_like(batch)  # [batch_size, input_dim]
        noisy_batch = sqrt_alpha_t * batch + sqrt_one_minus_alpha_t * noise  # [batch_size, input_dim]
        
        # 손실 계산
        loss = diffusion_loss(model, noisy_batch, t_float, noise)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
    
    # Scheduler step
    scheduler.step()
    
    # 평균 손실 계산
    avg_loss = total_loss / len(train_dataloader)
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s")
    
    # 검증 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)  # [batch_size, input_dim]
            batch_size_current = batch.size(0)
            t = torch.randint(0, timesteps, (batch_size_current,), device=device).long()  # [batch_size]
            t_float = t.float()  # [batch_size]
            alpha_t = alphas_cumprod[t].unsqueeze(-1)  # [batch_size, 1]
            noise = torch.randn_like(batch)  # [batch_size, input_dim]
            noisy_batch = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise  # [batch_size, input_dim]
            
            # 손실 계산
            loss = diffusion_loss(model, noisy_batch, t_float, noise)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss:.6f}")
    
    # 손실 감소 확인
    if epoch > 0 and avg_loss >= prev_loss:
        print(f"Warning: Loss did not decrease from previous epoch. Previous: {prev_loss:.6f}, Current: {avg_loss:.6f}")
    prev_loss = avg_loss

    # Best 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        print("Best model updated.")
    else:
        print(f"No improvement in validation loss. Current best: {best_val_loss:.6f}")

# 전체 학습 시간 출력
total_time = time.time() - start_time
print(f"모델 학습 완료 - 총 소요 시간: {total_time:.2f}s")

# Best 모델 저장
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), './unet_diffusion_model_best.pth')
print("Best 모델 저장 완료!")
