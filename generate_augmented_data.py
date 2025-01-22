# generate_augmented_data.py

import torch
import torch.nn as nn
import numpy as np
import json
import time
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.io as pio

# Plotly 설정 (HTML로 출력)
pio.renderers.default = 'browser'  # 'notebook', 'browser' 등 환경에 맞게 설정

# 1. Residual Block 정의
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

# 2. UNet 기반 모델 정의 (BatchNorm과 Dropout 추가)
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

# 3. Custom Dataset 정의
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

# 4. 손실 함수 정의 (예측된 노이즈와 실제 노이즈 간의 MSE)
def diffusion_loss(model, x, t, noise):
    """
    x: [batch_size, input_dim] (noisy_input)
    t: [batch_size] (time step)
    noise: [batch_size, input_dim] (actual noise)
    """
    pred_noise = model(x, t)  # [batch_size, input_dim]
    loss = nn.MSELoss()(pred_noise, noise)
    return loss

# 5. 키포인트 데이터 전처리 및 정렬 함수
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

# 6. 데이터 증강 함수
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

# 7. 시각화 함수 (Plotly 사용, y축 반전 추가)
def plot_keypoints_plotly(original, augmented=None, skeleton=None, keypoint_labels=None, title="Keypoints Plotly Visualization"):
    """
    Plot keypoints using Plotly for interactive visualization.
    
    Parameters:
    - original: [78] flat array
    - augmented: [78] flat array (optional)
    - skeleton: 연결된 키포인트 인덱스 리스트 (optional)
    - keypoint_labels: 키포인트 이름 리스트 (optional)
    - title: 시각화 제목
    """
    fig = go.Figure()
    
    # 원본 키포인트
    orig_x = original[::3]
    orig_y = original[1::3]
    fig.add_trace(go.Scatter(
        x=orig_x, y=orig_y,
        mode='markers+text',
        name='Original',
        marker=dict(color='blue', size=10),
        text=[keypoint_labels[idx] if keypoint_labels and idx < len(keypoint_labels) else str(idx) for idx in range(len(orig_x))],
        textposition="top center"
    ))
    
    # 원본 Skeleton
    if skeleton:
        for joint_pair in skeleton:
            idx1, idx2 = joint_pair
            if idx1 < len(orig_x) and idx2 < len(orig_x):
                fig.add_trace(go.Scatter(
                    x=[orig_x[idx1], orig_x[idx2]],
                    y=[orig_y[idx1], orig_y[idx2]],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    showlegend=False
                ))
    
    # 증강된 키포인트
    if augmented is not None:
        aug_x = augmented[::3]
        aug_y = augmented[1::3]
        fig.add_trace(go.Scatter(
            x=aug_x, y=aug_y,
            mode='markers+text',
            name='Augmented',
            marker=dict(color='red', size=10),
            text=[keypoint_labels[idx] if keypoint_labels and idx < len(keypoint_labels) else str(idx) for idx in range(len(aug_x))],
            textposition="top center"
        ))
        
        # 증강된 Skeleton
        if skeleton:
            for joint_pair in skeleton:
                idx1, idx2 = joint_pair
                if idx1 < len(aug_x) and idx2 < len(aug_x):
                    fig.add_trace(go.Scatter(
                        x=[aug_x[idx1], aug_x[idx2]],
                        y=[aug_y[idx1], aug_y[idx2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Relative X',
        yaxis_title='Relative Y',
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'),  # y축 반전
        xaxis=dict(constrain='domain'),
        showlegend=True,
        width=800,
        height=600
    )
    
    fig.show()

# 8. 유사성 평가 함수
def evaluate_similarity(original_data, augmented_data):
    """
    original_data: numpy array [num_samples, input_dim]
    augmented_data: numpy array [num_samples, input_dim]
    
    Returns:
        mse: Mean Squared Error
        mae: Mean Absolute Error
    """
    mse = mean_squared_error(original_data, augmented_data)
    mae = mean_absolute_error(original_data, augmented_data)
    return mse, mae

# 9. Reverse Diffusion (no_grad 사용)
def reverse_diffusion_no_grad(model, alphas_cumprod, timesteps, num_samples, device, input_dim):
    """
    모델을 사용하여 새로운 샘플 생성 (no_grad 사용)
    """
    with torch.no_grad():
        # Initialize with noise
        samples = torch.randn(num_samples, input_dim).to(device)
        
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.float32).to(device)
            alpha_t = alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # 예측된 노이즈
            pred_noise = model(samples, t_tensor)
            
            # 현재 스텝에서의 x_0 예측
            x0_pred = (samples - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t
            
            # 노이즈 샘플링 (t > 0일 때만 노이즈 추가)
            if t > 0:
                noise = torch.randn_like(samples)
            else:
                noise = torch.zeros_like(samples)
            
            # 다음 샘플 업데이트
            samples = sqrt_alpha_t * x0_pred + sqrt_one_minus_alpha_t * noise
        
        return samples  # [num_samples, input_dim]

# 10. 역정규화 함수
def denormalize_keypoints(aligned_keypoints, normalization_params):
    """
    정규화된 키포인트 데이터를 원래 스케일로 역정규화합니다.
    
    Args:
        aligned_keypoints (numpy.ndarray): [num_samples, 78] 정렬된 키포인트 데이터.
        normalization_params (dict): 정규화에 사용된 평균과 표준편차.
            {"x_mean": float, "x_std": float, "y_mean": float, "y_std": float}
    
    Returns:
        numpy.ndarray: 역정규화된 키포인트 데이터.
    """
    denormalized = aligned_keypoints.copy()
    # X 좌표 역정규화
    denormalized[:, ::3] = (denormalized[:, ::3] * normalization_params["x_std"]) + normalization_params["x_mean"]
    # Y 좌표 역정규화
    denormalized[:, 1::3] = (denormalized[:, 1::3] * normalization_params["y_std"]) + normalization_params["y_mean"]
    return denormalized

# 11. 데이터 로드 및 전처리 함수
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

# 12. 메인 함수
def main():
    # 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 모델 파라미터 설정
    input_dim = 78
    hidden_dim = 256
    time_embed_dim = 32
    dropout_prob = 0.3
    timesteps = 100
    
    # 모델 초기화
    model = UNet(input_dim, hidden_dim, time_embed_dim, dropout_prob=dropout_prob).to(device)
    
    # 가중치 초기화 (Kaiming Normal)
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(initialize_weights)
    
    # 모델 가중치 로드
    model_path = './unet_diffusion_model_best.pth'  # 저장된 모델 경로
    try:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        print("모델 가중치 로드 완료!")
    except FileNotFoundError:
        print(f"모델 가중치 파일을 찾을 수 없습니다: {model_path}")
        return
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생: {e}")
        return
    
    # 데이터 로드 및 전처리
    input_json_path = '/home/wook/diff/error_keypoints.json'  # 오류 키포인트 데이터 경로
    reference_point_idx = 0  # 허리 키포인트의 인덱스로 설정 (데이터셋에 따라 조정 필요)
    
    try:
        aligned_error_keypoints, normalization_params = load_and_preprocess_data(
            input_json_path, reference_point_idx=reference_point_idx
        )  # [num_samples, 78], normalization params
        print(f"Aligned Keypoints shape: {aligned_error_keypoints.shape}")
        print(f"Normalization Params: {normalization_params}")
    except FileNotFoundError:
        print(f"입력 JSON 파일을 찾을 수 없습니다: {input_json_path}")
        return
    except Exception as e:
        print(f"데이터 로드 및 전처리 중 오류 발생: {e}")
        return
    
    # 데이터 증강 (옵션)
    # 증강된 데이터를 추가로 생성하려면 다음 주석을 해제하세요.
    augmented_keypoints_list = augment_keypoints(aligned_error_keypoints)
    all_keypoints = np.vstack([aligned_error_keypoints] + augmented_keypoints_list)
    print(f"Augmented Keypoints shape: {all_keypoints.shape}")
    
    # Torch 텐서로 변환
    all_keypoints_tensor = torch.tensor(all_keypoints, dtype=torch.float32).to(device)
    
    # 데이터 정보 출력
    print(f"Keypoints shape: {all_keypoints_tensor.shape}")  # [num_samples, 78]
    print(f"Sample keypoints (first 5 samples):\n{all_keypoints_tensor[:5]}")
    
    # Forward Diffusion 과정의 알파 누적 곱 계산
    alphas = 1 - np.linspace(0.0001, 0.02, timesteps)
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32).to(device)
    
    # Reverse Diffusion Process (no_grad 사용)
    num_augmented_samples = 1000  # 원하는 샘플 수로 조정 가능
    
    print(f"{num_augmented_samples}개의 증강된 키포인트를 생성 중입니다...")
    start_time = time.time()
    augmented_samples = reverse_diffusion_no_grad(model, alphas_cumprod, timesteps, num_augmented_samples, device, input_dim)
    generation_time = time.time() - start_time
    print(f"증강된 데이터 생성 완료 - 소요 시간: {generation_time:.2f}s")
    
    # 증강된 데이터 가져오기
    augmented_samples = augmented_samples.detach().cpu().numpy()
    
    # 원본 데이터와 증강된 데이터 비교 (유사성 평가)
    # 원본 데이터에서 num_augmented_samples만큼 샘플링하여 비교
    original_samples = all_keypoints[:num_augmented_samples]
    
    mse, mae = evaluate_similarity(original_samples, augmented_samples)
    print(f"유사성 평가 지표:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    
    # 개별 샘플의 MAE 계산
    mae_per_sample = np.mean(np.abs(original_samples - augmented_samples), axis=1)
    
    # 가장 유사한 샘플의 인덱스 찾기 (MAE 최소값)
    min_mae_index = np.argmin(mae_per_sample)
    min_mae_value = mae_per_sample[min_mae_index]
    print(f"가장 유사한 샘플 인덱스: {min_mae_index}, MAE: {min_mae_value:.6f}")
    
    # 가장 유사한 샘플 시각화
    if min_mae_index < num_augmented_samples:
        original_sample = original_samples[min_mae_index]
        augmented_sample = augmented_samples[min_mae_index]
        keypoint_labels = [
            "코", "왼쪽 눈", "오른쪽 눈", "왼쪽 귀", "오른쪽 귀",
            "왼쪽 어깨", "오른쪽 어깨", "왼쪽 팔꿈치", "오른쪽 팔꿈치",
            "왼쪽 손목", "오른쪽 손목", "왼쪽 엉덩이", "오른쪽 엉덩이",
            "왼쪽 무릎", "오른쪽 무릎", "왼쪽 발목", "오른쪽 발목"
        ]
        skeleton = [
            (5, 7), (7, 9),    # 왼팔
            (6, 8), (8, 10),   # 오른팔
            (5, 6),            # 어깨
            (5, 11), (6, 12),  # 몸통
            (11, 13), (13, 15), # 왼다리
            (12, 14), (14, 16)  # 오른다리
        ]
        plot_keypoints_plotly(
            original=original_sample,
            augmented=augmented_sample,
            skeleton=skeleton,
            keypoint_labels=keypoint_labels,
            title=f"Most Similar Sample (Index {min_mae_index}, MAE: {min_mae_value:.6f})"
        )
    else:
        print(f"Sample index {min_mae_index}은(는) 데이터 범위를 벗어났습니다.")
    
    # 역정규화된 데이터로 시각화 (증강된 데이터)
    denormalized_augmented = denormalize_keypoints(
        augmented_samples,
        normalization_params=normalization_params
    )
    
    # 가장 유사한 샘플의 역정규화된 데이터 시각화
    if min_mae_index < len(denormalized_augmented):
        denorm_augmented_sample = denormalized_augmented[min_mae_index]
        plot_keypoints_plotly(
            original=denorm_augmented_sample,
            augmented=None,
            skeleton=skeleton,
            keypoint_labels=keypoint_labels,
            title=f"Denormalized Augmented Sample (Index {min_mae_index})"
        )
    else:
        print(f"Sample index {min_mae_index}은(는) 데이터 범위를 벗어났습니다.")
    
    # 증강된 데이터를 역정규화하여 실제 스케일로 변환 후 저장 (선택 사항)
    augmented_data_denormalized = denormalize_keypoints(
        augmented_samples,
        normalization_params=normalization_params
    )
    
    with open('./augmented_keypoints_denormalized.json', 'w') as f:
        json.dump(augmented_data_denormalized.tolist(), f)
    print("역정규화된 증강 데이터를 'augmented_keypoints_denormalized.json'에 저장했습니다.")
    
    # (옵션) 증강된 데이터를 정규화된 상태로 저장하고 싶다면 다음을 사용하세요:
    # augmented_data_normalized = augmented_samples.tolist()
    # with open('./augmented_keypoints_normalized.json', 'w') as f:
    #     json.dump(augmented_data_normalized, f)
    # print("정규화된 증강 데이터를 'augmented_keypoints_normalized.json'에 저장했습니다.")

if __name__ == "__main__":
    main()
