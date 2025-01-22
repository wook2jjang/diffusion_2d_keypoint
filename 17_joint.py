# 17_joint.py

import json
import numpy as np

def select_joints(data, selected_indices):
    """
    특정 관절만 선택하여 데이터 형식을 맞춥니다.
    :param data: numpy array [num_samples, J*3] 또는 [num_samples, J, C]
    :param selected_indices: 선택할 관절의 인덱스 리스트 (0부터 시작)
    :return: 선택된 관절만 포함된 numpy array [num_samples, selected_J*3] 또는 [num_samples, selected_J, C]
    """
    if len(data.shape) == 2 and data.shape[1] % 3 == 0:
        # [num_samples, J*3] 형식
        num_joints = data.shape[1] // 3
        data = data.reshape(data.shape[0], num_joints, 3)
        selected_data = data[:, selected_indices, :]  # [num_samples, selected_J, 3]
        return selected_data.reshape(data.shape[0], len(selected_indices)*3)  # [num_samples, 51]
    elif len(data.shape) == 3 and data.shape[2] == 3:
        # [num_samples, J, C] 형식
        selected_data = data[:, selected_indices, :]  # [num_samples, selected_J, 3]
        return selected_data
    else:
        raise ValueError("데이터 형식이 지원되지 않습니다. [num_samples, J*3] 또는 [num_samples, J, C] 형식이어야 합니다.")

def process_augmented_data(input_path, output_path, selected_joint_indices):
    """
    증강된 데이터를 로드하여 특정 관절만 선택하고, 저장합니다.
    :param input_path: 입력 증강 데이터 JSON 파일 경로
    :param output_path: 출력할 처리된 증강 데이터 JSON 파일 경로
    :param selected_joint_indices: 선택할 관절의 인덱스 리스트 (0부터 시작)
    """
    # 데이터 로드
    with open(input_path, 'r') as f:
        augmented_data = json.load(f)
    
    augmented_data = np.array(augmented_data)  # [num_samples, 78]
    print(f"Original augmented_data shape: {augmented_data.shape}")  # 예: (1000,78)
    
    # 17개의 관절만 선택
    try:
        processed_data = select_joints(augmented_data, selected_joint_indices)  # [1000,51]
        print(f"Processed data shape: {processed_data.shape}")  # [1000,51]
    except ValueError as e:
        print(f"Error in selecting joints: {e}")
        return
    
    # 데이터 검증
    if processed_data.shape[1] != 51:
        raise ValueError(f"Processed data shape is incorrect: {processed_data.shape}. Expected second dimension to be 51.")
    
    # 리스트로 변환하여 저장
    processed_data = processed_data.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
    
    print(f"처리된 증강 데이터가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    input_augmented_path = './augmented_keypoints_denormalized.json'  # 기존 증강 데이터 경로
    output_augmented_path = './augmented_keypoints_denormalized_17joints.json'  # 처리된 증강 데이터 저장 경로
    
    # 선택할 17개의 관절 인덱스 (0부터 시작)
    selected_joint_indices = list(range(17))  # [0, 1, 2, ..., 16]
    
    # 또는 특정 관절 인덱스 선택 (데이터셋에 따라 수정 필요)
    # 예: selected_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    process_augmented_data(input_augmented_path, output_augmented_path, selected_joint_indices)