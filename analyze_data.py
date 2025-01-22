import json
import os

# 라벨 파일 로드
label_file_path = '/home/wook/MotionBERT/fitness-aqa/OHP/Labeled_Dataset/label/knees/train_error.json'
with open(label_file_path, 'r') as f:
    labels = json.load(f)

# 라벨 개수 계산
total_labels = len(labels)
error_labels = sum(1 for label in labels.values() if label == 1)
normal_labels = total_labels - error_labels

# 라벨 개수를 텍스트 파일로 저장
label_counts = {
    "Total Labels": total_labels,
    "Normal Labels (0)": normal_labels,
    "Error Labels (1)": error_labels
}

label_counts_output_path = '/home/wook/diff/label_counts.txt'
with open(label_counts_output_path, 'w') as f:
    for key, value in label_counts.items():
        f.write(f"{key}: {value}\n")

print(f"라벨 개수 정보 저장 완료! 총 {total_labels}개의 라벨 중 정상 라벨은 {normal_labels}개, 오류 라벨은 {error_labels}개입니다.")

# 오류 프레임 ID 추출
error_frame_ids = [key for key, value in labels.items() if value == 1]

# 오류 데이터 저장
error_data = []

for frame_id in error_frame_ids:
    # JSON 파일 경로 구성 (동영상 단위로 저장된 파일 이름만 추출)
    base_id = "_".join(frame_id.split("_")[:2])  # e.g., "80825_1"
    frame_number = frame_id.split("_")[-1]      # e.g., "78"
    json_path = f"/home/wook/MotionBERT/fitness-aqa/OHP/Labeled_Dataset/2d_one/{base_id}.json"

    if not os.path.exists(json_path):
        print(f"파일을 찾을 수 없습니다: {json_path}")
        continue

    # JSON 파일 로드 및 프레임별 데이터 추출
    with open(json_path, 'r') as f_json:
        all_frames = json.load(f_json)

    # 해당 프레임 데이터 찾기
    frame_key = f"{frame_number}.jpg"
    frame_data = next((frame for frame in all_frames if frame["image_id"] == frame_key), None)

    if frame_data:
        error_data.append(frame_data)
    else:
        print(f"프레임을 찾을 수 없습니다: {frame_key} in {json_path}")

# 오류 데이터 저장
error_output_path = '/home/wook/diff/error_keypoints.json'
with open(error_output_path, 'w') as f_error:
    json.dump(error_data, f_error)

print(f"오류 키포인트 데이터 저장 완료! 총 {len(error_data)}개의 프레임 추출됨.")
