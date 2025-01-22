# diffusion_2d_keypoint
U-net 구조를 가진 DiffusionModel로 2D-keypoint 생성
데이터의 경우, [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA) 사용.

1. analyze_data.py(2D-keypoint 증강만을 원한다면 생략가능): 
운동 데이터 라벨링 결과를 처리하고, 오류 데이터(프레임)의 세부 정보를 추출 및 저장

2. train_model.py: 
Diffusion Model을 사용하여 키포인트 데이터의 예측 및 재구성을 수행하는 UNet 기반 학습 파이프라인

3. generate_augmented_data.py: 
Diffusion 모델을 사용하여 증강된 키포인트 데이터를 생성, 평가, 시각화 및 저장

4. 17_joint.py(26joint -> 17joint / 즉, 26joint를 사용한다면 생략가능)
증강된 키포인트 데이터에서 특정 관절(17개 관절)**만 선택하여 처리하고, 결과를 저장



Fitness-AQA (Fitness Action Quality Assessment) [ECCV'22] 인용
@inproceedings{parmar2022domain,
  title={Domain Knowledge-Informed Self-supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXVIII},
  pages={105--123},
  year={2022},
  organization={Springer}
}
