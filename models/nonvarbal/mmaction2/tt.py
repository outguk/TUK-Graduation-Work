import pickle
import numpy as np


#  결과 확인용 코드

with open("mmpose/keypoints/results.pkl", "rb") as f:
    test_data = pickle.load(f)

print(f"테스트 데이터 키포인트 개수: {test_data['annotations'][0]['keypoint'].shape}")

# 올바른 차원 (1, 1, T, 17, 3) 형태인지 확인 후 변환
if len(test_data['annotations'][0]['keypoint'].shape) == 3:
    test_data['annotations'][0]['keypoint'] = np.expand_dims(test_data['annotations'][0]['keypoint'], axis=0)  # (1, T, V, C)
    test_data['annotations'][0]['keypoint'] = np.expand_dims(test_data['annotations'][0]['keypoint'], axis=0)  # (1, 1, T, V, C)
