import os
import json
import numpy as np
import pickle
from tqdm import tqdm

def json_to_pkl(json_dir, output_pkl_path):
    """
    JSON 데이터를 PKL로 변환.

    Args:
        json_dir (str): JSON 파일들이 있는 디렉토리.
        output_pkl_path (str): 변환된 .pkl 파일 저장 경로.
    """
    all_keypoints = []  # [M x T x V x C]
    all_keypoint_scores = []  # [M x T x V]

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    for json_file in tqdm(json_files, desc="Converting JSON to PKL"):
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)

        # JSON에서 keypoints와 신뢰도 값 추출
        keypoints = np.array(data["keypoints"])  # (V, 3)
        coordinates = keypoints[:, :2]  # (V, 2)
        scores = keypoints[:, 2]  # (V,)

        # 모델 입력 형식으로 변환
        all_keypoints.append(coordinates)  # [T x V x C]
        all_keypoint_scores.append(scores)  # [T x V]

    # M = 1 (한 명의 관절 데이터만 사용), T = 프레임 수
    all_keypoints = np.expand_dims(np.array(all_keypoints), axis=0)  # [M x T x V x C]
    all_keypoint_scores = np.expand_dims(np.array(all_keypoint_scores), axis=0)  # [M x T x V]

    # PKL 파일로 저장
    with open(output_pkl_path, 'wb') as f:
        pickle.dump({"keypoint": all_keypoints, "keypoint_score": all_keypoint_scores}, f)
    print(f"PKL file saved at {output_pkl_path}")

# 실행
json_dir = "keypoints"  # JSON 파일들이 저장된 디렉토리
output_pkl_path = "keypoints/results.pkl"  # 저장할 PKL 파일 경로
json_to_pkl(json_dir, output_pkl_path)
