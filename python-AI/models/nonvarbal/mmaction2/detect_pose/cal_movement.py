# 프레임 단위로 움직임(키포인트) 변화량 계산 
# 임계값 movement_threshold를 기준으로 움직임이 크면 모델 입력 데이터로 사용, 작으면 정상 행동으로 분류
# movent_threshold: 30.0

import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import re

def natural_key(text):
    #파일 순서 정렬 
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def calculate_movement(previous_keypoints, current_keypoints):
    """10프레임 동안 키포인트 움직임 변화량을 계산"""
    movement = np.linalg.norm(previous_keypoints - current_keypoints, axis=(1, 2))
    mean_movement = np.mean(movement)
    return mean_movement

def json_to_pkl(video_filename, frames_per_annotation=10, movement_threshold=35.0):
    """
    영상 이름 기반으로 JSON 키포인트를 읽고, PKL 변환
    - video_filename: 예) 'book'
    """
    json_dir = os.path.join("data", "keypoints", video_filename)
    output_pkl_path = os.path.join(json_dir, "results.pkl")

    annotations = []
    split_xsub_val = []
    normal_behavior_frames = []
    movement_values = []

    json_files = sorted(
        [f for f in os.listdir(json_dir) if f.endswith('.json')],
        key=natural_key
    )

    for i in range(0, len(json_files), frames_per_annotation):
        keypoints_list = []
        scores_list = []
        frame_dirs = []
        img_shape = None
        original_shape = None

        for j in range(frames_per_annotation):
            if i + j >= len(json_files):
                break

            json_file = json_files[i + j]
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)
            
             # ✅ 디버깅 코드 시작
            if "keypoints" not in data:
                print(f"[❗오류] '{json_file}' → keypoints 키 없음 → 건너뜀")
                continue

            keypoints = np.array(data["keypoints"])

            if keypoints.ndim != 2 or keypoints.shape[1] < 2:
                print(f"[❗오류] '{json_file}' → keypoints shape 이상함: {keypoints.shape} → 건너뜀")
                continue

            print(f"[✅ 정상] '{json_file}' → keypoints shape: {keypoints.shape}")
            # ✅ 디버깅 코드 끝

            keypoints = np.array(data["keypoints"])
            coordinates = keypoints[:, :2].astype(int)
            scores = keypoints[:, 2]

            keypoints_list.append(coordinates)
            scores_list.append(scores)
            frame_dirs.append(json_file.replace(".json", ""))

            if img_shape is None:
                img_shape = data.get("img_shape", (1080, 1920))
            if original_shape is None:
                original_shape = data.get("original_shape", (1080, 1920))

                # keypoints가 2개 미만이면 분석 스킵
        if len(keypoints_list) < 2:
            print(f"⚠️ 키포인트 부족: {len(keypoints_list)}개 → 건너뜀")
            continue

        # 변환 및 차원 검사
        arr1 = np.array(keypoints_list[:-1])
        arr2 = np.array(keypoints_list[1:])

        if arr1.ndim != 3 or arr2.ndim != 3:
            print(f"⚠️ 차원 이상 arr1: {arr1.shape}, arr2: {arr2.shape} → 건너뜀")
            continue

        # 변화량 계산
        movement = calculate_movement(arr1, arr2)

        # movement = calculate_movement(np.array(keypoints_list[:-1]), np.array(keypoints_list[1:]))
        movement_values.append(movement)

        should_save = movement > movement_threshold

        if should_save:
            annotation = {
                "frame_dir": frame_dirs[0],
                "total_frames": len(keypoints_list),
                "keypoint": np.expand_dims(np.array(keypoints_list), axis=0),
                "keypoint_score": np.expand_dims(np.array(scores_list), axis=0),
                "img_shape": img_shape,
                "original_shape": original_shape,
                "label": -1
            }
            annotations.append(annotation)
            split_xsub_val.extend(frame_dirs)
        else:
            normal_behavior_frames.append(frame_dirs[0])

        print(f"\n 프레임 {frame_dirs[0]} - 변화량: {movement:.2f} → {'모델 입력' if should_save else '정상 행동'}")

    print(f"\n 평균 변화량: {np.mean(movement_values):.2f}, 최대: {np.max(movement_values):.2f}, 최소: {np.min(movement_values):.2f}")
    print("\n 정상 행동 프레임들:")
    for frame in normal_behavior_frames:
        print(" -", frame)

    converted_data = {
        "split": {"xsub_val": split_xsub_val},
        "annotations": annotations
    }

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(converted_data, f)

    print(f"\n ✅ 결과 저장 완료: {output_pkl_path}")

# ✅ 실행 (슬라이딩 윈도우 없이 키포인트 변화량 적용)
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python cal_movement.py <video_filename (ex: book)>")
        sys.exit(1)

    video_filename = sys.argv[1]
    try:
        json_to_pkl(video_filename)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
