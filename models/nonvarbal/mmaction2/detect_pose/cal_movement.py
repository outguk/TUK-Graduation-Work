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

def json_to_pkl(json_dir, output_pkl_path, frames_per_annotation=10, movement_threshold=30.0):
    """JSON 데이터를 읽어서 PKL로 변환 (키포인트 변화량 적용)"""
    annotations = []
    split_xsub_val = []
    normal_behavior_frames = []  # 정상 행동으로 분류된 프레임 리스트
    movement_values = []  # 모든 프레임의 움직임 변화량 저장 리스트

    # 🔹 JSON 파일 정렬하여 가져오기
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')], key=natural_key)

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

        # 🔹 키포인트 변화량 계산
        movement = calculate_movement(np.array(keypoints_list[:-1]), np.array(keypoints_list[1:]))
        movement_values.append(movement)

        # 🔹 움직임이 threshold보다 크면 저장, 작으면 정상 행동으로 분류
        should_save = movement > movement_threshold

        if should_save:
            # ✅ 모델 입력 데이터로 저장
            keypoints_array = np.expand_dims(np.array(keypoints_list), axis=0)
            scores_array = np.expand_dims(np.array(scores_list), axis=0)

            annotation = {
                "frame_dir": frame_dirs[0],  # 첫 번째 프레임의 이름 사용
                "total_frames": len(keypoints_list),
                "keypoint": keypoints_array,
                "keypoint_score": scores_array,
                "img_shape": img_shape,
                "original_shape": original_shape,
                "label": -1
            }
            annotations.append(annotation)
            split_xsub_val.extend(frame_dirs)

            # print(f"\n 저장된 프레임: {frame_dirs[0]} (움직임 변화량: {movement:.2f})")
        else:
            normal_behavior_frames.append(frame_dirs[0])

        # ✅ 디버깅: 프레임별 움직임 변화량 출력
        print(f"\n 프레임 {frame_dirs[0]} - 변화량: {movement:.2f} (임계값: {movement_threshold}) {'✅ 모델 입력' if should_save else '⏹ 정상 행동'}")

    # 🔥 전체 움직임 변화량 분석
    print("\n 움직임 변화량 분석:")
    print(f"  ▶ 평균 변화량: {np.mean(movement_values):.2f}")
    print(f"  ▶ 최소 변화량: {np.min(movement_values):.2f}")
    print(f"  ▶ 최대 변화량: {np.max(movement_values):.2f}")

    # 🔥 정상 행동으로 분류된 프레임 출력
    print("\n⏹ 정상 행동으로 분류된 프레임들:")
    for frame in normal_behavior_frames:
        print(f"  - {frame}")

    # 🔹 변환된 데이터 저장
    converted_data = {
        "split": {"xsub_val": split_xsub_val},
        "annotations": annotations
    }

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(converted_data, f)

    print(f"\n✅ PKL file saved at {output_pkl_path}")

# ✅ 실행 (슬라이딩 윈도우 없이 키포인트 변화량 적용)
json_to_pkl("data/keypoints", "data/keypoints/results.pkl", frames_per_annotation=10, movement_threshold=30.0)


