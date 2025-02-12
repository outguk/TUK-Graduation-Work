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
    """파일 이름을 자연스럽게 정렬하기 위한 키 생성"""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def calculate_movement(previous_keypoints, current_keypoints):
    """10프레임 동안 키포인트 움직임 변화량을 계산"""
    movement = np.linalg.norm(previous_keypoints - current_keypoints, axis=(1, 2))
    mean_movement = np.mean(movement)
    return mean_movement

def json_to_pkl(json_dir, output_pkl_path, frames_per_annotation=10, movement_threshold=30.0):
    """JSON 데이터를 읽어서 PKL로 변환 (키포인트 변화량만 이용)"""
    annotations = []
    split_xsub_val = []
    normal_behavior_frames = []  # 정상 행동으로 분류된 프레임 리스트
    movement_values = []  # 모든 프레임의 움직임 변화량 저장 리스트

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')], key=natural_key)

    for i in range(0, len(json_files), frames_per_annotation):
        keypoints_list = []
        frame_dirs = []

        for j in range(frames_per_annotation):
            if i + j >= len(json_files):
                break

            json_file = json_files[i + j]
            with open(os.path.join(json_dir, json_file), 'r') as f:
                data = json.load(f)

            keypoints_list.append(np.array(data["keypoints"])[:, :2].astype(int))
            frame_dirs.append(json_file.replace(".json", ""))

        if len(keypoints_list) < frames_per_annotation:
            continue

        # 🔹 키포인트 변화량 계산
        movement = calculate_movement(np.array(keypoints_list[:-1]), np.array(keypoints_list[1:]))
        movement_values.append(movement)

        # 🔹 움직임이 threshold보다 크면 저장, 작으면 정상 행동으로 분류
        should_save = movement > movement_threshold

        if should_save:
            annotations.append({"frame_dir": frame_dirs[0], "total_frames": len(keypoints_list)})
            split_xsub_val.extend(frame_dirs)
        else:
            normal_behavior_frames.append(frame_dirs[0])

        # ✅ 디버깅: 프레임별 움직임 변화량 출력
        print(f"\n-> 프레임 {frame_dirs[0]} - 움직임 변화량: {movement:.2f} (임계값: {movement_threshold}) {'O 모델 입력' if should_save else 'X 정상 행동'}")

    # 🔥 전체 움직임 변화량 분석
    print("\n 움직임 변화량 분석:")
    print(f"  ▶ 평균 움직임 변화량: {np.mean(movement_values):.2f}")
    print(f"  ▶ 최소 움직임 변화량: {np.min(movement_values):.2f}")
    print(f"  ▶ 최대 움직임 변화량: {np.max(movement_values):.2f}")

    # 🔥 정상 행동으로 분류된 프레임 출력
    print("\n 정상 행동으로 분류된 프레임들:")
    for frame in normal_behavior_frames:
        print(f"  - {frame}")

    # 🔹 변환된 데이터 저장
    converted_data = {
        "split": {"xsub_val": split_xsub_val},
        "annotations": annotations
    }
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(converted_data, f)

    print(f"\n PKL file saved at {output_pkl_path}")

# 실행
json_to_pkl("../mmpose/keypoints", "results.pkl")


##키포인트 변화량 + 차렷 자세 키포인트 비율 계산
# import os
# import json
# import numpy as np
# import pickle
# from tqdm import tqdm
# import re

# def natural_key(text):
#     """파일 이름을 자연스럽게 정렬하기 위한 키 생성"""
#     return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# def calculate_movement(previous_keypoints, current_keypoints):
#     """10프레임 동안 키포인트 움직임 변화량을 계산"""
#     movement = np.linalg.norm(previous_keypoints - current_keypoints, axis=(1, 2))
#     mean_movement = np.mean(movement)
#     return mean_movement

# def is_at_attention_pose(keypoints_seq, frame_number):
#     """10프레임 평균 키포인트를 이용해 차렷 자세인지 판단"""
#     keypoints_seq = np.array(keypoints_seq)
#     avg_keypoints = np.mean(keypoints_seq, axis=0)

#     # 주요 키포인트 가져오기
#     nose = avg_keypoints[0]  # 코
#     left_ear, right_ear = avg_keypoints[3], avg_keypoints[4]  # 왼쪽/오른쪽 귀
#     left_shoulder, right_shoulder = avg_keypoints[5], avg_keypoints[6]
#     left_elbow, right_elbow = avg_keypoints[7], avg_keypoints[8]
#     left_wrist, right_wrist = avg_keypoints[9], avg_keypoints[10]
#     left_hip, right_hip = avg_keypoints[11], avg_keypoints[12]
#     left_foot, right_foot = avg_keypoints[15], avg_keypoints[16]

#     # 🔹 머리 움직임
#     head_tilt = abs(left_ear[0] - right_ear[0])  # 머리 기울기
#     head_movement = np.linalg.norm(nose - left_shoulder)  # 코의 이동량

#     # 🔹 어깨 & 엉덩이 정렬
#     shoulder_balance = abs(left_shoulder[0] - right_shoulder[0])
#     hip_balance = abs(left_hip[0] - right_hip[0])

#     # 🔹 발 정렬
#     foot_alignment = abs(left_foot[0] - right_foot[0])

#     # 🔹 상체 기울기 (어깨-엉덩이 정렬)
#     body_tilt = abs((left_shoulder[0] - left_hip[0]) - (right_shoulder[0] - right_hip[0]))

#     # 🔹 팔 위치
#     elbow_distance = abs(left_elbow[0] - right_elbow[0])
#     wrist_distance = abs(left_wrist[0] - right_wrist[0])

#     # 📌 차렷 자세 판별 기준
#     head_tilt_threshold =70
#     head_movement_threshold = 150
#     shoulder_threshold = 180
#     hip_threshold = 100
#     foot_threshold = 60
#     body_tilt_threshold = 80
#     elbow_threshold = 250
#     wrist_threshold = 200

#     is_attention = (
#         head_tilt < head_tilt_threshold and
#         head_movement < head_movement_threshold and
#         shoulder_balance < shoulder_threshold and
#         hip_balance < hip_threshold and
#         foot_alignment < foot_threshold and
#         body_tilt < body_tilt_threshold and
#         elbow_distance < elbow_threshold and
#         wrist_distance < wrist_threshold
#     )

#     # 디버깅 코드 추가
#     if not is_attention:
#         print(f"\n🔍 프레임 {frame_number} - 차렷 자세 판별 실패:")
#         print(f"  ▶ 머리 기울기: {head_tilt:.2f} / 허용값: {head_tilt_threshold}")
#         print(f"  ▶ 머리 움직임: {head_movement:.2f} / 허용값: {head_movement_threshold}")
#         print(f"  ▶ 어깨 균형: {shoulder_balance:.2f} / 허용값: {shoulder_threshold}")
#         print(f"  ▶ 엉덩이 균형: {hip_balance:.2f} / 허용값: {hip_threshold}")
#         print(f"  ▶ 발 정렬: {foot_alignment:.2f} / 허용값: {foot_threshold}")
#         print(f"  ▶ 상체 기울기: {body_tilt:.2f} / 허용값: {body_tilt_threshold}")
#         print(f"  ▶ 팔꿈치 거리: {elbow_distance:.2f} / 허용값: {elbow_threshold}")
#         print(f"  ▶ 손목 거리: {wrist_distance:.2f} / 허용값: {wrist_threshold}")

#     return is_attention

# def json_to_pkl(json_dir, output_pkl_path, frames_per_annotation=10, movement_threshold=50.0):
#     """JSON 데이터를 읽어서 PKL로 변환"""
#     annotations = []
#     split_xsub_val = []
#     normal_behavior_frames = []

#     json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')], key=natural_key)

#     for i in tqdm(range(0, len(json_files), frames_per_annotation), desc="Processing JSON files"):
#         keypoints_list = []
#         frame_dirs = []

#         for j in range(frames_per_annotation):
#             if i + j >= len(json_files):
#                 break

#             json_file = json_files[i + j]
#             with open(os.path.join(json_dir, json_file), 'r') as f:
#                 data = json.load(f)

#             keypoints_list.append(np.array(data["keypoints"])[:, :2].astype(int))
#             frame_dirs.append(json_file.replace(".json", ""))

#         if len(keypoints_list) < frames_per_annotation:
#             continue

#         movement = calculate_movement(np.array(keypoints_list[:-1]), np.array(keypoints_list[1:]))

#         if movement > movement_threshold:
#             should_save = True
#         else:
#             should_save = not is_at_attention_pose(keypoints_list, frame_dirs[0])

#         if should_save:
#             annotations.append({"frame_dir": frame_dirs[0], "total_frames": len(keypoints_list)})
#             split_xsub_val.extend(frame_dirs)
#         else:
#             normal_behavior_frames.append(frame_dirs[0])

#     print("\n✅ 정상 행동으로 분류된 프레임들:", normal_behavior_frames)

# json_to_pkl("../mmpose/keypoints", "results.pkl")
