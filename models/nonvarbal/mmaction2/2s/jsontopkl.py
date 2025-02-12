import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import re

# JSON 파일을 PKL 파일로 변환
# frames_per_annotation: 한 annotation 당 프레임 수 10으로 설정
# json_dir: JSON 파일이 있는 디렉토리
# output_pkl_path: PKL 파일 저장 경로
def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def json_to_pkl(json_dir, output_pkl_path, frames_per_annotation=10):
    annotations = []
    split_xsub_val = []

    #자연수 순서대로 정렬되도록 변경. / 기존 1->10->11 19->2->20->21->... 30..->39->4 ->5 ->6
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')], key=natural_key)

    for i in tqdm(range(0, len(json_files), frames_per_annotation), desc="Converting JSON to PKL"):
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

        if len(keypoints_list) == 0:
            continue

        # 차원 확장 및 배열로 변환
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

        # if len(annotations) <= 2:  # 처음 2개 annotation만 출력
        #     print(f"\nannotation {len(annotations)} keypoint shape:", keypoints_array.shape)
        #     print(f"첫 번째 키포인트:", keypoints_array[0][0][0])

    converted_data = {
        "split": {"xsub_val": split_xsub_val},
        "annotations": annotations
    }

    with open(output_pkl_path, 'wb') as f:
        pickle.dump(converted_data, f)

    print(f"✅ PKL file saved at {output_pkl_path}")

json_to_pkl("../mmpose/keypoints", "../mmpose/keypoints/results.pkl")