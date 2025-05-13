import cv2
import numpy as np
import os
import json
import re
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

#RTMPose 모델 사용
#연산량이 적어 빠르게 동작, 작은 메모리 사용 (실시간 애플리케이션에 적합)
#정확도는 hrnet보다 낮을 수 있음

#이미지 크기, 밝기 등 전처리

# MMPose 모듈 등록
register_all_modules()

# 모델 설정 (RTMPose)
rtm_config_file = 'rtmpose-m_8xb256-420e_coco-256x192.py'
rtm_checkpoint_file = 'rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
model = init_model(rtm_config_file, rtm_checkpoint_file, device='cuda:0')

# 입력 이미지 크기 설정 (가로 640px 유지, 세로 비율 유지)
TARGET_WIDTH = 640  

# 자연 정렬 함수 (파일 정렬 시 사용)
def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

# 밝기 조정 (히스토그램 평활화 적용)
def adjust_brightness(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])  # 밝기 채널(Y)만 조정
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# 감마 보정 (어두운 영역 강조)
def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# 대비 조정 (CLAHE 적용)
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# 이미지 리사이징 함수 (비율 유지)
def resize_image(image, target_width=TARGET_WIDTH):
    h, w = image.shape[:2]
    scale = target_width / w
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

# 키포인트 복구 함수 (원본 이미지 크기로 좌표 변환)
def recover_keypoints(keypoints, scale):
    keypoints[:, :, 0] /= scale
    keypoints[:, :, 1] /= scale
    return keypoints

# 키포인트 시각화 함수 (입력 이미지 위에 키포인트 표시)
def visualize_keypoints(image_path, keypoints, keypoint_scores=None, output_path='output.jpg', kpt_score_thr=0.5):
    image = cv2.imread(image_path)

    # 키포인트 데이터가 (x, y)만 있는 경우, score=1.0 추가
    processed_keypoints = []
    for kp in keypoints:
        if len(kp) == 2:
            processed_keypoints.append([kp[0], kp[1], 1.0])
        else:
            processed_keypoints.append(kp)

    # COCO 포맷 스켈레톤 (관절 연결선)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for i, (x, y, score) in enumerate(processed_keypoints):
        if score > kpt_score_thr:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 초록색 점

    for i, j in skeleton:
        if processed_keypoints[i][2] > kpt_score_thr and processed_keypoints[j][2] > kpt_score_thr:
            pt1 = (int(processed_keypoints[i][0]), int(processed_keypoints[i][1]))
            pt2 = (int(processed_keypoints[j][0]), int(processed_keypoints[j][1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # 파란색 선

    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

# 이미지 처리 함수 (디렉토리 내 모든 이미지 처리)
def process_images(input_dir, output_dir, model, vis_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')],
        key=natural_key
    )

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # 이미지 전처리 (밝기 및 대비 조정 적용)
        image = adjust_brightness(image)  # 밝기 보정
        image = gamma_correction(image, gamma=1.5)  # 감마 보정
        image = enhance_contrast(image)  # 대비 강화

        # 이미지 리사이징 적용
        resized_image, scale = resize_image(image)

        # 모델 추론
        results = inference_topdown(model, resized_image)

        if len(results) == 0 or len(results[0].pred_instances.keypoints) == 0:
            print(f"No keypoints detected in {image_path}")
            continue

        # 원본 크기로 키포인트 좌표 복구
        keypoints = recover_keypoints(results[0].pred_instances.keypoints, scale)
        keypoint_scores = results[0].pred_instances.keypoint_scores

        # JSON 저장 (신뢰도 포함)
        json_keypoints = [[float(kp[0]), float(kp[1]), float(score)] for kp, score in zip(keypoints[0], keypoint_scores[0])]
        json_file = os.path.join(output_dir, image_file.replace('.jpg', '.json').replace('.png', '.json'))
        with open(json_file, 'w') as f:
            json.dump({"keypoints": json_keypoints}, f, indent=4)

        # 시각화 및 저장
        if vis_dir:
            vis_path = os.path.join(vis_dir, image_file)
            visualize_keypoints(image_path, keypoints[0], keypoint_scores[0], output_path=vis_path)

        print(f"Processed: {image_path} -> JSON: {json_file}")

# 실행
input_dir = "raw_data"  # 입력 이미지 폴더
output_dir = "keypoints"  # JSON 저장 폴더
vis_dir = "visualizations"  # 시각화 이미지 저장 폴더

process_images(input_dir, output_dir, model, vis_dir=vis_dir)
