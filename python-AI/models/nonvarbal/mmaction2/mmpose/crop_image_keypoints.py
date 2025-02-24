import os
import cv2
import re
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from tqdm import tqdm

# mmpose 모듈 등록
register_all_modules()

# 모델 불러오기 (HRNet)
# 현재 스크립트(crop_image_keypoints.py)의 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 모델 설정 파일 & 체크포인트 파일 경로 설정
CONFIG_PATH = os.path.join(SCRIPT_DIR, "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth")
model = init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cuda:0')

def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def central_crop(image):
    """
    이미지의 중앙 영역 crop
    가로 영상 -> 세로 크기와 동일한 정사각형 영역을 crop
    단, 인물이 이미지 중앙에 있어야 함 (인물이 중앙에 없는 경우 별도 처리 필요) 0214
    """
    h, w = image.shape[:2]
    # 만약 가로가 세로보다 크다면, 중앙 정사각형 crop (예: 1920×1080 → 1080×1080)
    if w > h:
        crop_size = h  # 세로 길이를 crop 크기로 사용
        start_x = (w - crop_size) // 2
        cropped = image[:, start_x:start_x+crop_size]
    else:
        # 세로 영상이면 원본 사용 (또는 별도 처리)
        cropped = image
    return cropped

def visualize_keypoints_from_image(image, keypoints, keypoint_scores=None, output_path='output.jpg', kpt_score_thr=0.5):
    """
    관절 데이터를 이미지에 시각화하는 함수.
    image: numpy array (이미지)
    keypoints: [[x, y], ...] 또는 [[x, y, score], ...]
    keypoint_scores: None이 아니면 따로 제공된 점수 리스트
    """
    img_vis = image.copy()
    # 관절 데이터가 [x, y] 형식이면 score 1.0 추가
    if keypoint_scores is None and len(keypoints[0]) == 2:
        keypoints = [kp + [1.0] for kp in keypoints]
    elif keypoint_scores is not None:
        keypoints = [kp + [score] for kp, score in zip(keypoints, keypoint_scores)]
    
    # COCO 포맷 스켈레톤 연결 정보
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # 관절 점 그리기
    for i, (x, y, score) in enumerate(keypoints):
        if score > kpt_score_thr:
            cv2.circle(img_vis, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # 관절 연결선 그리기
    for i, j in skeleton:
        if keypoints[i][2] > kpt_score_thr and keypoints[j][2] > kpt_score_thr:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(img_vis, pt1, pt2, (255, 0, 0), 2)
    
    cv2.imwrite(output_path, img_vis)
    # print(f"Result saved to {output_path}")

def process_images_central_crop(input_dir, output_dir, model, vis_dir=None):
    """
    중앙 영역 Crop 방식을 적용하여 이미지 처리.
    """
    os.makedirs(output_dir, exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')],
        key=natural_key
    )
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            continue

        # 중앙 영역 crop 적용
        cropped_img = central_crop(img)
        
        # 모델 추론 (numpy array를 직접 전달)
        results = inference_topdown(model, cropped_img)
        if len(results) == 0 or len(results[0].pred_instances.keypoints) == 0:
            print(f"No keypoints detected in {image_path}")
            continue
        
        # 첫 번째 인스턴스의 키포인트 및 신뢰도 추출
        keypoints = results[0].pred_instances.keypoints[0].tolist()
        keypoint_scores = results[0].pred_instances.keypoint_scores[0].tolist()
        json_keypoints = [[float(kp[0]), float(kp[1]), float(score)] 
                          for kp, score in zip(keypoints, keypoint_scores)]
        
        # JSON 저장
        json_file = os.path.join(output_dir, image_file.rsplit('.', 1)[0] + '.json')
        with open(json_file, 'w') as f:
            json.dump({"keypoints": json_keypoints}, f, indent=4)
        
        # 시각화: cropped 이미지를 사용
        if vis_dir:
            vis_path = os.path.join(vis_dir, image_file)
            visualize_keypoints_from_image(cropped_img, keypoints, keypoint_scores, output_path=vis_path)
        
        # print(f"Processed (central crop): {image_path} -> JSON: {json_file}")

# 실행 예시 (중앙 영역 crop 방식) 경로 변경
input_dir = "data/frames"          # 입력 이미지 디렉토리
output_dir = "data/keypoints"        # JSON 결과 저장 디렉토리
vis_dir = "data/visualizations"      # 시각화 이미지 저장 디렉토리
process_images_central_crop(input_dir, output_dir, model, vis_dir=vis_dir)
