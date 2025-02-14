from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
# from mmpose import visulization
import pickle  # 결과 저장을 위해 사용
import os
import json
import cv2
import re


# Hrnet 모델 사용
# 연산량이 많고 메모리 사용량량 높음
# 정확도는 높음


#mmpose 모듈 등록
register_all_modules()

#모델 불러오기
config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')  

# 자연 정렬 함수: 파일명 내 숫자 부분을 올바르게 분리하여 정렬
def natural_key(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


#keypoint 저장된 데이터를 이용하여 원본 이미지위에 keypoint 시각화
def visualize_keypoints(image_path, keypoints, keypoint_scores=None, output_path='output.jpg', kpt_score_thr=0.1):
    
    """
    관절 데이터 이미지 시각화

    image_path (str): 입력 이미지 경로.
    keypoints (list): 관절 좌표 리스트 [[x, y], ...] 또는 [[x, y, score], ...].
    keypoint_scores (list): 관절 신뢰도 점수 리스트. None일 경우 keypoints에서 추출.
    output_path (str): 결과 이미지 저장 경로.
    kpt_score_thr (float): 신뢰도 임계값.
    
    """
    # 이미지 로드
    image = cv2.imread(image_path)

    # 관절 데이터가 [x, y] 형식이면 신뢰도 추가
    if keypoint_scores is None and len(keypoints[0]) == 2:
        keypoints = [kp + [1.0] for kp in keypoints]  # score를 1.0으로 기본 설정
    elif keypoint_scores is not None:
        keypoints = [kp + [score] for kp, score in zip(keypoints, keypoint_scores)]

    # COCO 포맷 관절 연결 정보 (스켈레톤)
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 머리와 눈
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체와 팔
        (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)  # 하체
    ]

    # 관절 점 그리기
    for i, (x, y, score) in enumerate(keypoints):
        if score > kpt_score_thr:  # 신뢰도 임계값 적용
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 관절 점

    # 관절 연결선 그리기
    for i, j in skeleton:
        if keypoints[i][2] > kpt_score_thr and keypoints[j][2] > kpt_score_thr:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # 관절 연결선

    # 결과 이미지 저장
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

def process_images(input_dir, output_dir, model, vis_dir=None):
    """
    디렉토리 내 이미지 처리 및 시각화.

    input_dir (str): 입력 이미지 디렉토리.
    output_dir (str): JSON 및 .pkl 결과 저장 디렉토리.
    model: MMPose 모델.
    vis_dir (str): 시각화된 이미지 저장 디렉토리. None일 경우 시각화하지 않음.
    save_pkl (bool): .pkl 파일 저장 여부.
    """
    # 결과 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # visualizations 디렉토리 생성 
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # raw_data 디렉토리에서 이미지 파일 목록 가져오기 (자연 정렬 적용)
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')],
        key=natural_key
    )

    all_results = []  # 모든 프레임의 결과를 저장할 리스트

    # 각 이미지 처리
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        results = inference_topdown(model, image_path)

        if len(results) == 0 or len(results[0].pred_instances.keypoints) == 0:
            print(f"No keypoints detected in {image_path}")
            continue

        # 키포인트와 신뢰도 추출
        keypoints = results[0].pred_instances.keypoints[0].tolist()  # numpy -> list
        keypoint_scores = results[0].pred_instances.keypoint_scores[0].tolist()

        # JSON 데이터 생성
        json_keypoints = [[float(kp[0]), float(kp[1]), float(score)] for kp, score in zip(keypoints, keypoint_scores)]

        # JSON 파일 저장
        json_file = os.path.join(output_dir, image_file.replace('.jpg', '.json').replace('.png', '.json'))
        with open(json_file, 'w') as f:
            json.dump({"keypoints": json_keypoints}, f, indent=4)

        # 결과를 리스트에 추가
        all_results.append(results)

        # 시각화 및 저장
        if vis_dir:
            vis_path = os.path.join(vis_dir, image_file)
            visualize_keypoints(image_path, keypoints, keypoint_scores, output_path=vis_path)

        print(f"Processed: {image_path} -> JSON: {json_file}")

# 실행
input_dir = "raw_data"  # 입력 이미지 디렉토리

output_dir = "keypoints"  # JSON 및 .pkl 결과 저장 디렉토리
vis_dir = "visualizations"  # keypoints 시각화된 이미지 저장 디렉토리

process_images(input_dir, output_dir, model, vis_dir=vis_dir)





# image_path = 'raw_data/test2.jpg'
# results = inference_topdown(model, image_path)


# # print(results)
# #스켈레톤 추출 데이터 저장
# pklFile_outdir = 'convert/convert/test1_results.pkl' 
# with open(pklFile_outdir, 'wb') as f:
#     pickle.dump(results, f)
# print("저장완료 ${pklFile_outdir}")


# print("mmpose 모델 실행 완료")
# # print(f"Result saved to {output_image_path}")
