# import pickle

# with open('keypoints/all_results.pkl', 'rb') as f:
#     inference_data = pickle.load(f)

# # 데이터 구조 확인
# print(type(inference_data))
# print(len(inference_data))
# print(inference_data[0])
import numpy as np
import pickle

# COCO 포맷을 STGCN 포맷으로 변환
def convert_to_stgcn_format(keypoints):
    COCO_TO_STGCN = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]
    return keypoints[COCO_TO_STGCN]

# 좌표 정규화 함수
def normalize_keypoints(keypoints, img_shape):
    h, w = img_shape
    keypoints[:, 0] /= w  # x 좌표 정규화
    keypoints[:, 1] /= h  # y 좌표 정규화
    return keypoints

# PySkl 입력 형식으로 변환
def convert_skeleton_data(data, img_shape=(1920, 1080), clip_len=100):
    """
    MMPose 데이터에서 PySkl 입력 형식으로 변환
    Args:
        data: MMPose 추론 결과 데이터 (list 형태)
        img_shape: 이미지 해상도
        clip_len: 클립 길이
    Returns:
        keypoints_array: [M x T x V x C] 형태의 키포인트 배열
        scores_array: [M x T x V] 형태의 신뢰도 배열
    """
    keypoints_list = []
    scores_list = []

    for sample in data:
        # 각 sample은 리스트로 감싸져 있으므로, 첫 번째 요소를 가져옵니다.
        sample_data = sample[0]  # 리스트의 첫 번째 요소가 PoseDataSample 객체
        pred_instances = sample_data.pred_instances

        keypoints = pred_instances.keypoints[0]  # [V x C]
        scores = pred_instances.keypoint_scores[0]  # [V]

        # 좌표 정규화 및 포맷 변환
        keypoints = normalize_keypoints(keypoints, img_shape)
        keypoints = convert_to_stgcn_format(keypoints)
        scores = scores[:len(keypoints)]  # 순서에 맞게 자르기

        keypoints_list.append(keypoints)
        scores_list.append(scores)

    # PySkl 입력 형식으로 변환 [M x T x V x C] 및 [M x T x V]
    keypoints_array = np.expand_dims(keypoints_list, axis=0)  # [1 x T x V x C]
    scores_array = np.expand_dims(scores_list, axis=0)       # [1 x T x V]

    # 부족한 프레임 패딩
    if keypoints_array.shape[1] < clip_len:
        pad_len = clip_len - keypoints_array.shape[1]
        keypoints_array = np.pad(keypoints_array, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        scores_array = np.pad(scores_array, ((0, 0), (0, pad_len), (0, 0)))

    return keypoints_array, scores_array

# 데이터 변환 실행
with open('keypoints/all_results.pkl', 'rb') as f:
    data = pickle.load(f)

keypoints_array, scores_array = convert_skeleton_data(data)

# 변환된 데이터 저장
with open('keypoints/converted_skeleton_result.pkl', 'wb') as f:
    pickle.dump({'keypoint': keypoints_array, 'keypoint_score': scores_array}, f)

print("Skeleton data converted and saved to 'converted_skeleton.pkl'.")