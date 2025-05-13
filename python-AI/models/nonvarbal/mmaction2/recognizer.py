import pickle
from collections import Counter
from mmaction.apis import init_recognizer, inference_skeleton
import mmcv

# 모델 구성 파일과 체크포인트 파일 경로
config_file = 'configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py'
checkpoint_file = 'checkpoints/best_acc_top1_epoch_27.pth'

# 모델 초기화
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

# 스켈레톤 데이터 파일 경로
pkl_file = 'mmpose/keypoints/all_results.pkl'

# 스켈레톤 데이터 로드
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# 데이터 형식 변환
pose_results = []
for result in data:
    for instance in result[0].pred_instances:
        keypoints = instance.keypoints
        keypoint_scores = instance.keypoint_scores
        pose_results.append({
            'keypoints': keypoints,
            'keypoint_scores': keypoint_scores
        })

for i, pose_result in enumerate(pose_results[:5]):  # 첫 5개 데이터만 확인
    print(f"Pose Result {i}:")
    print(pose_result)


# 이미지 크기 설정 (예: 256x256)
img_shape = (640, 480)

# 클래스 레이블 정보
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 각 프레임별로 예측 수행 및 결과 저장
frame_results = []
for pose_result in pose_results:
    result = inference_skeleton(model, [pose_result], img_shape)
    frame_results.append(result.pred_label.item())

# 각 행동의 빈도 계산
action_counts = Counter(frame_results)

# 결과 출력
print('Action Counts:', action_counts)

# 결과를 텍스트 파일로 저장
output_file = 'output_results.txt'
with open(output_file, 'w') as f:
    for action, count in action_counts.items():
        action_label = class_labels[action]
        f.write(f'Action {action} ({action_label}): {count}\n')

print(f'Results saved to {output_file}')
