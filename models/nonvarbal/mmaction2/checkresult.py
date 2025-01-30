# import pickle

# # 새로운 결과 파일 경로
# result_file = 'work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/test_full_results.pkl'

# # 결과 파일 로드
# with open(result_file, 'rb') as f:
#     results = pickle.load(f)

# # 결과 구조 확인
# print("Type of results:", type(results))
# if isinstance(results, list):
#     print("Length of results:", len(results))
#     if len(results) > 0:
#         print("First result:", results[0])
# else:
#     print("Content of results:", results)


import pickle
from collections import Counter

# 결과 파일 경로
result_file = 'work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/test_full_results.pkl'

# 결과 파일 로드
with open(result_file, 'rb') as f:
    results = pickle.load(f)

# 예측된 행동 레이블 수집
predicted_labels = [res['pred_label'].item() for res in results]  # tensor -> int 변환

# 각 행동의 빈도 계산
action_counts = Counter(predicted_labels)

# 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 결과 출력
print("Action Counts:")
for label, count in action_counts.items():
    action_name = class_labels[label]
    print(f"Action {label} ({action_name}): {count} times")

