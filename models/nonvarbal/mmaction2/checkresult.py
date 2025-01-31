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

## 행동 검출 결과 확인코드 
# import pickle
# from collections import Counter

# # 결과 파일 경로
# result_file = 'work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/test_full_results.pkl'

# # 결과 파일 로드
# with open(result_file, 'rb') as f:
#     results = pickle.load(f)

# # 예측된 행동 레이블 수집
# predicted_labels = [res['pred_label'].item() for res in results]  # tensor -> int 변환

# # 각 행동의 빈도 계산
# action_counts = Counter(predicted_labels)

# # 클래스 레이블 정의
# class_labels = [
#     "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
#     "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
#     "자세(비스듬히)", "자세(비비꼬기)"
# ]

# # 결과 출력
# print("Action Counts:")
# for label, count in action_counts.items():
#     action_name = class_labels[label]
#     print(f"Action {label} ({action_name}): {count} times")

##행동 + 예측 확률 확인 
import pickle
import numpy as np

# 결과 파일 경로
result_file = "work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/test_full_results.pkl"

# 결과 파일 로드
with open(result_file, "rb") as f:
    results = pickle.load(f)

# 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 프레임 수와 FPS 설정
frames_per_sample = 10
fps = 30

for idx, res in enumerate(results):
    probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
    
    # 확률이 높은 3개의 클래스 찾기
    top3_indices = np.argsort(probs)[-3:][::-1]  # 확률이 높은 순으로 정렬 (내림차순)
    top3_probs = probs[top3_indices]  # 확률 값 가져오기
    top3_classes = [class_labels[i] for i in top3_indices]  # 클래스명 변환
    
    # 샘플의 실제 영상 구간 계산
    start_frame = idx * frames_per_sample
    end_frame = start_frame + frames_per_sample - 1
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    # 출력
    print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s):")
    for i in range(3):
        print(f"   {i+1}. {top3_classes[i]} ({top3_probs[i] * 100:.2f}%)")