# import pickle
# import numpy as np

# # 결과 파일 경로
# result_file = "data/test_results.pkl"

# # 결과 파일 로드
# with open(result_file, "rb") as f:
#     results = pickle.load(f)

# # 클래스 레이블 정의
# class_labels = [
#     "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
#     "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
#     "자세(비스듬히)", "자세(비비꼬기)"
# ]

# # 설정값
# frames_per_annotation = 10  # 한 샘플당 사용된 프레임 수
# block_duration = 2  # 2초 구간 (각 구간에서 10 프레임 추출됨)
# threshold = 0.85  # 정상 행동 필터링 기준 확률 (85%)
# time_per_frame = block_duration / frames_per_annotation  # 1프레임당 걸리는 시간 (0.2초)

# for idx, res in enumerate(results):
#     probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
    
#     # 확률이 가장 높은 클래스와 두 번째로 높은 클래스 찾기
#     top_indices = np.argsort(probs)[-2:][::-1]  # 확률이 높은 순으로 정렬 (내림차순)
#     top1_index, top2_index = top_indices
#     top1_prob, top2_prob = probs[top1_index], probs[top2_index]
#     top1_class, top2_class = class_labels[top1_index], class_labels[top2_index]
    
#     # 실제 프레임 번호 계산
#     start_frame = idx * frames_per_annotation
#     end_frame = start_frame + frames_per_annotation - 1

#     # 실제 시간 계산 (프레임 번호를 기반으로 계산)
#     start_time = start_frame * time_per_frame
#     end_time = end_frame * time_per_frame
    
#     # 85% 미만이면 정상 행동으로 처리하고, 기존 top1 클래스도 출력
#     if top1_prob < threshold:
#         original_class = top1_class
#         top1_class = f"정상 행동 (원래: {original_class})"
#         # 출력
#         print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}):")
#         print(f"   1. {top1_class} ({top1_prob * 100:.2f}%)")
#         print(f"   2. {top2_class} ({top2_prob * 100:.2f}%)")
#     else:
#         # 출력
#         print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}):")
#         print(f"   1. {top1_class} ({top1_prob * 100:.2f}%)")

##행동 + 예측 확률 확인 
import pickle
import numpy as np

# 결과 파일 경로
result_file = "data/test_results.pkl"

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
block_duration = 2  # 2초 구간 (각 구간에서 10 프레임 추출됨)
frames_per_annotation = 10    # 각 샘플 당 사용된 프레임 수 (예: 10 프레임)

for idx, res in enumerate(results):
    probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
    
    # 확률이 높은 3개의 클래스 찾기
    top3_indices = np.argsort(probs)[-3:][::-1]  # 확률이 높은 순으로 정렬 (내림차순)
    top3_probs = probs[top3_indices]  # 확률 값 가져오기
    top3_classes = [class_labels[i] for i in top3_indices]  # 클래스명 변환
    
    # 구간 시간 계산 (각 샘플이 2초 간격에 해당)
    start_time = idx * block_duration
    end_time = (idx + 1) * block_duration
    # 실제 사용된 프레임 번호 계산 (샘플 0이면 0~9 프레임, 샘플 1이면 10~19 프레임)
    start_frame = idx * frames_per_annotation
    end_frame = start_frame + frames_per_annotation - 1

    # 출력
    print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}):")
    for i in range(3):
        print(f"   {i+1}. {top3_classes[i]} ({top3_probs[i] * 100:.2f}%)")
