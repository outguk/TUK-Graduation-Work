import pickle
import numpy as np

# 결과 파일 경로
result_file = "data/test_full_results.pkl"

# 결과 파일 로드
with open(result_file, "rb") as f:
    results = pickle.load(f)

# 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 설정값
frames_per_annotation = 10  # 한 샘플당 사용된 프레임 수
stride = 5  # 슬라이딩 윈도우 간격
threshold = 0.85  # 정상 행동 필터링 기준 확률 (85%)
time_per_annotation = 2.0  # 2초에 10프레임씩 저장됨
time_per_frame = time_per_annotation / frames_per_annotation  # 1프레임당 걸리는 시간 (0.2초)

for idx, res in enumerate(results):
    probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
    
    # 확률이 가장 높은 클래스 찾기
    top1_index = np.argmax(probs)  # 가장 높은 확률을 가진 클래스 인덱스
    top1_prob = probs[top1_index]  # Top-1 확률
    top1_class = class_labels[top1_index]  # 해당 클래스명
    
    # 슬라이딩 윈도우 적용 시 실제 프레임 번호 계산
    start_frame = idx * stride  # 슬라이딩 윈도우 적용
    end_frame = start_frame + frames_per_annotation - 1  # 샘플이 포함하는 마지막 프레임

    # 실제 시간 계산 (프레임 번호를 기반으로 계산)
    start_time = start_frame * time_per_frame
    end_time = end_frame * time_per_frame
    
    # 85% 미만이면 정상 행동으로 처리
    if top1_prob < threshold:
        top1_class = "정상 행동"
    
    # 출력
    print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}):")
    print(f"   1. {top1_class} ({top1_prob * 100:.2f}%)")
