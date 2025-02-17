import pickle
import numpy as np

# ✅ 결과 파일 경로
result_file = "data/test_results.pkl"
keypoints_pkl_file = "data/keypoints/results.pkl"  # 모델 입력으로 사용된 keypoints 데이터

# ✅ 결과 파일 로드
with open(result_file, "rb") as f:
    results = pickle.load(f)

# ✅ 모델에 입력된 keypoints 데이터 로드 (정상 행동이 제외된 데이터)
with open(keypoints_pkl_file, "rb") as f:
    keypoints_data = pickle.load(f)

# ✅ 모델에 입력된 샘플들의 frame_dir 가져오기
used_frame_dirs = [ann["frame_dir"] for ann in keypoints_data["annotations"]]

# ✅ 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# ✅ 프레임 수 설정
frames_per_annotation = 10  # 각 샘플 당 사용된 프레임 수 (10 프레임)
fps = 5  # 초당 5프레임 (2초당 10프레임)

# ✅ 결과 출력 (정상 행동 제외)
previous_frame = None  # 이전 프레임 번호 저장 변수

for idx, (res, frame_dir) in enumerate(zip(results, used_frame_dirs)):
    probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
    
    # 🔹 확률이 높은 3개의 클래스 찾기
    top3_indices = np.argsort(probs)[-3:][::-1]  # 확률이 높은 순으로 정렬 (내림차순)
    top3_probs = probs[top3_indices]  # 확률 값 가져오기
    top3_classes = [class_labels[i] for i in top3_indices]  # 클래스명 변환
    
    # 🔹 현재 프레임 번호 가져오기 (frame_dir에서 숫자 부분 추출)
    current_frame = int(frame_dir.split('_')[-1])  # 예: frame_40 → 40
    
    # 🔹 정상 행동이 빠진 경우, 중간 프레임을 건너뛰도록 조정
    if previous_frame is not None:
        frame_gap = current_frame - previous_frame  # 이전 프레임과 현재 프레임 차이
        if frame_gap > frames_per_annotation:
            print(f"\n⚠ [정상 행동 구간] {previous_frame + frames_per_annotation} ~ {current_frame - 1} 프레임 건너뜀")
    
    start_frame = current_frame
    end_frame = start_frame + frames_per_annotation - 1

    # 🔹 **시간 계산 (프레임 → 초 변환)**
    start_time = start_frame / fps
    end_time = end_frame / fps

    # ✅ 출력
    print(f"\n🔹 샘플 {idx + 1} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}, frame_dir: {frame_dir}):")
    for i in range(3):
        print(f"   {i+1}. {top3_classes[i]} ({top3_probs[i] * 100:.2f}%)")
    
    # 🔹 현재 프레임을 previous_frame으로 업데이트
    previous_frame = current_frame
