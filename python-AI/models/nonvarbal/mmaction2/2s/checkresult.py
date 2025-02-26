import pickle
import numpy as np
import json

# 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 프레임 수 설정
frames_per_annotation = 10  # 각 샘플 당 사용된 프레임 수 (10 프레임)
fps = 5  # 초당 5프레임 (2초당 10프레임)
threshold = 0.85  # 정상 행동 필터링 기준 확률 (85%)

def load_pickle_file(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def get_used_frame_dirs(keypoints_data):
    return [ann["frame_dir"] for ann in keypoints_data["annotations"]]

def process_results(results, used_frame_dirs):
    json_results = []
    previous_frame = None
    sample_idx = 1  # 샘플 번호 (정상 행동 포함)

    for res, frame_dir in zip(results, used_frame_dirs):
        probs = np.array(res['pred_score'])  # 확률 리스트 (NumPy 배열 변환)
        
        # 확률이 높은 3개의 클래스 찾기
        top3_indices = np.argsort(probs)[-3:][::-1]  # 확률이 높은 순으로 정렬 (내림차순)
        top3_probs = probs[top3_indices]  # 확률 값 가져오기
        top3_classes = [class_labels[i] for i in top3_indices]  # 클래스명 변환
        
        top1_prob = top3_probs[0]  # 가장 높은 확률 값
        is_normal = bool(top1_prob < threshold)  # 정상 행동 여부

        # 현재 프레임 번호 가져오기 (frame_dir에서 숫자 부분 추출)
        current_frame = int(frame_dir.split('_')[-1])  # 예: frame_40 → 40
        
        # 정상 행동이 빠진 경우, 중간 프레임을 건너뛰도록 JSON에 추가
        if previous_frame is not None:
            frame_gap = current_frame - previous_frame  # 이전 프레임과 현재 프레임 차이
            if frame_gap > frames_per_annotation:
                skipped_start = previous_frame + frames_per_annotation
                skipped_end = current_frame - 1
                skipped_start_time = skipped_start / fps
                skipped_end_time = skipped_end / fps
                
                # 정상 행동 구간 추가
                json_results.append({
                    "sample_number": sample_idx,
                    "time_range": f"{skipped_start_time:.2f}s ~ {skipped_end_time:.2f}s",
                    "frame_range": f"{skipped_start} ~ {skipped_end}",
                    "frame_dir": f"frame_{skipped_start}",
                    "is_normal": True
                })

                # print(f"\n⚠ [정상 행동 구간] {skipped_start_time:.2f}s ~ {skipped_end_time:.2f}s (프레임: {skipped_start} ~ {skipped_end})")
                sample_idx += 1

        start_frame = current_frame
        end_frame = start_frame + frames_per_annotation - 1

        # 시간 계산 (프레임 → 초 변환)
        start_time = start_frame / fps
        end_time = end_frame / fps

        # 터미널 출력
        # print(f"\n샘플 {sample_idx} (영상 구간: {start_time:.2f}s ~ {end_time:.2f}s, 프레임: {start_frame} ~ {end_frame}, frame_dir: {frame_dir}):")
        # if is_normal:
        #     print("   정상 행동")
        # else:
        #     for i in range(3):
        #         print(f"   {i+1}. {top3_classes[i]} ({top3_probs[i] * 100:.2f}%)")

        # JSON 데이터 저장
        sample_data = {
            "sample_number": sample_idx,
            "time_range": f"{start_time:.2f}s ~ {end_time:.2f}s",
            "frame_range": f"{start_frame} ~ {end_frame}",
            "frame_dir": frame_dir,
            "is_normal": is_normal
        }
        
        if not is_normal:
            sample_data["top_classes"] = [
                {"class": top3_classes[i], "probability": round(top3_probs[i] * 100, 2)}
                for i in range(3)
            ]

        json_results.append(sample_data)

        # 샘플 번호 증가
        sample_idx += 1

        # 🔹 현재 프레임을 previous_frame으로 업데이트
        previous_frame = current_frame

    return json_results

def save_json_results(json_results, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump({"results": json_results}, f, indent=4, ensure_ascii=False)
    print(f"\n JSON 결과 저장 완료: {output_filepath}\n")

def main(result_file, keypoints_pkl_file, output_json_file):
    results = load_pickle_file(result_file)
    keypoints_data = load_pickle_file(keypoints_pkl_file)
    used_frame_dirs = get_used_frame_dirs(keypoints_data)
    json_results = process_results(results, used_frame_dirs)

    #json 파일 저장장
    save_json_results(json_results, output_json_file)
    
    # Python 딕셔너리 형식으로 변환
    dict_results = {res["sample_number"]: res for res in json_results}

    
    # print(json_results)
    
    # Python 딕셔너리 반환
    return dict_results  
