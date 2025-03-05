import pickle
import numpy as np
import json
import os


# 클래스 레이블 정의
class_labels = [
    "손동작(머리)", "손동작(얼굴)", "손동작(몸긁기)", "손동작(손톱)", "머리동작(고개흔들기)",
    "머리동작(좌우흔들기)", "머리동작(숙이기)", "팔동작(뒷짐)", "팔동작(무의미반동)", "자세(좌우흔들기)",
    "자세(비스듬히)", "자세(비비꼬기)"
]

# 프레임 수 설정
frames_per_annotation = 10  # 각 샘플 당 사용된 프레임 수 (10 프레임)
fps = 5  # 초당 5프레임 (2초당 10프레임)
threshold = 0.50  # 정상 행동 필터링 기준 확률 (50%)

# 뒷짐 감지 관련 설정
wrist_distance_threshold =100  # 손목 간 거리 임계값 (130픽셀 이하면 뒷짐으로 간주)

def load_pickle_file(filepath):
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

def get_used_frame_dirs(keypoints_data):
    return [ann["frame_dir"] for ann in keypoints_data["annotations"]]

def check_wrist_distance(keypoints_data, frame_dir):
    """손목 간 거리 확인하여 뒷짐 자세 여부 판단"""
    # 먼저 어노테이션에서 찾기 시도
    for annotation in keypoints_data["annotations"]:
        if annotation["frame_dir"] == frame_dir:
            keypoints = annotation["keypoint"][0]  # 첫 번째 프레임의 키포인트
            
            # 손목 좌표 추출
            left_wrist = keypoints[0][9][:2]   # 왼쪽 손목
            right_wrist = keypoints[0][10][:2]  # 오른쪽 손목
            
            # 손목 사이의 거리 계산
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            
            # 임계값보다 손목 사이 거리가 작으면 뒷짐으로 간주
            is_hands_behind = wrist_distance < wrist_distance_threshold
            
            return {
                "wrist_distance": float(wrist_distance),
                "is_hands_behind": is_hands_behind
            }
    
    # 어노테이션에서 찾지 못하면 개별 JSON 파일에서 찾기
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../data", "keypoints", f"{frame_dir}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # JSON 파일에서 키포인트 추출
            keypoints_list = data.get("keypoints", [])
            if keypoints_list and len(keypoints_list) >= 17:  # 최소 17개의 키포인트가 필요
                # 9번(왼쪽 손목)과 10번(오른쪽 손목) 키포인트
                left_wrist = np.array(keypoints_list[9][:2])
                right_wrist = np.array(keypoints_list[10][:2])
                
                # 손목 사이의 거리 계산
                wrist_distance = np.linalg.norm(left_wrist - right_wrist)
                
                # 임계값보다 손목 사이 거리가 작으면 뒷짐으로 간주
                is_hands_behind = wrist_distance < wrist_distance_threshold
                
                print(f"JSON 파일에서 {frame_dir}의 손목 거리: {wrist_distance}, 뒷짐 여부부 {is_hands_behind}")
                
                return {
                    "wrist_distance": float(wrist_distance),
                    "is_hands_behind": is_hands_behind
                }
        except Exception as e:
            print(f"JSON 파일 처리 오류: {e}")
            pass
    
    return {"wrist_distance": None, "is_hands_behind": False}

def process_results(results, used_frame_dirs, keypoints_data):
    json_results = []
    previous_frame = None
    sample_idx = 1  # 샘플 번호 (정상 행동 포함)
    
    # 뒷짐 자세 추적
    last_hands_behind_frame = None  # 마지막으로 뒷짐이 감지된 프레임
    hands_behind_count = 0  # 뒷짐으로 감지/재분류된 샘플 수

    # 먼저 모든 샘플의 기본 정보 생성
    initial_samples = []
    
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
        
        # 손목 간 거리 확인
        wrist_check = check_wrist_distance(keypoints_data, frame_dir)
        
        # 정상 행동이 빠진 경우, 중간 프레임을 일정한 크기의 구간으로 분할하여 처리
        if previous_frame is not None:
            frame_gap = current_frame - previous_frame  # 이전 프레임과 현재 프레임 차이
            if frame_gap > frames_per_annotation:
                skipped_start = previous_frame + frames_per_annotation
                skipped_end = current_frame - 1
                
                # 스킵된 구간을 frames_per_annotation 크기의 여러 구간으로 분할
                for segment_start in range(skipped_start, skipped_end + 1, frames_per_annotation):
                    segment_end = min(segment_start + frames_per_annotation - 1, skipped_end)
                    segment_start_time = segment_start / fps
                    segment_end_time = segment_end / fps
                    
                    # 세그먼트 프레임 디렉토리 이름
                    segment_frame_dir = f"frame_{segment_start}"
                    
                    # 세그먼트의 손목 거리 확인
                    segment_wrist_check = check_wrist_distance(keypoints_data, segment_frame_dir)
                    
                    # 기본적으로는 정상으로 설정하고, 후처리 단계에서 필요시 재분류
                    segment_data = {
                        "sample_number": sample_idx,
                        "time_range": f"{segment_start_time:.2f}s ~ {segment_end_time:.2f}s",
                        "frame_range": f"{segment_start} ~ {segment_end}",
                        "frame_dir": segment_frame_dir,
                        "is_normal": True,
                        "frame_num": segment_start,  # 프레임 번호 추가 (후처리용)
                        "is_skipped_segment": True   # 스킵 구간 표시
                    }
                    
                    # 손목 거리 정보 추가
                    if segment_wrist_check["wrist_distance"] is not None:
                        segment_data["wrist_distance"] = segment_wrist_check["wrist_distance"]
                    
                    initial_samples.append(segment_data)
                    sample_idx += 1

        start_frame = current_frame
        end_frame = start_frame + frames_per_annotation - 1

        # 시간 계산 (프레임 → 초 변환)
        start_time = start_frame / fps
        end_time = end_frame / fps

        # 기본 JSON 데이터 생성 (후처리용)
        sample_data = {
            "sample_number": sample_idx,
            "time_range": f"{start_time:.2f}s ~ {end_time:.2f}s",
            "frame_range": f"{start_frame} ~ {end_frame}",
            "frame_dir": frame_dir,
            "is_normal": is_normal,
            "frame_num": current_frame,  # 프레임 번호 추가 (후처리용)
            "is_skipped_segment": False  # 모델 예측 샘플 표시
        }
        
        if wrist_check["wrist_distance"] is not None:
            sample_data["wrist_distance"] = wrist_check["wrist_distance"]
        
        if not is_normal:
            sample_data["top_classes"] = [
                {"class": top3_classes[i], "probability": round(float(top3_probs[i]) * 100, 2)}
                for i in range(3)
            ]
            
            # 팔동작(뒷짐)이 감지된 경우 표시
            if any(c == "팔동작(뒷짐)" for c in top3_classes[:1]):
                sample_data["is_hands_behind_detected"] = True
                last_hands_behind_frame = current_frame
                hands_behind_count += 1

        initial_samples.append(sample_data)
        sample_idx += 1

        # 현재 프레임을 previous_frame으로 업데이트
        previous_frame = current_frame

    print(f"초기 샘플 개수: {len(initial_samples)}")
    
    # 후처리: 정상으로 분류된 샘플들 중 뒷짐 상태가 계속되는 샘플 재분류
    processed_samples = []
    last_hands_behind_idx = -1
    
    # 모든 뒷짐 샘플 위치 미리 파악
    hands_behind_indices = []
    for i, sample in enumerate(initial_samples):
        # 이미 뒷짐으로 감지된 경우
        if not sample["is_normal"] and any(c.get("class") == "팔동작(뒷짐)" 
                                          for c in sample.get("top_classes", []) if c.get("class")):
            hands_behind_indices.append(i)
        # 손목 거리가 임계값보다 작은 경우
        elif "wrist_distance" in sample and sample["wrist_distance"] is not None and sample["wrist_distance"] < 115:
            # 임계값을 115로 낮춰서 확실한 뒷짐만 감지
            hands_behind_indices.append(i)
    
    print(f"뒷짐으로 감지된 샘플 인덱스: {hands_behind_indices}")
    
    # 초기 샘플들을 순회하며 뒷짐 상태를 전파
    for i, sample in enumerate(initial_samples):
        current_sample = sample.copy()  # 복사본 사용
        
        # 이미 뒷짐으로 감지된 경우 (모델이 직접 감지)
        if not current_sample["is_normal"] and any(c.get("class") == "팔동작(뒷짐)" 
                                                 for c in current_sample.get("top_classes", []) if c.get("class")):
            # 인덱스 업데이트
            last_hands_behind_idx = i
        
        # 정상으로 분류되었지만 손목 거리가 임계값보다 작은 경우
        elif (current_sample["is_normal"] and 
              "wrist_distance" in current_sample and 
              current_sample["wrist_distance"] is not None and 
              current_sample["wrist_distance"] < 115):  # 뒷짐 임계값 더 엄격하게
            
            # 뒷짐으로 재분류
            current_sample["is_normal"] = False
            current_sample["reclassified"] = True
            current_sample["top_classes"] = [
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0},
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0},
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0}
            ]
            last_hands_behind_idx = i
            hands_behind_count += 1
            
        # 정상으로 분류된 샘플 + 손목 거리 있음 + 마지막 뒷짐 샘플플과 가까움
        elif (current_sample["is_normal"] and
              "wrist_distance" in current_sample and
              current_sample["wrist_distance"] is not None and
              last_hands_behind_idx >= 0 and
              i - last_hands_behind_idx <= 1):  
            
            # 마지막 뒷짐 샘플과 손목 거리 비교 
            prev_sample = processed_samples[last_hands_behind_idx]
            prev_distance = prev_sample.get("wrist_distance", 9999)
            current_distance = current_sample.get("wrist_distance", 9999)
            
            # 손목 거리가 비슷하거나 현재 손목 거리가 더 작으면 뒷짐 상태 계속됨
            if ((abs(current_distance - prev_distance) < 25 or 
                current_distance < prev_distance) and current_distance < 120):
                
                # 뒷짐 상태가 계속되는 것으로 판단하고 재분류
                current_sample["is_normal"] = False
                current_sample["reclassified"] = True
                current_sample["continued_from_previous"] = True
                current_sample["top_classes"] = [
                    {"class": "팔동작(뒷짐)", "probability": 70.0},
                    {"class": "팔동작(뒷짐)", "probability": 70.0},
                    {"class": "팔동작(뒷짐)", "probability": 70.0}
                ]
                last_hands_behind_idx = i  # 이 샘플도 뒷짐 상태로 업데이트
                hands_behind_count += 1
                
        # 필요없는 필드 제거
        for field in ["frame_num", "is_skipped_segment"]:
            if field in current_sample:
                del current_sample[field]
            
        processed_samples.append(current_sample)
    
    print(f"뒷짐으로 감지된 샘플 인덱스: {hands_behind_indices}")
    print(f"총 {len(processed_samples)}개 샘플 중 {hands_behind_count}개가 뒷짐 자세로 감지/재분류됨")
    
    return processed_samples

def save_json_results(json_results, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump({"results": json_results}, f, indent=4, ensure_ascii=False)
    print(f"\n JSON 결과 저장 완료: {output_filepath}\n")

# def main(result_file, keypoints_pkl_file, output_json_file):
#     results = load_pickle_file(result_file)
#     keypoints_data = load_pickle_file(keypoints_pkl_file)
#     used_frame_dirs = get_used_frame_dirs(keypoints_data)
#     json_results = process_results(results, used_frame_dirs, keypoints_data)
#     print(f"최종 결과에는 {len(json_results)}개 샘플이 포함되었습니다.")

#     # json 파일 저장
#     save_json_results(json_results, output_json_file)
    
#     # Python 딕셔너리 형식으로 변환
#     dict_results = {res["sample_number"]: res for res in json_results}
    
#     # Python 딕셔너리 반환
#     return dict_results

def main(result_file, keypoints_pkl_file, output_json_file):
        
    results = load_pickle_file(result_file)
    keypoints_data = load_pickle_file(keypoints_pkl_file)
    used_frame_dirs = get_used_frame_dirs(keypoints_data)
    
    # 결과 처리
    json_results = process_results(results, used_frame_dirs, keypoints_data)
    
    # json 파일 저장
    save_json_results(json_results, output_json_file)
    
    # Python 딕셔너리 형식으로 변환
    dict_results = {sample["sample_number"]: sample for sample in json_results}
    
    return dict_results