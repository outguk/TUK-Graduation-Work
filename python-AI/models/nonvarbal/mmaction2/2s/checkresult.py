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
wrist_distance_threshold = 100  # 손목 간 거리 임계값 (130픽셀 이하면 뒷짐으로 간주)

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
    # 전체 세그먼트 목록이 있으면(정상+비정상), 그걸 그대로 반환
    if "all_segments" in keypoints_data:
        return keypoints_data["all_segments"]
    # 그렇지 않으면 기존 annotations 기반
    return [ann["frame_dir"] for ann in keypoints_data["annotations"]]

def check_wrist_distance(keypoints_data, frame_dir):
    """손목 간 거리 확인하여 뒷짐 자세 여부 판단"""
    # 먼저 어노테이션에서 찾기
    for annotation in keypoints_data["annotations"]:
        if annotation["frame_dir"] == frame_dir:
            keypoints = annotation["keypoint"][0]  # 첫 번째 프레임의 키포인트
            
            # 손목 좌표 추출
            left_wrist = keypoints[0][9][:2]    # 왼쪽 손목
            right_wrist = keypoints[0][10][:2]  # 오른쪽 손목
            
            # 손목 사이의 거리 계산
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            
            return {
                "wrist_distance": float(wrist_distance),
                "is_hands_behind": wrist_distance < wrist_distance_threshold
            }
    
    # 어노테이션에서 찾지 못하면 개별 JSON 파일에서 찾기
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "../data", "keypoints", f"{frame_dir}.json"
    )
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            keypoints_list = data.get("keypoints", [])
            if keypoints_list and len(keypoints_list) >= 17:
                # 왼쪽 손목(9), 오른쪽 손목(10)
                left_wrist = np.array(keypoints_list[9][:2])
                right_wrist = np.array(keypoints_list[10][:2])
                
                wrist_distance = np.linalg.norm(left_wrist - right_wrist)
                is_hands_behind = wrist_distance < wrist_distance_threshold
                
                print(f"JSON에서 {frame_dir} 손목거리: {wrist_distance}, 뒷짐: {is_hands_behind}")
                
                return {
                    "wrist_distance": float(wrist_distance),
                    "is_hands_behind": is_hands_behind
                }
        except Exception as e:
            print(f"JSON 파일 처리 오류: {e}")
            pass
    
    return {"wrist_distance": None, "is_hands_behind": False}

def process_results(results, used_frame_dirs, keypoints_data):
    print("=== MAPPING DEBUG ===")
    stgcn_dirs = keypoints_data["split"]["xsub_val"]
    pred_map   = {frame: results[i] for i, frame in enumerate(stgcn_dirs)}

    json_results = []
    previous_frame = None
    sample_idx = 1  # 샘플 번호 (정상 행동 포함)
    
    # 뒷짐 자세 추적
    last_hands_behind_frame = None
    hands_behind_count = 0

    # 먼저 모든 샘플의 기본 정보 생성
    initial_samples = []
    
        # ── frame_dir 기준으로 예측+정상구간 매핑 ──
    for frame_dir in used_frame_dirs:
        print(f"{frame_dir}: {'PREDICTED' if frame_dir in pred_map else 'NORMAL'}")

        if frame_dir in pred_map:
            # STGCN 결과가 있는 구간
            res = pred_map[frame_dir]
            probs = np.array(res['pred_score'])
            top3_idx     = np.argsort(probs)[-3:][::-1]
            top3_probs   = probs[top3_idx]
            top3_classes = [class_labels[j] for j in top3_idx]
            is_normal = bool(top3_probs[0] < threshold)
        else:
            # 예측 없는 구간 → 정상 처리
            top3_probs   = []
            top3_classes = []
            is_normal    = True

        # 시간 계산 (2초 단위)
        idx         = used_frame_dirs.index(frame_dir)
        segment_dur = frames_per_annotation / fps  # e.g. 10/5 = 2.0
        start_time  = idx * segment_dur
        end_time    = (idx + 1) * segment_dur

        # 손목 체크
        wrist = check_wrist_distance(keypoints_data, frame_dir)

        # 샘플 생성
        sample_data = {
            "sample_number": sample_idx,
            "time_range":    f"{start_time:.2f}s ~ {end_time:.2f}s",
            "frame_range":   f"{frame_dir.split('_')[-1]} ~ {int(frame_dir.split('_')[-1]) + frames_per_annotation - 1}",
            "frame_dir":     frame_dir,
            "is_normal":     is_normal,
        }
        if wrist.get("wrist_distance") is not None:
            sample_data["wrist_distance"] = float(wrist["wrist_distance"])

        if not is_normal:
            sample_data["top_classes"] = [
                {"class": top3_classes[k],
                "probability": float(round(float(top3_probs[k]) * 100, 2))}
                for k in range(len(top3_classes))
            ]

        initial_samples.append(sample_data)
        sample_idx += 1

    print(f"초기 샘플 개수: {len(initial_samples)}")
    
    # 후처리: 정상으로 분류된 샘플들 중 뒷짐 상태가 계속되는 샘플 재분류
    processed_samples = []
    last_hands_behind_idx = -1
    
    # 뒷짐 샘플 위치 파악
    hands_behind_indices = []
    for i, sample in enumerate(initial_samples):
        # 이미 뒷짐으로 감지된 경우
        if (not sample["is_normal"] and 
            any(c.get("class") == "팔동작(뒷짐)" for c in sample.get("top_classes", []))):
            hands_behind_indices.append(i)
        # 손목 거리가 임계값보다 작은 경우
        elif ("wrist_distance" in sample and 
              sample["wrist_distance"] is not None and 
              sample["wrist_distance"] < 115):
            hands_behind_indices.append(i)
    
    print(f"뒷짐으로 감지된 샘플 인덱스: {hands_behind_indices}")
    
    for i, sample in enumerate(initial_samples):
        current_sample = sample.copy()
        
        # 이미 뒷짐으로 감지된 경우
        if (not current_sample["is_normal"] and 
            any(c.get("class") == "팔동작(뒷짐)" for c in current_sample.get("top_classes", []))):
            last_hands_behind_idx = i
        
        # 정상인데 손목 거리가 작은 경우 -> 재분류
        elif (current_sample["is_normal"] and 
              "wrist_distance" in current_sample and 
              current_sample["wrist_distance"] is not None and
              current_sample["wrist_distance"] < 115):
            current_sample["is_normal"] = False
            current_sample["reclassified"] = True
            current_sample["top_classes"] = [
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0},
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0},
                {"class": "팔동작(뒷짐) - 지속", "probability": 99.0}
            ]
            last_hands_behind_idx = i
            hands_behind_count += 1
        
        # 정상인데, 이전 뒷짐 샘플과 연결성이 있는 경우 -> 재분류
        elif (current_sample["is_normal"] and
              "wrist_distance" in current_sample and
              current_sample["wrist_distance"] is not None and
              last_hands_behind_idx >= 0 and
              i - last_hands_behind_idx <= 1):
            
            prev_sample = processed_samples[last_hands_behind_idx]
            prev_distance = prev_sample.get("wrist_distance", 9999)
            current_distance = current_sample.get("wrist_distance", 9999)
            
            if ((abs(current_distance - prev_distance) < 25 or 
                current_distance < prev_distance) and current_distance < 120):
                current_sample["is_normal"] = False
                current_sample["reclassified"] = True
                current_sample["continued_from_previous"] = True
                current_sample["top_classes"] = [
                    {"class": "팔동작(뒷짐)", "probability": 70.0},
                    {"class": "팔동작(뒷짐)", "probability": 70.0},
                    {"class": "팔동작(뒷짐)", "probability": 70.0}
                ]
                last_hands_behind_idx = i
                hands_behind_count += 1
        
        # 필요없는 필드 제거
        for field in ["frame_num", "is_skipped_segment"]:
            if field in current_sample:
                del current_sample[field]
        
        processed_samples.append(current_sample)
    
    print(f"뒷짐으로 감지된 샘플 인덱스: {hands_behind_indices}")
    print(f"총 {len(processed_samples)}개 샘플 중 {hands_behind_count}개가 뒷짐 자세로 감지/재분류됨")
    
    
    # ── ① 영상 순서대로 재정렬 & 정상 누락 채우기 ──
    final_results = []
    next_idx = 1
    for frame_dir in used_frame_dirs:
        matched = [s for s in processed_samples if s["frame_dir"] == frame_dir]
        if matched:
            for m in matched:
                m["sample_number"] = next_idx
                final_results.append(m)
                next_idx += 1
        else:
            # 모델에 안 들어간 정상 구간
            num = int(frame_dir.split("_")[-1])
            ft = {
                "sample_number": next_idx,
                "time_range":  f"{num/fps:.2f}s ~ {(num+frames_per_annotation-1)/fps:.2f}s",
                "frame_range": f"{num} ~ {num+frames_per_annotation-1}",
                "frame_dir":   frame_dir,
                "is_normal":   True,
                "top_classes": []
            }
            final_results.append(ft)
            next_idx += 1

    # ── ② 반환 변경 ──
    return final_results








def save_json_results(json_results, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump({"results": json_results}, f, indent=4, ensure_ascii=False)
    print(f"\n JSON 결과 저장 완료: {output_filepath}\n")


def main(result_file, keypoints_pkl_file, output_json_file):
    """
    result_file: ST-GCN++ 모델 추론 결과(.pkl)
    keypoints_pkl_file: JSON→PKL 변환된 키포인트 파일
    output_json_file: 최종 결과를 저장할 JSON 파일 경로
    """
    results = load_pickle_file(result_file)
    print("=== STGCN PREDICTIONS DEBUG ===")
    print(f"Loaded {len(results)} predictions")
    for idx, r in enumerate(results[:5]):
        print(f"  [{idx}] pred_score[:3]={r['pred_score'][:3]}")
    print("...")

    
    
    keypoints_data = load_pickle_file(keypoints_pkl_file)
    used_frame_dirs = get_used_frame_dirs(keypoints_data)
    print("=== FRAME DIRS DEBUG ===")
    print(f"Total frame dirs: {len(used_frame_dirs)}")
    print("First 5 dirs:", used_frame_dirs[:5])
    print("Last 5 dirs: ", used_frame_dirs[-5:])

    # 결과 처리
    json_results = process_results(results, used_frame_dirs, keypoints_data)
    
    # JSON 파일로 저장
    save_json_results(json_results, output_json_file)
    
    # **리스트 형태** 그대로 반환
    return json_results
