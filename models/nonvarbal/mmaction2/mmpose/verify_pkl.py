import pickle

# 변환된 PKL 파일 경로
pkl_path = 'keypoints/results.pkl'

# PKL 파일 로드
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 데이터 형식 출력
print("Keypoints Shape:", data["keypoint"].shape)  # 예상: [1, T, V, C]
print("Keypoint Scores Shape:", data["keypoint_score"].shape)  # 예상: [1, T, V]
