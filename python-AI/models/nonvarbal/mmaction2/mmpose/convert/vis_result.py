import pickle
import cv2

# 저장된 추론 결과 로드
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

# 시각화 함수
def visualize_pose(image_path, results, output_path='output.jpg', kpt_score_thr=0.5):
    image = cv2.imread(image_path)
    keypoints = results[0].pred_instances.keypoints[0]
    keypoint_scores = results[0].pred_instances.keypoint_scores[0]

    # COCO 포맷 관절 연결 정보
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for i, (x, y) in enumerate(keypoints):
        if keypoint_scores[i] > kpt_score_thr:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    for i, j in skeleton:
        if keypoint_scores[i] > kpt_score_thr and keypoint_scores[j] > kpt_score_thr:
            pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
            pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")

# 시각화 실행
visualize_pose('demo.jpg', results, output_path='demo_pose_result.jpg')
