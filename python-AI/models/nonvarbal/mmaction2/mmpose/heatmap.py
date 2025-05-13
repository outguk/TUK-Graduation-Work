import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmengine.config import Config
from mmpose.registry import TRANSFORMS  # 🔹 MMPose 변환 레지스트리 불러오기

# 🔹 MMPose 모델 로드 (설정 수정 포함)
register_all_modules()
config_path = "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py"
checkpoint_path = "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

cfg = Config.fromfile(config_path)

# 🔹 GetHeatmap을 등록 (직접 추가)
if "GetHeatmap" not in TRANSFORMS.module_dict:
    from mmpose.datasets.transforms import GenerateTarget
    TRANSFORMS.register_module(name="GetHeatmap", module=GenerateTarget)

# 🔹 히트맵을 활성화하도록 pipeline 설정 변경
cfg.test_dataloader.dataset.pipeline.append(dict(type="GetHeatmap"))

# 🔹 모델 초기화 (변경된 설정 적용)
model = init_model(cfg, checkpoint_path, device="cuda:0")

def visualize_heatmap(image_path):
    """MMPose의 히트맵을 시각화하는 함수"""
    img = cv2.imread(image_path)

    # 🔹 포즈 추론 실행
    results = inference_topdown(model, img)

    if len(results) == 0 or len(results[0].pred_instances.keypoints) == 0:
        print(f"❌ 키포인트 탐지 실패: {image_path}")
        return

    # 🔹 pred_fields에서 히트맵 가져오기
    if hasattr(results[0], "pred_fields"):
        print("✅ pred_fields 존재")
        print("📌 사용 가능한 필드 목록:", results[0].pred_fields.keys())

        if "heatmaps" in results[0].pred_fields:
            print("✅ heatmaps 존재")
            heatmaps = results[0].pred_fields["heatmaps"].cpu().numpy()
        else:
            print("❌ heatmaps 없음")
            return
    else:
        print("❌ pred_fields 없음")
        return

    # 🔹 히트맵을 시각적으로 표현하기 위해 Matplotlib 사용
    num_keypoints = heatmaps.shape[0]  # 예: 17개 관절
    fig, axes = plt.subplots(1, num_keypoints, figsize=(20, 4))

    for i in range(num_keypoints):
        ax = axes[i]
        heatmap = heatmaps[i]

        # 🔹 히트맵을 원본 이미지 크기로 Resize
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # 🔹 컬러 히트맵 변환 (OpenCV Jet Color 적용)
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 🔹 원본 이미지와 히트맵 합성
        overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)

        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Keypoint {i}")
        ax.axis("off")

    plt.show()

# 🔹 테스트 실행
visualize_heatmap("frame_10.jpg")
