import cv2
import os
import sys

def video_to_frames(video_path, output_dir, file_format='jpg'):
    """
    영상을 2초 간격, 각 구간에서 10개의 프레임 추출하여 이미지 파일로 저장
    
    Args:
        video_path (str): 비디오 파일 경로.
        output_dir (str): 프레임을 저장할 디렉토리 경로.
        file_format (str): 저장할 이미지 파일 형식 ('jpg' 또는 'png').
    
    Returns:
        int: 저장된 프레임의 총 개수.
    """

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        raise ValueError("❌ ERROR: 비디오 파일 경로가 제공되지 않았습니다.")
    video_path = os.path.abspath(video_path)

    # ✅ 비디오 파일 확인 로그 추가
    print(f"\n📂 [DEBUG] 비디오 파일 경로 확인: {video_path}")

    # 비디오 파일명에서 확장자 제거하여 폴더 이름 생성
    video_output_dir = os.path.abspath(output_dir)
    
    # 저장 디렉토리 생성
    os.makedirs(video_output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)  # 비디오 파일 열기
    if not video.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    print("\n✅ 비디오 파일이 정상적으로 열렸습니다!")

    # 비디오의 FPS(초당 프레임 수) 확인
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"영상의 FPS: {fps}")

    # 2초 동안의 총 프레임 수
    block_duration_frames = int(round(2.0 * fps)) # 2->2.5로 변경 
    
    ## 실제 사용할 데이터는 2초 이상의 영상이기 때문에 예외처리 X
    # if block_duration_frames < 10:
    #     raise ValueError("FPS가 너무 낮아 2초 동안 10프레임을 추출할 수 없습니다.")
    
    # 각 2초 구간 내에서 균등하게 10개의 프레임 인덱스 계산 (0부터 block_duration_frames-1 사이)
    sample_indices = [int(round(i * (block_duration_frames - 1) / 9)) for i in range(10)]
    print("각 2초 구간에서 추출할 프레임 인덱스:", sample_indices)

    frame_count = 0  # 전체 프레임 번호 (영상 내 순번) 
    saved_count = 0  # 저장된 프레임 개수 # 실제 이미지 파일 이름에 들어갈 번호

    success, frame = video.read()  # 첫 번째 프레임 읽기
    while success:
        # 현재 프레임이 속한 2초 블록 내의 인덱스 계산
        current_index_in_block = frame_count % block_duration_frames

        # 샘플링 인덱스에 해당하면 파일 저장 (파일명은 저장된 프레임 번호 사용)
        if current_index_in_block in sample_indices:
            frame_filename = os.path.join(video_output_dir, f"frame_{saved_count}.{file_format}")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        success, frame = video.read()  # 다음 프레임 읽기
        frame_count += 1

    video.release()  # 비디오 객체 해제
    print(f"총 {saved_count}개의 프레임이 {video_output_dir}에 저장되었습니다.")
    return saved_count

# 경로 수정 (통합실행)
if __name__ == "__main__":
    video_path = 'data/videos/test.mp4'  # 비디오 파일 경로
    output_dir = 'data/frames'           # 프레임 저장 경로
    file_format = 'jpg'                      # 저장 파일 형식 ('jpg' 또는 'png')

    saved_frames = video_to_frames(video_path, output_dir, file_format)
    print(f"{saved_frames}개의 프레임이 저장되었습니다.")
