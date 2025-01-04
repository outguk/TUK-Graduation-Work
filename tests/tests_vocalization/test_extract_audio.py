import os
import sys
import pytest

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from models.vocalization.vocalization_analysis import extract_audio

def test_extract_audio():
    # 1. 테스트용 동영상 파일 경로
    video_file = "tests/test_video/test1.mp4"  # 테스트용 동영상 (다시 가져와야 함)
    output_audio_file = "tests/test_video/test1_audio.wav"  # 출력 오디오 경로

    # 2. 함수 실행
    result = extract_audio(video_file, output_audio_file)

    # 3. 결과 검증
    # (1) 반환된 경로가 올바른지 확인
    assert result == output_audio_file, "Returned file path is incorrect."

    # (2) 출력 파일이 실제로 생성되었는지 확인
    assert os.path.exists(output_audio_file), "Audio file was not created."

    # (3) 생성된 파일이 비어 있지 않은지 확인
    assert os.path.getsize(output_audio_file) > 0, "Audio file is empty."

    # 4. 테스트 완료 후 파일 정리
    # os.remove(output_audio_file)


# 테스트 성공 (wav파일 -> 다른 파일로 하는 것이 좋은 지 유의의)