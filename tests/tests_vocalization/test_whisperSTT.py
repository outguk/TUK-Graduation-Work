import os
import sys
import pytest
from unittest.mock import patch

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 디버깅 출력
print("Correct project root added to sys.path:", project_root)
print("sys.path:", sys.path)

# 필요한 모듈 임포트
from models.vocalization.vocalization_analysis import transcribe_audio

def test_transcribe_audio():
    # 테스트용 동영상 파일 경로 및 출력 텍스트 파일 경로
    video_file = os.path.abspath("tests/test_video/test1_audio.wav")  # 실제 테스트용 파일 필요

    # Step 1: 동영상 파일 확인
    assert os.path.exists(video_file), f"Test video file does not exist: {video_file}"

    # Step 2: 함수 실행 및 텍스트 추출
    transcription = transcribe_audio(video_file, model_name="base")

    # Step 3: 결과 검증
    # (1) 반환된 텍스트가 비어 있지 않은지 확인
    assert transcription.strip(), "Transcription is empty."

    # (2) Whisper가 반환한 텍스트의 일부를 출력 (검증을 위한 참조)
    print("Transcribed Text:", transcription['text'][:100])

    # 4. 테스트 완료 후 파일 정리
    # os.remove(transcription)s