import os
import sys
import pytest

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from models.vocalization.util_functions import get_sampling_rate, change_sampling_rate

def test_extract_audio():
    # 1. 테스트용 동영상 파일 경로
    audio_path = os.path.abspath("tests/test_video/test1_audio.wav")  # 전처리할 오디오 경로

    # 2. 함수 실행
    result = get_sampling_rate(audio_path)

def test_change_sampling_rate():
    # 1. 테스트용 동영상 파일 경로, 변경 파일이 저장될 경로로
    audio_path = os.path.abspath("tests/test_video/test1_audio.wav")
    pre_audio_path = os.path.abspath("tests/test_video/test1_pre_audio.wav")

    # 2. 함수 실행
    change_sampling_rate(audio_path, 16000, pre_audio_path)

    # 3. 변경 결과 검증
    get_sampling_rate(audio_path) # 경로와 이름 같으면 덮어쓰기 됨

    