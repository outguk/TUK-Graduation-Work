import os
import sys
import pytest
import librosa
import matplotlib.pyplot as plt
import numpy as np

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from models.vocalization.util_functions import get_sampling_rate, change_sampling_rate, band_pass_filter, save_filtered_audio, remove_noise

def test_extract_audio():
    # 1. 테스트용 동영상 파일 경로
    audio_path = os.path.abspath("python-AI/tests/test_video/test1_audio.wav")  # 전처리할 오디오 경로

    # 2. 함수 실행
    get_sampling_rate(audio_path)

def test_change_sampling_rate():
    # 1. 테스트용 동영상 파일 경로, 변경 파일이 저장될 경로
    audio_path = os.path.abspath("python-AI/tests/test_video/test1_audio.wav")
    pre_audio_path = os.path.abspath("python-AI/tests/test_video/test1_pre_audio.wav")

    # 2. 함수 실행
    change_sampling_rate(audio_path, 16000, pre_audio_path)

    # 3. 변경 결과 검증
    get_sampling_rate(audio_path) # 경로와 이름 같으면 덮어쓰기 됨

def test_remove_noise():
    # 1. 샘플링이 변경된 동영상 파일 경로와 노이즈가 제거된 파일이 저장될 경로
    audio_path = os.path.abspath("python-AI/tests/test_video/test1_audio.wav")
    remove_noise_audio_path = os.path.abspath("python-AI/tests/test_video/test1_remove_noise_audio.wav")

    # 2. 노이즈 제거 함수 적용
    remove_noise(audio_path, remove_noise_audio_path)

def test_band_pass_filter():
    # 오디오 로드
    audio_file = os.path.abspath("python-AI/tests/test_video/test1_pre_audio.wav")
    # 필터링 전 비교할 데이터
    data, sr = librosa.load(audio_file, sr=16000)
    # 알아보기 쉽도록 정규화 제거
    data_pcm = data * 32768

    # 필터링 된 오디오 데이터
    filtered_data = band_pass_filter(audio_file)
    # 알아보기 쉽도록 정규화 제거
    filtered_data_pcm = filtered_data * 32768

    # 필터 전후 비교
    # 시간 축 생성
    duration = len(data) / sr  # 신호의 전체 길이 (초)
    time_original = np.linspace(0, duration, len(data))
    time_filtered = np.linspace(0, duration, len(filtered_data))

    # 시각화
    plt.figure(figsize=(14, 8))

    # 원본 신호
    plt.subplot(2, 1, 1)
    plt.plot(time_original, data_pcm, label="Original Signal", color="blue", alpha=0.7)
    plt.title("Original Audio Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # 필터링된 신호
    plt.subplot(2, 1, 2)
    plt.plot(time_filtered, filtered_data_pcm, label="Filtered Signal", color="orange", alpha=0.7)
    plt.title("Filtered Audio Signal (Band-Pass)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # 전체 레이아웃 조정 및 표시
    plt.tight_layout()
    plt.show()

def test_save_filtered_audio():
    # 1. 1차 전처리 완료 오디오 파일 경로
    pre_audio_path = os.path.abspath("python-AI/tests/test_video/test1_pre_audio.wav")
    result_audio_path = os.path.abspath("python-AI/tests/test_video/result_pre_audio.wav")

    # 2. 함수 실행 (저장되었는 지 확인인)
    save_filtered_audio(pre_audio_path,result_audio_path)

    