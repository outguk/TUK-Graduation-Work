# 공통 유틸 함수용
import logging
import os
import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, silence
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write

# 오디오 파일의 샘플링 레이트를 확인하는 함수
def get_sampling_rate(audio_file_path):
    """
    Args:
        audio_file_path (str): 오디오 파일 경로.
    
    Returns:
        int: 오디오 파일의 샘플링 레이트 (Hz).
    """
    audio = AudioSegment.from_file(audio_file_path)
    logging.info(f"샘플링 레이트 : {audio.frame_rate}")

# 오디오 파일의 샘플링 레이트를 변경하는 함수
def change_sampling_rate(audio_file_path, target_rate, output_path):
    """
    Args:
        audio_file_path (str): 원본 오디오 파일 경로.
        target_rate (int): 변경할 샘플링 레이트 (Hz).
        output_path (str): 변경된 파일 저장 경로.
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        resampled_audio = audio.set_frame_rate(target_rate)
        resampled_audio.export(output_path, format="wav")
        logging.info(f"샘플링 레이트를 {target_rate} Hz로 변경하여 저장했습니다: {output_path}")
    except Exception as e:
        logging.info(f"오류 발생: {e}")

def remove_noise(input_path, output_path, noise_duration=1):
    """
    입력 오디오에서 노이즈를 제거하고 필터링된 신호를 WAV 파일로 저장.

    Args:
        input_path (str): 원본 오디오 파일 경로.
        output_path (str): 노이즈 제거 후 저장할 파일 경로.
        noise_duration (float): 노이즈 샘플 길이 (초 단위).

    Returns:
        ndarray: 노이즈 제거된 신호 데이터. (여기서 사용하지는 않음)
    """
    try:
        # Step 1: 원본 오디오 로드
        data, sr = librosa.load(input_path, sr=None)

        # Step 2: 노이즈 샘플 추출 (첫 noise_duration 초 사용)
        noise_sample = data[:int(noise_duration * sr)]

        # Step 3: 노이즈 제거
        reduced_noise = nr.reduce_noise(y=data, sr=sr, y_noise=noise_sample)

        # Step 4: 결과를 16비트 PCM 값으로 변환
        pcm_data = np.int16(reduced_noise / np.max(np.abs(reduced_noise)) * 32767)

        # Step 5: WAV 파일로 저장
        write(output_path, sr, pcm_data)
        logging.info(f"노이즈 제거된 오디오가 저장되었습니다: {output_path}")

        return reduced_noise

    except Exception as e:
        logging.info(f"오류 발생: {e}")
        return None


def band_pass_filter(audio_path, lowcut=85, highcut=7999):
    """
    대역 통과 필터를 적용하여 저주,고주파 성분 제거.

    Args:
        audio_path (str): 입력 오디오 파일 경로로.
        lowcut (float): 차단 하한 주파수 (Hz).
        highcut (int): 차단 상한 주파수 (Hz).

    Returns:
        array: 필터링된 신호 데이터.
    """
    # Step 1: 원본 오디오 로드
    data, sr = librosa.load(audio_path, sr=16000)
    # Step 2: Nyquist 주파수는 디지털 신호 처리에서 샘플링 레이트의 절반 값으로 정의
    nyquist = 0.5 * sr 
    # 주파수를 Nyquist 주파수로 나누어 정규화된 주파수로 변환
    low = lowcut / nyquist
    high = highcut / nyquist

    # Step 3: 주파수 값 검증
    if not (0 < low < 1 and 0 < high < 1):
        raise ValueError(f"정규화된 주파수 값이 잘못되었습니다. low: {low}, high: {high}")

    """
    butter(): 대역 통과 필터를 설계하는 함수로, Butterworth 필터를 생성
    1: 필터의 차수 -> 차수가 높을수록 계산이 복잡해지지만 정확도?
    [low, high]: 대역 통과 필터의 정규화된 주파수 범위.
    btype='band': 필터의 유형을 지정, 여기서는 대역 통과 필터(Band-Pass Filter)
    """
    # Step 4: 대역 통과 필터 생성
    b, a = butter(1, [low, high], btype='band') # b: 필터의 분자 계수, a: 필터의 분모 계수.

    # Step 5: 설계된 필터(b, a)를 실제 데이터(data)에 적용하여 필터를 통과한 신호 생성
    filtered_data = lfilter(b, a, data)

    # Step 6 : 필터를 통과한 오디오 신호 데이터 반환
    return filtered_data

def save_filtered_audio(pre_audio_input_path, output_path):
    """
    필터링된 오디오 데이터를 WAV 파일로 저장.

    Args:
        pre_audio_input_path (str): 입력 오디오 파일 경로(샘플링 레이트가 변경된 1차 전처리 오디오 데이터 경로)
        output_path (str): 필터링된 데이터를 저장할 파일 경로.
    """
    try:

        # Step 1: 1차 전처리 오디오에 대역 통과 필터 적용
        filtered_data = band_pass_filter(pre_audio_input_path)

        # Step 2: 필터링된 데이터를 16비트 PCM 값으로 변환
        pcm_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)

        # Step 3: WAV 파일로 저장
        write(output_path, 16000, pcm_data)
        logging.info(f"필터링된 오디오가 저장되었습니다: {output_path}")

    except Exception as e:
        logging.info(f"오디오 저장 중 오류 발생: {e}")




# 발화 구간 추출 함수
# 큰 오디오 처리와 동시 입력 시 성능 이슈로 인한 비동기 or 병렬 처리 고려 필요요
def process_audio_chunks(audio_file_path, min_silence_len=1500, silence_thresh=-40):
    """
    오디오 파일을 로드하고 무음/비무음 구간을 처리하는 함수.

    Args:
        audio_file_path (str): 오디오 파일 경로.
        min_silence_len (int): 무음으로 간주할 최소 길이(ms).
        silence_thresh (int): 무음으로 간주할 데시벨 임계값(dBFS).

    Returns:
        tuple: (audio, 비무음 구간 리스트)
    """
    try:
        # Step 1 : 오디오 파일 로드
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file_path}")
        audio = AudioSegment.from_file(audio_file_path)
        logging.info(f"오디오 로드 완료: {audio_file_path}")

        # Step 2 : # 2. 무음 구간 탐지, 무음 구간의 시작과 끝을 밀리초 단위로 리스트에 저장 ex) [(0, 500), (2000, 3000)].
        silent_chunks = silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        logging.info(f"무음 구간 감지: {len(silent_chunks)}개 구간")

        # Step 3 : 발화 구간 계산
        total_duration = len(audio) / 1000  # 총 길이 (초)
        non_silent_chunks = [(silent_chunks[i-1][1] / 1000, start / 1000) for i, (start, _) in enumerate(silent_chunks[1:], start=1)]
        if silent_chunks and silent_chunks[0][0] > 0:
            non_silent_chunks.insert(0, (0, silent_chunks[0][0] / 1000))
        if silent_chunks and silent_chunks[-1][1] < len(audio):
            non_silent_chunks.append((silent_chunks[-1][1] / 1000, total_duration))
        
        logging.info(f"음성 구간 추출 완료: {len(non_silent_chunks)}개")
        return audio, non_silent_chunks

    except Exception as e:
        logging.error(f"오디오 처리(발화 구간 추출) 중 오류 발생: {e}")
        raise
