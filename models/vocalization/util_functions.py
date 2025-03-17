# 공통 유틸 함수용
import logging
import os
from pydub import AudioSegment, silence
from scipy.signal import butter, lfilter

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


def band_pass_filter(data, sr, lowcut=85, highcut=8000):
    """
    Args:
        data (str): 오디오 데이터
        target_rate (int): 변경할 샘플링 레이트 (Hz).
        output_path (str): 변경된 파일 저장 경로.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data




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
