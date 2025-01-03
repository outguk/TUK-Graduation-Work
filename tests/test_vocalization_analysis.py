# 발성 분석 테스트 코드

import os
import moviepy as mp
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from collections import Counter
from cmudict import entries as cmu_dict
import json

# Step 1: Extract Audio from Video
# 동영상에서 오디오 추출
def extract_audio(video_file, output_audio_file="audio.wav"):
    # 동적으로 경로 설정
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)  # 폴더 생성
    output_audio_file = os.path.join(output_directory, output_audio_file)
    
    # 오디오 추출 시작 메시지 출력
    print(f"Extracting audio from video: {video_file}")
    
    # MoviePy를 사용하여 동영상 파일을 로드하고 오디오 트랙 추출
    try:
      video = mp.editor.VideoFileClip(video_file)
    except FileNotFoundError:
      print(f"Error: Video file '{video_file}' not found.")
      return None

    video.audio.write_audiofile(output_audio_file)
    # 오디오 추출 완료 후 메시지 출력
    print(f"Audio extracted to: {output_audio_file}")
    return output_audio_file

# Step 2: Speech-to-Text with Whisper
# 음성을 텍스트로 변환 (Whisper 사용)
def transcribe_audio(audio_file, model_name="base"):
    # 텍스트 변환 시작 메시지 출력
    print(f"Transcribing audio: {audio_file} with model: {model_name}")
    # Whisper 모델 로드
    model = whisper.load_model(model_name)
    # 로드된 모델로 음성 텍스트 변환 수행
    transcription = model.transcribe(audio_file)
    # 변환된 텍스트의 첫 100자를 출력하여 검증
    print(f"Transcription completed. Text: {transcription['text'][:100]}...")
    return transcription

# Step 3: Analyze Speaking Speed (Words per Minute)
# 말하기 속도 분석 (분당 단어 수 계산)
def calculate_speaking_speed(transcription, duration):
    # 말하기 속도 계산 시작 메시지 출력
    print(f"Calculating speaking speed for duration: {duration} seconds")
    # 변환된 텍스트를 단어로 분리
    words = transcription['text'].split()
    word_count = len(words)  # 단어 수 계산
    # 분당 단어 수 계산
    words_per_minute = word_count / (duration / 60)
    # 계산된 말하기 속도 출력
    print(f"Speaking speed: {words_per_minute} words per minute")
    return words_per_minute

# Step 4: Volume Analysis
# 음량 분석
def analyze_volume(audio_file):
    # 음량 분석 시작 메시지 출력
    print(f"Analyzing volume for audio file: {audio_file}")
    # Pydub을 사용하여 오디오 파일 로드
    audio = AudioSegment.from_wav(audio_file)
    # 오디오의 비침묵 구간 감지
    segments = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS - 16)
    volume_stats = []

    # 감지된 각 구간의 음량 수준 분석
    for start, end in segments:
        segment = audio[start:end]
        db_level = segment.dBFS  # 구간의 데시벨 수준 가져오기
        volume_stats.append(db_level)

    # 평균, 최대, 최소 음량 계산
    avg_volume = sum(volume_stats) / len(volume_stats) if volume_stats else None
    max_volume = max(volume_stats) if volume_stats else None
    min_volume = min(volume_stats) if volume_stats else None
    # 계산된 음량 통계 출력
    print(f"Volume analysis - Average: {avg_volume}, Max: {max_volume}, Min: {min_volume}")

    return {
        "average_volume": avg_volume,
        "max_volume": max_volume,
        "min_volume": min_volume,
    }

# Step 5: Pronunciation Analysis
# 발음 분석
def analyze_pronunciation(transcription):
    # 발음 분석 시작 메시지 출력
    print(f"Analyzing pronunciation for transcription text.")
    # 변환된 텍스트를 소문자로 변환하고 단어로 분리
    words = transcription['text'].lower().split()
    # CMU 발음 사전 로드
    cmu = {word: phonemes for word, phonemes in cmu_dict()}
    pronunciation_issues = []

    # 각 단어를 CMU 사전과 대조
    for word in words:
        if word not in cmu:
            pronunciation_issues.append(word)  # 사전에 없는 단어를 기록

    # 발음 분석 결과 출력
    print(f"Pronunciation analysis - Total words: {len(words)}, Mispronounced: {len(pronunciation_issues)}")
    return {
        "total_words": len(words),
        "mispronounced_words": pronunciation_issues,
    }

# Main Function
# 메인 함수
def analyze_presentation(video_file):
    # 분석 프로세스 시작 메시지 출력
    print(f"Starting analysis for video file: {video_file}")
    # Step 1: 동영상에서 오디오 트랙 추출
    audio_file = extract_audio(video_file)
    # Step 2: 추출된 오디오 텍스트 변환
    transcription = transcribe_audio(audio_file)

    # Pydub을 사용하여 오디오 파일 로드 후 길이 측정
    audio = AudioSegment.from_wav(audio_file)
    duration = len(audio) / 1000  # 밀리초를 초로 변환

    # Step 3: 말하기 속도 계산
    speaking_speed = calculate_speaking_speed(transcription, duration)
    # Step 4: 음량 분석 수행
    volume_analysis = analyze_volume(audio_file)
    # Step 5: 발음 분석 수행
    pronunciation_analysis = analyze_pronunciation(transcription)

    # 결과를 딕셔너리로 정리
    results = {
        "speaking_speed_wpm": speaking_speed,
        "volume_analysis": volume_analysis,
        "pronunciation_analysis": pronunciation_analysis,
    }

    # 분석 완료 메시지 출력
    print("Analysis completed. Saving results to analysis_results.json")
    # 결과를 JSON 파일로 저장
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 결과 저장 성공 메시지 출력
    print("Results saved successfully.")
    return results

# Example Usage
# 예제 실행
if __name__ == "__main__":
    # 처리할 동영상 파일 경로 지정
    video_path = "C:\Users\dlsrn\OneDrive\바탕 화면\종합설계\Sample\Sample\01.원천데이터\1.언어적\2. A01 고등학생\A01_S01_M_F_08_139_02_WA_MO.mp4"  # 동영상 파일 경로
    print(f"Processing video: {video_path}")
    # 분석 수행 및 최종 결과 출력
    results = analyze_presentation(video_path)
    print("Final Results:")
    print(json.dumps(results, indent=4))
