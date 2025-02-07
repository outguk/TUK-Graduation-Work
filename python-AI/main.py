# 메인 실행 파일
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

import uvicorn
import os
import asyncio
import json
import logging
import shutil
from models.vocalization.vocalization_analysis import extract_audio, transcribe_audio, analyze_speaking_speed, analyze_volume# models 디렉토리에서 AI 모델 로드

app = FastAPI()
""" 2/7 개선할 사항
- 현재 같은 이름 파일 이름을 업로드하면 기존 파일을 덮어씀
중복 파일 이름 방지를 위해 UUID 또는 타임스탬프를 파일 이름에 추가하는 것이 좋음. """

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# JSON 데이터 구조 정의
class RequestData(BaseModel):
    text: str

# 업로드된 파일 저장 디렉토리
UPLOAD_DIR = "./uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    """
    사용자가 업로드한 MP4 파일을 저장하고 분석하는 API
    """
    # 파일 확장자 검증
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

    # 파일 저장 경로 설정
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # 파일 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer) # 업로드된 파일을 서버에 저장

    logging.info(f"Video uploaded: {file_path}")

    # 오디오 추출
    audio_path = file_path.replace(".mp4", ".wav")
    await extract_audio(file_path, audio_path)

    # 음성 텍스트 변환 (STT)
    transcription = await transcribe_audio(audio_path)

    # 분석 수행
    speaking_speed = analyze_speaking_speed(transcription, audio_path)
    volume_analysis = analyze_volume(audio_path)

    # 결과 반환
    results = {
        "filename": file.filename,
        "speaking_speed": speaking_speed,
        "volume_analysis": volume_analysis,
    }

    logging.info(f" {file.filename} 파일 분석 완료")

    return results

# FastAPI 실행 (터미널에서 실행할 경우)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)