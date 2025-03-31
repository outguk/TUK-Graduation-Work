# 메인 실행 파일
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
import uvicorn
import os
import asyncio
import json
import logging
import shutil
import glob

# ===== 기존 AI 분석 + MongoDB =====
from models.vocalization.vocalization_analysis import (
    extract_audio,
    transcribe_audio,
    analyze_speaking_speed,
    analyze_volume
)
from models.nonvarbal.nonvarbal_analysis import video_nonverbal_analysis
from models.vocalization.vocalization_evaluate import (
    evaluate_speaking_speed,
    evaluate_volume
)
from mongodb import collection  # MongoDB 연결

# ===== RDS + SQLAlchemy =====
# (이미 config.py, database.py, db_models/user.py에 분리되어 있다고 가정)
from database import SessionLocal  # DB 세션 팩토리
from db_models.user import User
from pydantic import BaseModel, EmailStr


# FastAPI 앱 생성
app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 업로드 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def cleanup_intermediate_files(video_name: str):
    """
    분석 후 생성된 임시 파일/폴더들을 삭제한다.
    video_name: 예) "book" ( book.mp4 → "book" )
    """
    data_dir = os.path.join(BASE_DIR, "models", "nonvarbal", "data")

    # 1) frames/<영상이름>/ 폴더
    frames_path = os.path.join(data_dir, "frames", video_name)
    # 2) keypoints/<영상이름>/ 폴더
    keypoints_path = os.path.join(data_dir, "keypoints", video_name)
    # 3) visualization/<영상이름>/ 폴더
    visualizations_path = os.path.join(data_dir, "visualizations", video_name)
    # 4) test_results.pkl
    test_results_file = os.path.join(data_dir, "test_results.pkl")
    # 5) inference_results.json
    inference_json = os.path.join(data_dir, "inference_results.json")

    # 폴더 삭제
    for path in [frames_path, keypoints_path, visualizations_path]:
        if os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f"[CLEANUP] 디렉토리 삭제: {path}")

    # 파일 삭제
    for path in [test_results_file, inference_json]:
        if os.path.isfile(path):
            os.remove(path)
            logging.info(f"[CLEANUP] 파일 삭제: {path}")

    logging.info(f"[CLEANUP] {video_name} 분석 과정에서 생성된 임시 파일/폴더 삭제 완료.")


# ========== 기존 업로드 + 분석 + MongoDB 저장 ==========

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    logging.info(f"📂 업로드된 파일: {file.filename}, Content-Type: {file.content_type}")

    # 파일 확장자 검증
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

    # 업로드된 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logging.info(f"Video uploaded: {file_path}")

    # 오디오 추출
    audio_path = file_path.replace(".mp4", ".wav")
    await extract_audio(file_path, audio_path)

    # 음성 텍스트 변환 (STT)
    transcription = await transcribe_audio(audio_path)

    # 음성 분석
    speaking_speed = analyze_speaking_speed(transcription, audio_path)
    volume_analysis = analyze_volume(audio_path)

    # 비언어 분석
    absolute_file_path = os.path.abspath(file_path)
    nonverbal_analysis = video_nonverbal_analysis(absolute_file_path)

    # 평가
    speed_score = evaluate_speaking_speed(speaking_speed)
    volume_score = evaluate_volume(volume_analysis)

    # MongoDB 저장 (분석 결과 문서)
    document = {
        "filename": file.filename,
        "speaking_speed": speaking_speed,
        "speaking_evaluation": speed_score,
        "volume_analysis": volume_analysis,
        "volume_evaluation": volume_score,
        "nonverbal_analysis": nonverbal_analysis
    }
    inserted = await collection.insert_one(document)
    logging.info(f"분석 결과 MongoDB에 저장함: ID={inserted.inserted_id}")

    # 임시 파일/폴더 정리
    video_name = os.path.splitext(file.filename)[0]  # e.g. "book"
    cleanup_intermediate_files(video_name)

    return {"message": "분석 완료", "id": str(inserted.inserted_id)}


@app.get("/get-analysis/")
async def get_analysis(filename: str = Query(..., description="업로드된 파일 이름")):
    document = await collection.find_one({"filename": filename})
    if not document:
        raise HTTPException(status_code=404, detail="해당 파일 분석 결과 없음")
    document["_id"] = str(document["_id"])  # ObjectId → 문자열 변환
    return document


# ========== RDS 사용자 관리 ==========

# Pydantic 스키마
class UserCreate(BaseModel):
    name: str
    #email: EmailStr
    password: str

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register-rds")
def register_user(user_in: UserCreate, db=Depends(get_db)):
    # username or email 중복 체크
    existing = db.query(User).filter(
        (User.name == user_in.name) #| (User.email == user_in.email)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or Email already exists.")

    new_user = User(
        name=user_in.name,
        #email=user_in.email,
        password=user_in.password  # 실제로는 bcrypt 등으로 해싱 권장
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.name,
        #"email": new_user.email
    }

@app.get("/users-rds/{user_id}")
def get_user_rds(user_id: int, db=Depends(get_db)):
    user = db.query(User).get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found in RDS")
    return {
        "id": user.id,
        "name": user.name,
        #"email": user.email
    }


# ===== FastAPI 실행 =====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
