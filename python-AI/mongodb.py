# mongodb.py
from motor.motor_asyncio import AsyncIOMotorClient
import os

# 복사해둔 MongoDB URI 넣기 (비밀번호는 URL 인코딩된 상태여야 함!)

MONGO_DB_URI = "mongodb+srv://cjhh0707:ihQQhTuqTRQ6N9Kj@analysis-db.nppaekk.mongodb.net/analysis_db?retryWrites=true&w=majority&appName=analysis-db"# 예: mongodb+srv://user:password@cluster.mongodb.net

client = AsyncIOMotorClient(MONGO_DB_URI)
db = client["analysis_db"]  # 데이터베이스 이름
collection = db["results"]  # 컬렉션 이름
