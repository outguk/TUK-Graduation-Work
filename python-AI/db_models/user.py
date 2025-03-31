from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users' #테이블 이름 설정

    id = Column(Integer, primary_key=True, index=True) #기본키 
    name = Column(String(100), unique=True, nullable=False) #이름
    password = Column(String(200), nullable=False) #비밀번호
