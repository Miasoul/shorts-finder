# Python 3.9 기반의 공식 이미지 사용
FROM python:3.9

# 컨테이너 내부 작업 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 모든 파일을 컨테이너 내부로 복사
COPY . /app

# 필수 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션 실행 (Fly.io는 8080 포트를 사용)
CMD ["python", "main.py"]