from flask import Flask, request, jsonify
import os
import pickle
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import re
import sqlite3
import base64
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import time
import concurrent.futures
import functools
import threading
from cachetools import TTLCache, cached

app = Flask(__name__)
CORS(app)

# 캐시 설정 - 검색 결과를 저장하는 캐시 (TTL: 1시간)
book_cache = TTLCache(maxsize=1000, ttl=3600)

# SQLite 데이터베이스 초기화
DB_PATH = "users.db"
DB_LOCK = threading.Lock()  # 데이터베이스 접근을 위한 락

def init_db():
    """SQLite 데이터베이스 초기화 (회원 테이블 생성)"""
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

init_db()  # 서버 실행 시 데이터베이스 초기화

# CNN 모델 로드 (전역 변수로 한 번만 로드)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# 세션 재사용을 위한 전역 세션 객체
session = requests.Session()

# 데이터베이스 연결 함수 - 컨텍스트 매니저
class DatabaseConnection:
    def __enter__(self):
        self.conn = sqlite3.connect(DB_PATH)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

# 📌 [회원가입 API]
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    name = data.get("name")
    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400

    hashed_password = generate_password_hash(password)  # 비밀번호 해싱

    try:
        with DB_LOCK:
            with DatabaseConnection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)", 
                              (username, hashed_password, name))
                conn.commit()
        return jsonify({"message": "회원가입 성공!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "이미 존재하는 아이디입니다."}), 400

# 📌 [로그인 API]
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400

    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password, name FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

    if user and check_password_hash(user[0], password):
        return jsonify({"message": "로그인 성공!", "name": user[1]}), 200
    else:
        return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

# 책 정보 캐싱 데코레이터
@functools.lru_cache(maxsize=1000)
def fetch_book_info(book_key):
    """책 정보를 가져오는 함수 (캐싱 적용)"""
    url = f'https://read365.edunet.net/alpasq/api/detail/info?bookKey={book_key}&speciesKey=34169559713&provCode=J10&neisCode=J100000477'
    response = session.get(url, timeout=5)  # 타임아웃 설정
    res = response.json()
    title = res['data']['title']
    img = res['data']['coverUrl']
    status = res['data']['status']

    return {"title": title, "status": status, "img": img, "bookKey": book_key}

# 📌 [도서 검색 API] - bookKey로 검색
@app.route('/search_book', methods=['POST'])
def search_book():
    data = request.json
    book_key = data.get("book_key")
    
    if not book_key:
        return jsonify({"error": "도서 키가 제공되지 않았습니다."}), 400
    
    try:
        # 캐시된 fetch_book_info 함수 사용
        book_info = fetch_book_info(book_key)
        return jsonify(book_info), 200
    except Exception as e:
        return jsonify({"error": f"도서 정보 검색 중 오류 발생: {str(e)}"}), 500

# 📌 [도서명으로 검색하는 API] - 병렬 처리 및 캐싱 적용
@app.route('/search_book_name', methods=['POST'])
def search_book_name_api():
    data = request.json
    book_name = data.get("book_name")
    
    if not book_name:
        return jsonify({"error": "도서명이 제공되지 않았습니다."}), 400
    
    # 캐시에서 검색
    cache_key = f"search_{book_name}"
    if cache_key in book_cache:
        return jsonify(book_cache[cache_key]), 200
    
    try:
        results = search_book_name(book_name)
        # 결과 캐싱
        book_cache[cache_key] = results
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"도서명 검색 중 오류 발생: {str(e)}"}), 500

def get_book_keys_for_page(book_name, page):
    """단일 페이지에서 도서 키 추출"""
    search_url = "https://read365.edunet.net/alpasq/api/search"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "searchKeyword": book_name,
        "neisCode": ["J100000477"],
        "provCode": "J10",
        "page": str(page),
        "schoolName": "관양고등학교",
        "coverYn": "N"
    }

    response = session.post(search_url, json=payload, headers=headers, timeout=5)
    if not response.ok:
        return []

    data = response.json().get("data", {})
    book_list = data.get("bookList", [])
    if not book_list:
        return []

    return [book.get("bookKey") for book in book_list if "bookKey" in book]

def search_book_name(book_name):
    """도서명으로 검색하는 함수 - 병렬 처리 적용"""
    all_book_keys = []
    
    # 여러 페이지를 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_book_keys_for_page, book_name, i+1) for i in range(5)]
        for future in concurrent.futures.as_completed(futures):
            book_keys = future.result()
            if book_keys:
                all_book_keys.extend(book_keys)
    
    all_details = []
    
    # 도서 상세 정보도 병렬로 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 각 책 키에 대해 상세 정보 요청 병렬 처리
        future_to_key = {executor.submit(fetch_book_info, key): key for key in all_book_keys}
        
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                book_detail = future.result()
                all_details.append(book_detail)
            except Exception as e:
                all_details.append({
                    "bookKey": key,
                    "error": str(e)
                })
    
    return {
        "keyword": book_name,
        "total_count": len(all_details),
        "books": all_details
    }

# 이미지 특징 추출 함수 - 최적화
def extract_cnn_features(image, model):
    # 이미지가 이미 로드되어 있는지 확인
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
    
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)  # verbose=0으로 출력 줄임
    return features.flatten()

# 이미지 특징 벡터 캐시
feature_cache = {}

# 특징 벡터 로딩 최적화 - 한 번에 모든 특징 로드
def load_all_features(save_folder):
    """폴더 내 모든 특징 벡터를 한 번에 로드"""
    global feature_cache
    
    if feature_cache:
        return feature_cache
        
    features = {}
    for feature_file in os.listdir(save_folder):
        if feature_file.endswith("_features.pkl"):
            video_name = feature_file.replace("_features.pkl", "")
            feature_path = os.path.join(save_folder, feature_file)
            
            try:
                with open(feature_path, 'rb') as f:
                    feature_list = pickle.load(f)
                features[video_name] = feature_list
            except Exception as e:
                print(f"Error loading {feature_file}: {str(e)}")
    
    feature_cache = features
    return features

# 유사 비디오 검색 함수 최적화
def find_similar_video_from_saved_features(capture_image, save_folder):
    best_match_score = -1
    best_video = None
    best_frame_time = None

    # 캡처 이미지 특징 추출
    capture_features = extract_cnn_features(capture_image, model)
    
    # 전체 특징 벡터 로드
    all_features = load_all_features(save_folder)
    
    # 병렬 처리를 위한 작업 정의
    def process_video(video_name, feature_list):
        best_score = -1
        best_frame = None
        
        for i, frame_features in enumerate(feature_list):
            similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_frame = i * 3  # 프레임 간격
        
        return (video_name, best_score, best_frame)
    
    # 병렬 처리로 모든 비디오 처리
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(all_features))) as executor:
        futures = [executor.submit(process_video, video_name, feature_list) 
                  for video_name, feature_list in all_features.items()]
        
        for future in concurrent.futures.as_completed(futures):
            video_name, score, frame_time = future.result()
            if score > best_match_score:
                best_match_score = score
                best_video = video_name
                best_frame_time = frame_time

    return best_video, best_match_score, best_frame_time

# 📌 [영상 검색 API]
@app.route('/find_similar_video', methods=['POST'])
def find_similar_video():
    img_url = request.json.get('image_url')
    if not img_url:
        return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400
    
    try:
        # Base64 이미지 디코딩
        base64_string = img_url.split(',')[1]
        img_data = base64.b64decode(base64_string)
        
        # 임시 파일로 저장
        temp_img_path = "output_image.jpg"
        with open(temp_img_path, "wb") as file:
            file.write(img_data)
        
        # 특징 비교
        save_folder = "./data"
        best_video, best_match_score, best_frame_time = find_similar_video_from_saved_features(temp_img_path, save_folder)
        
        if best_video is None or best_frame_time is None:
            return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404
        
        matches = re.findall(r'\[([^\]]+)\]', best_video)
        if len(matches) >= 2:
            extracted_id = matches[1]
            youtube_link = f"https://www.youtube.com/watch?v={extracted_id}&t={best_frame_time}"
            
            return jsonify({
                'best_video': best_video,
                'best_match_score': float(best_match_score),
                'best_frame_time': best_frame_time,
                'youtube_link': youtube_link
            })
        else:
            return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 📌 서버 시작 시 리소스 로딩 (미리 캐시 준비)
def preload_resources():
    """서버 시작 시 필요한 리소스를 미리 로드"""
    # 1. 특징 벡터 미리 로드
    save_folder = "./data"
    load_all_features(save_folder)
    print("🚀 특징 벡터 미리 로드 완료")

# 📌 서버 실행
if __name__ == '__main__':
    save_folder = "./data"
    # 백그라운드에서 리소스 미리 로드
    threading.Thread(target=preload_resources).start()
    app.run(host="0.0.0.0", port=44324, debug=True)
