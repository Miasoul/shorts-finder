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
from playwright.sync_api import sync_playwright  # 추가: Playwright 임포트
import time  # 추가: 요청 간 딜레이를 위한 time 모듈

app = Flask(__name__)
CORS(app)

# SQLite 데이터베이스 초기화
DB_PATH = "users.db"

def init_db():
    """SQLite 데이터베이스 초기화 (회원 테이블 생성)"""
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
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)", (username, hashed_password,name))
        conn.commit()
        conn.close()
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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password, name FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[0], password):
        return jsonify({"message": "로그인 성공!", "name": user[1]}), 200
    else:
        return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

# 📌 [새로운 기능] - 도서 검색 API (bookKey로 검색)
@app.route('/search_book', methods=['POST'])
def search_book():
    data = request.json
    book_key = data.get("book_key")
    
    if not book_key:
        return jsonify({"error": "도서 키가 제공되지 않았습니다."}), 400
    
    try:
        book_info = fetch_book_info(book_key)
        return jsonify(book_info), 200
    except Exception as e:
        return jsonify({"error": f"도서 정보 검색 중 오류 발생: {str(e)}"}), 500

def fetch_book_info(book_key):
    """책 정보를 가져오는 함수"""
    
    url = f'https://read365.edunet.net/alpasq/api/detail/info?bookKey={book_key}&speciesKey=34169559713&provCode=J10&neisCode=J100000477'
    response = requests.get(url)
    res = response.json()
    title = res['data']['title']
    img = res['data']['coverUrl']
    status = res['data']['status']

    return {"title": title, "status": status, "img": img}

# 📌 [새로운 기능] - 도서명으로 검색하는 API
@app.route('/search_book_name', methods=['POST'])
def search_book_name_api():
    data = request.json
    book_name = data.get("book_name")
    
    if not book_name:
        return jsonify({"error": "도서명이 제공되지 않았습니다."}), 400
    
    try:
        results = search_book_name(book_name)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"도서명 검색 중 오류 발생: {str(e)}"}), 500

def search_book_name(book_name):
    """도서명으로 검색하는 함수"""
    # Step 1: 검색 API에 요청
    for i in range(5):
        search_url = "https://read365.edunet.net/alpasq/api/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "searchKeyword": book_name,
            "neisCode": ["J100000477"],
            "provCode": "J10",
            "page": i,
            "schoolName": "관양고등학교",
            "coverYn": "N"
        }

        response = requests.post(search_url, json=payload, headers=headers)

        if not response.ok:
            return {"error": f"검색 API 요청 실패: {response.status_code} - {response.text}"}

        # Step 2: bookKey 전부 추출
        data = response.json().get("data", {})
        book_list = data.get("bookList", [])

        book_keys = [book.get("bookKey") for book in book_list if "bookKey" in book]

        if not book_keys:
            return {"message": "검색 결과가 없습니다.", "books": []}

        # Step 3: bookKey별로 상세 정보 요청
        details = []

        for i, key in enumerate(book_keys, 1):
            try:
                # 직접 fetch_book_info 함수를 사용해 상세 정보 가져오기
                book_detail = fetch_book_info(key)
                book_detail["bookKey"] = key  # bookKey도 결과에 포함
                details.append(book_detail)
                # 서버에 부담 주지 않게 0.3초 대기
                
            except Exception as e:
                # 오류가 발생한 항목은 오류 정보와 함께 추가
                details.append({
                    "bookKey": key,
                    "error": str(e)
                })

        # Step 4: 결과 딕셔너리 생성 및 반환
    result = {
        "keyword": book_name,
        "total_count": len(details),
        "books": details
    }
    
    return result

# 📌 [기존 기능 유지] - CNN 모델 준비
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# 📌 [기존 기능 유지] - 이미지 특징 추출 함수
def extract_cnn_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# 📌 [기존 기능 유지] - 저장된 특징 벡터와 입력 이미지 비교
def find_similar_video_from_saved_features(capture_image, save_folder):
    best_match_score = -1
    best_video = None
    best_frame_time = None

    capture_img = cv2.imread(capture_image)
    if capture_img is None:
        return None, None, None

    capture_features = extract_cnn_features(capture_img, model)

    for feature_file in os.listdir(save_folder):
        if feature_file.endswith("_features.pkl"):
            video_name = feature_file.replace("_features.pkl", "")
            feature_path = os.path.join(save_folder, feature_file)

            with open(feature_path, 'rb') as f:
                feature_list = pickle.load(f)

            for i, frame_features in enumerate(feature_list):
                similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]

                if similarity_score > best_match_score:
                    best_match_score = similarity_score
                    best_video = video_name
                    best_frame_time = i * 3  

    return best_video, best_match_score, best_frame_time

# 📌 [기존 기능 유지] - 영상 검색 API
@app.route('/find_similar_video', methods=['POST'])
def find_similar_video():
    img_url = request.json.get('image_url')
    base64_string = img_url.split(',')[1]

    img_data = base64.b64decode(base64_string)

    with open("output_image.jpg", "wb") as file:
        file.write(img_data)

    if not img_url:
        return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400

    try:
        best_video, best_match_score, best_frame_time = find_similar_video_from_saved_features('./output_image.jpg', save_folder)

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

# 📌 [기존 기능 유지] - 서버 실행
if __name__ == '__main__':
    save_folder = "./data"
    app.run(host="0.0.0.0", port=44324, debug=True)
