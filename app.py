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
from playwright.sync_api import sync_playwright
import time
import logging
from logging.handlers import RotatingFileHandler
import secrets
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
# 비밀 키 설정 - 세션 및 CSRF 방어에 사용
app.secret_key = secrets.token_hex(16)

# CORS 설정 - 필요한 도메인만 허용
CORS(app)

# 요청 제한 설정 (rate limiting)
# Flask-Limiter 최신 버전 지원
try:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"]
    )
except TypeError:
    # 이전 버전 지원
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )

# 로깅 설정
if not os.path.exists('logs'):
    os.mkdir('logs')
    
file_handler = RotatingFileHandler('logs/server.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('서버 시작')

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
            name TEXT NOT NULL,
            failed_attempts INTEGER DEFAULT 0,
            last_failed_attempt TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()  # 서버 실행 시 데이터베이스 초기화

# 입력 검증 함수
def validate_input(text):
    """입력값에 SQL 인젝션이나 XSS 시도가 있는지 검사"""
    if text is None:
        return False
        
    dangerous_patterns = [
        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", 
        "UNION", "OR 1=1", "--", "/*", "*/", "<script>",
        "onerror", "javascript:", "eval("
    ]
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern.lower() in text_lower:
            app.logger.warning(f"위험한 입력 패턴 감지: {pattern}")
            return False
    return True

# 📌 [회원가입 API] - 입력 검증 추가
@app.route('/register', methods=['POST'])
@limiter.limit("5 per minute")  # 회원가입 요청 제한
def register():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")
        realname = data.get("realname")
        
        # 입력 검증
        if not username or not password or not validate_input(username) or not validate_input(realname):
            return jsonify({"error": "잘못된 입력이 감지되었습니다."}), 400

        # 비밀번호 강도 검증
        if len(password) < 8:
            return jsonify({"error": "비밀번호는 최소 8자 이상이어야 합니다."}), 400

        hashed_password = generate_password_hash(password)  # 비밀번호 해싱

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, name) VALUES (?, ?, ?)", 
                      (username, hashed_password, realname))
        conn.commit()
        conn.close()
        
        app.logger.info(f"새 사용자 등록 성공: {username}")
        return jsonify({"message": "회원가입 성공!"}), 201
    
    except sqlite3.IntegrityError:
        return jsonify({"error": "이미 존재하는 아이디입니다."}), 400
    except Exception as e:
        app.logger.error(f"회원가입 오류: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다."}), 500

# 📌 [로그인 API] - 보안 강화
@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")  # 로그인 요청 제한
def login():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if not username or not password or not validate_input(username):
            return jsonify({"error": "잘못된 입력이 감지되었습니다."}), 400

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 사용자 정보 및 실패 시도 카운트 조회
        cursor.execute("SELECT id, password, name, failed_attempts FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        # 존재하지 않는 사용자 또는 비밀번호 불일치
        if not user or not check_password_hash(user[1], password):
            # 존재하는 사용자면 실패 카운트 증가
            if user:
                cursor.execute("UPDATE users SET failed_attempts = failed_attempts + 1, last_failed_attempt = CURRENT_TIMESTAMP WHERE id = ?", 
                             (user[0],))
                conn.commit()
                
                # 10회 이상 실패시 계정 잠금 로직 추가 가능
                
            conn.close()
            app.logger.warning(f"로그인 실패 - 사용자: {username}")
            return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401
            
        # 로그인 성공 - 실패 카운트 초기화
        cursor.execute("UPDATE users SET failed_attempts = 0 WHERE id = ?", (user[0],))
        conn.commit()
        conn.close()
        
        app.logger.info(f"로그인 성공: {username}")
        return jsonify({"message": "로그인 성공!", "name": user[2]}), 200
        
    except Exception as e:
        app.logger.error(f"로그인 오류: {str(e)}")
        return jsonify({"error": "서버 오류가 발생했습니다."}), 500

# 📌 [도서 검색 API] - 입력 검증 및 요청 제한 추가
@app.route('/search_book', methods=['POST'])
@limiter.limit("20 per minute")
def search_book():
    try:
        data = request.json
        book_key = data.get("book_key")
        
        if not book_key or not validate_input(book_key):
            return jsonify({"error": "잘못된 도서 키가 제공되었습니다."}), 400
        
        app.logger.info(f"도서 검색 요청: {book_key}")
        book_info = fetch_book_info(book_key)
        return jsonify(book_info), 200
    
    except Exception as e:
        app.logger.error(f"도서 검색 오류: {str(e)}")
        return jsonify({"error": "도서 정보 검색 중 오류가 발생했습니다."}), 500

def fetch_book_info(book_key):
    """책 정보를 가져오는 함수 - 타임아웃 및 예외 처리 강화"""
    url = f"https://read365.edunet.net/PureScreen/SearchDetail?bookKey={book_key}&speciesKey=34169559343&provCode=J10&neisCode=J100000477&schoolName=관양고등학교"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(url, timeout=15000)  # 타임아웃 연장

            # 데이터 추출 전 요소가 로드되었는지 확인
            page.wait_for_selector("xpath=/html/body/div[1]/div/div[1]/div/div/article[1]/div[1]/div[1]/div[1]/div[2]/h3", timeout=10000)
            
            title = page.locator("xpath=/html/body/div[1]/div/div[1]/div/div/article[1]/div[1]/div[1]/div[1]/div[2]/h3").inner_text()
            status = page.locator("xpath=/html/body/div[1]/div/div[1]/div/div/article[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div").inner_text()
            img = page.locator('xpath=/html/body/div[1]/div/div[1]/div/div/article[1]/div[1]/div[1]/div[1]/div[1]/div[1]/img').get_attribute('src')
            
            return {"title": title.strip(), "status": status.strip(), "img": img}
            
        except Exception as e:
            app.logger.error(f"도서 정보 가져오기 오류: {str(e)}")
            raise
        finally:
            browser.close()

# 📌 [도서명 검색 API] - 입력 검증 및 요청 제한 추가
@app.route('/search_book_name', methods=['POST'])
@limiter.limit("10 per minute")
def search_book_name_api():
    try:
        data = request.json
        book_name = data.get("book_name")
        
        if not book_name or not validate_input(book_name):
            return jsonify({"error": "잘못된 도서명이 제공되었습니다."}), 400
        
        app.logger.info(f"도서명 검색 요청: {book_name}")
        results = search_book_name(book_name)
        return jsonify(results), 200
        
    except Exception as e:
        app.logger.error(f"도서명 검색 오류: {str(e)}")
        return jsonify({"error": "도서명 검색 중 오류가 발생했습니다."}), 500

def search_book_name(book_name):
    """도서명으로 검색하는 함수 - 요청 제한 및 예외처리 강화"""
    # Step 1: 검색 API에 요청
    search_url = "https://read365.edunet.net/alpasq/api/search"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    payload = {
        "searchKeyword": book_name,
        "neisCode": ["J100000477"],
        "provCode": "J10",
        "schoolName": "관양고등학교",
        "coverYn": "N"
    }

    try:
        response = requests.post(search_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # HTTP 오류 발생시 예외 발생
    except requests.exceptions.RequestException as e:
        app.logger.error(f"교육넷 API 요청 실패: {str(e)}")
        return {"error": f"검색 API 요청 실패: {str(e)}"}

    # Step 2: bookKey 전부 추출
    data = response.json().get("data", {})
    book_list = data.get("bookList", [])

    book_keys = [book.get("bookKey") for book in book_list if "bookKey" in book]

    if not book_keys:
        return {"message": "검색 결과가 없습니다.", "books": []}

    # Step 3: bookKey별로 상세 정보 요청 - 최대 10개로 제한하여 서버 부하 방지
    details = []
    max_books = min(len(book_keys), 10)

    for i, key in enumerate(book_keys[:max_books], 1):
        try:
            book_detail = fetch_book_info(key)
            book_detail["bookKey"] = key  # bookKey도 결과에 포함
            details.append(book_detail)
            # 서버에 부담 주지 않게 0.5초 대기
            time.sleep(0.5)
        except Exception as e:
            app.logger.error(f"책 상세정보 가져오기 실패 (bookKey: {key}): {str(e)}")
            # 오류가 발생한 항목은 오류 정보와 함께 추가
            details.append({
                "bookKey": key,
                "error": "정보를 가져오는 중 오류가 발생했습니다."
            })

    # Step 4: 결과 딕셔너리 생성 및 반환
    result = {
        "keyword": book_name,
        "total_count": len(book_keys),
        "returned_count": len(details),
        "books": details
    }
    
    return result

# 📌 [기존 기능 유지] - CNN 모델 준비
# 모델 로딩을 함수화하여 필요할 때만 로드하도록 변경
def load_model():
    global model
    if 'model' not in globals():
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    return model

# 📌 [기존 기능 유지] - 이미지 특징 추출 함수 (메모리 사용량 개선)
def extract_cnn_features(image, model):
    try:
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = model.predict(image)
        return features.flatten()
    except Exception as e:
        app.logger.error(f"이미지 특징 추출 오류: {str(e)}")
        raise

# 📌 [기존 기능 유지] - 저장된 특징 벡터와 입력 이미지 비교 (예외 처리 강화)
def find_similar_video_from_saved_features(capture_image, save_folder):
    if not os.path.exists(save_folder):
        app.logger.error(f"특징 벡터 저장 폴더가 존재하지 않음: {save_folder}")
        return None, None, None
        
    best_match_score = -1
    best_video = None
    best_frame_time = None

    try:
        capture_img = cv2.imread(capture_image)
        if capture_img is None:
            app.logger.error(f"이미지를 로드할 수 없음: {capture_image}")
            return None, None, None

        # 모델 로드
        current_model = load_model()
        capture_features = extract_cnn_features(capture_img, current_model)

        # 안전한 파일 목록 조회
        feature_files = [f for f in os.listdir(save_folder) 
                        if f.endswith("_features.pkl") and os.path.isfile(os.path.join(save_folder, f))]
        
        for feature_file in feature_files:
            video_name = feature_file.replace("_features.pkl", "")
            feature_path = os.path.join(save_folder, feature_file)

            try:
                with open(feature_path, 'rb') as f:
                    feature_list = pickle.load(f)

                for i, frame_features in enumerate(feature_list):
                    similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]

                    if similarity_score > best_match_score:
                        best_match_score = similarity_score
                        best_video = video_name
                        best_frame_time = i * 3
            except Exception as e:
                app.logger.error(f"특징 파일 처리 중 오류 ({feature_file}): {str(e)}")
                continue

        return best_video, best_match_score, best_frame_time
        
    except Exception as e:
        app.logger.error(f"유사 영상 검색 중 오류: {str(e)}")
        return None, None, None

# 📌 [기존 기능 유지] - 영상 검색 API (보안 강화)
@app.route('/find_similar_video', methods=['POST'])
@limiter.limit("10 per minute")
def find_similar_video():
    try:
        if not request.json or 'image_url' not in request.json:
            return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400
            
        img_url = request.json.get('image_url')
        
        # 이미지 형식 검증
        if not img_url.startswith('data:image/'):
            return jsonify({'error': '유효하지 않은 이미지 형식입니다.'}), 400
            
        # Base64 데이터 추출 및 디코딩
        try:
            base64_string = img_url.split(',')[1]
            img_data = base64.b64decode(base64_string)
        except (IndexError, base64.binascii.Error):
            return jsonify({'error': '이미지 디코딩 실패'}), 400

        # 임시 파일 생성 - 고유한 이름으로 저장하여 동시 요청 처리
        temp_img_path = f"temp_image_{secrets.token_hex(8)}.jpg"
        try:
            with open(temp_img_path, "wb") as file:
                file.write(img_data)
                
            # 이미지 유효성 검사
            test_img = cv2.imread(temp_img_path)
            if test_img is None or test_img.size == 0:
                return jsonify({'error': '유효하지 않은 이미지 데이터입니다.'}), 400
                
            save_folder = "./data"
            best_video, best_match_score, best_frame_time = find_similar_video_from_saved_features(temp_img_path, save_folder)

            if best_video is None or best_frame_time is None:
                return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404
            
            # YouTube ID 추출
            matches = re.findall(r'\[([^\]]+)\]', best_video)
            if len(matches) >= 2:
                extracted_id = matches[1]
                youtube_link = f"https://www.youtube.com/watch?v={extracted_id}&t={best_frame_time}"

                app.logger.info(f"유사 영상 검색 성공: {best_video} (점수: {best_match_score:.4f})")
                return jsonify({
                    'best_video': best_video,
                    'best_match_score': float(best_match_score),
                    'best_frame_time': best_frame_time,
                    'youtube_link': youtube_link
                })
            else:
                return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404
                
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    except Exception as e:
        app.logger.error(f"영상 검색 처리 중 오류: {str(e)}")
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

# 📌 [새로운 기능] - 서버 상태 확인 API
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'server_time': time.time()}), 200

# 📌 [새로운 기능] - 404 오류 처리
@app.errorhandler(404)
def not_found(error):
    app.logger.warning(f"404 오류: {request.path} (IP: {request.remote_addr})")
    return jsonify({'error': '요청한 리소스를 찾을 수 없습니다.'}), 404

# 📌 [새로운 기능] - 500 오류 처리
@app.errorhandler(500)
def server_error(error):
    app.logger.error(f"500 오류: {str(error)} (IP: {request.remote_addr})")
    return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

# 📌 [서버 실행]
if __name__ == '__main__':
    save_folder = "./data"
    # 데이터 폴더가 없으면 생성
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # 프로덕션 환경에서는 debug=False로 설정하는 것이 좋습니다
    app.run(host="0.0.0.0", port=44324, debug=False)
