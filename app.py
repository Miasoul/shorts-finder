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
import concurrent.futures
import threading
import functools
import json
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# SQLite 데이터베이스 초기화
DB_PATH = "users.db"
CACHE_DB_PATH = "book_cache.db"  # 캐시용 데이터베이스 추가

# 전역 Playwright 인스턴스를 위한 변수
playwright = None
browser = None
browser_lock = threading.Lock()  # 스레드 안전을 위한 락
debug_mode = True  # 디버깅 메시지 출력 여부

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
    
    # 캐시 데이터베이스 초기화
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS book_cache (
            book_key TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            status TEXT NOT NULL,
            img TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    
    if debug_mode:
        print("[DB] 데이터베이스 초기화 완료")

def init_playwright():
    """전역 Playwright 인스턴스 초기화"""
    global playwright, browser
    if playwright is None:
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(
                headless=True,
                args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
            )
            if debug_mode:
                print("[Playwright] 브라우저 인스턴스 성공적으로 생성됨")
        except Exception as e:
            print(f"[Playwright] 브라우저 초기화 오류: {str(e)}")
            if playwright:
                playwright.stop()
            playwright = None
            browser = None
            raise
        
def get_browser():
    """스레드 안전하게 브라우저 인스턴스 반환"""
    with browser_lock:
        if browser is None or playwright is None:
            init_playwright()
        # 간단한 브라우저 상태 검사
        try:
            # 브라우저가 정상인지 테스트
            if browser:
                browser.new_page().close()  # 테스트 페이지 열고 닫기
            else:
                init_playwright()  # 브라우저 재초기화
        except Exception as e:
            if debug_mode:
                print(f"[Playwright] 브라우저 재초기화 필요: {str(e)}")
            # 문제가 있으면 재시작
            cleanup()
            init_playwright()
        return browser

# 서버 시작 시 DB와 Playwright 초기화
init_db()
init_playwright()

# 📌 [회원가입 API] - 기존 코드 유지
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

# 📌 [로그인 API] - 기존 코드 유지
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

# 캐시를 확인하고 있으면 반환, 없으면 None 반환
def get_cached_book_info(book_key):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title, status, img, timestamp FROM book_cache WHERE book_key = ?", 
        (book_key,)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # 캐시가 24시간 이내인지 확인
        cache_time = datetime.strptime(result[3], "%Y-%m-%d %H:%M:%S")
        if datetime.now() - cache_time < timedelta(hours=24):
            return {
                "title": result[0],
                "status": result[1],
                "img": result[2]
            }
    return None

# 캐시에 도서 정보 저장
def cache_book_info(book_key, title, status, img):
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "INSERT OR REPLACE INTO book_cache (book_key, title, status, img, timestamp) VALUES (?, ?, ?, ?, ?)",
        (book_key, title, status, img, timestamp)
    )
    conn.commit()
    conn.close()

# 최적화된 도서 정보 가져오기 함수
def fetch_book_info(book_key):
    """책 정보를 가져오는 함수 (캐싱 적용)"""
    # 캐시에서 먼저 확인
    cached_info = get_cached_book_info(book_key)
    if cached_info:
        return cached_info
    
    # 캐시에 없으면 직접 가져오기
    url = f"https://read365.edunet.net/PureScreen/SearchDetail?bookKey={book_key}&speciesKey=34169559343&provCode=J10&neisCode=J100000477&schoolName=관양고등학교"

    try:
        browser_instance = get_browser()
        page = browser_instance.new_page()
        # 페이지 로드 전 타임아웃 설정
        page.set_default_timeout(10000)  # 10초로 설정
        page.goto(url)
        
        # 페이지가 완전히 로드될 때까지 대기
        page.wait_for_load_state('networkidle')
        
        # CSS 셀렉터를 사용하여 더 안정적으로 선택 (XPath 대신)
        if page.query_selector('.book-detail-info h3') is not None:
            title = page.query_selector('.book-detail-info h3').inner_text().strip()
        else:
            # 대체 XPath 시도
            title = page.locator("//h3[contains(@class, 'tit') or contains(@class, 'title')]").first.inner_text().strip()
            if not title:
                title = page.title()  # 페이지 제목이라도 가져오기
        
        # 대출 상태 확인
        if page.query_selector('.book-detail-thumb .state') is not None:
            status = page.query_selector('.book-detail-thumb .state').inner_text().strip()
        else:
            # 대체 방법으로 상태 확인
            status_elem = page.locator("//*[contains(@class, 'state') or contains(@class, 'status')]").first
            status = status_elem.inner_text().strip() if status_elem.count() > 0 else "상태 정보 없음"
        
        # 이미지 URL 가져오기
        if page.query_selector('.book-detail-thumb img') is not None:
            img = page.query_selector('.book-detail-thumb img').get_attribute('src')
            # 상대 경로인 경우 절대 경로로 변환
            if img and img.startswith('/'):
                img = f"https://read365.edunet.net{img}"
        else:
            # 대체 방법
            img_elem = page.locator("//img[contains(@alt, '책표지') or contains(@class, 'cover')]").first
            img = img_elem.get_attribute('src') if img_elem.count() > 0 else "/api/placeholder/120/180"
        
        page.close()

        # 데이터 유효성 검사 - undefined나 빈 값이 아닌지 확인
        if not title or title == "undefined":
            title = f"도서 {book_key}"
        if not status or status == "undefined":
            status = "상태 정보 없음"
        if not img or img == "undefined" or "undefined" in img:
            img = "/api/placeholder/120/180"  # 기본 placeholder 이미지
        
        result = {"title": title, "status": status, "img": img}
        
        # 결과 캐싱
        cache_book_info(book_key, title, status, img)
        
        return result
    except Exception as e:
        if 'page' in locals():
            page.close()
        raise e

# 병렬로 도서 정보 가져오기
def fetch_book_info_parallel(book_keys, max_workers=10, limit=40):
    """여러 도서 정보를 병렬로 가져오는 함수"""
    # 결과 개수 제한
    book_keys = book_keys[:min(len(book_keys), limit)]
    results = []
    errors = []
    
    if debug_mode:
        print(f"[병렬처리] {len(book_keys)}개 도서 정보 병렬 처리 시작")
    
    # 캐시에서 먼저 확인하여 이미 있는 항목은 건너뛰기
    to_fetch_keys = []
    cached_results = []
    
    for key in book_keys:
        cached = get_cached_book_info(key)
        if cached:
            cached["bookKey"] = key
            cached_results.append(cached)
        else:
            to_fetch_keys.append(key)
    
    if debug_mode:
        print(f"[병렬처리] 캐시에서 {len(cached_results)}개 항목 로드, {len(to_fetch_keys)}개 항목 가져오기 필요")
    
    # 캐시에 없는 항목만 병렬로 가져오기
    if to_fetch_keys:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(fetch_book_info, key): key for key in to_fetch_keys}
            
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    data = future.result()
                    data["bookKey"] = key  # bookKey도 결과에 포함
                    results.append(data)
                except Exception as e:
                    errors.append(f"도서 {key} 처리 중 오류: {str(e)}")
                    # 기본 정보로 대체
                    results.append({
                        "bookKey": key, 
                        "title": f"도서 {key}", 
                        "status": "상태 정보 없음",
                        "img": "/api/placeholder/120/180"
                    })
    
    # 캐시된 결과와 새로 가져온 결과 합치기
    all_results = cached_results + results
    
    if debug_mode and errors:
        print(f"[병렬처리] {len(errors)}개 오류 발생: {errors[:3]}{'...' if len(errors) > 3 else ''}")
        
    return all_results

# 📌 [도서 검색 API] - 단일 도서 검색
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

# 📌 [최적화된 도서명 검색 API]
@app.route('/search_book_name', methods=['POST'])
def search_book_name_api():
    data = request.json
    book_name = data.get("book_name")
    limit = data.get("limit", 40)  # 기본값 40개로 설정
    
    if not book_name:
        return jsonify({"error": "도서명이 제공되지 않았습니다."}), 400
    
    try:
        results = search_book_name(book_name, limit)
        
        # 결과 검증 및 정제
        if "books" in results:
            # undefined 값 확인하고 처리
            for i, book in enumerate(results["books"]):
                if "title" not in book or not book["title"] or book["title"] == "undefined":
                    results["books"][i]["title"] = f"도서 {book.get('bookKey', i+1)}"
                if "status" not in book or not book["status"] or book["status"] == "undefined":
                    results["books"][i]["status"] = "상태 정보 없음"
                if "img" not in book or not book["img"] or book["img"] == "undefined" or "undefined" in book["img"]:
                    results["books"][i]["img"] = "/api/placeholder/120/180"
        
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"도서명 검색 중 오류 발생: {str(e)}"}), 500

def search_book_name(book_name, limit=40):
    """도서명으로 검색하는 함수 (최적화 버전)"""
    try:
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
            "coverYn": "Y"  # 표지 정보도 포함하도록 변경
        }

        # API 응답 시간 설정 및 요청
        response = requests.post(search_url, json=payload, headers=headers, timeout=10)

        if not response.ok:
            return {"error": f"검색 API 요청 실패: {response.status_code} - {response.text}"}

        # Step 2: API 응답에서 도서 정보 추출 및 전처리
        data = response.json().get("data", {})
        book_list = data.get("bookList", [])
        
        # API 응답에서 바로 기본 정보 추출 (표지 URL, 제목 등)
        processed_books = []
        for book in book_list[:min(len(book_list), limit)]:
            book_key = book.get("bookKey")
            if not book_key:
                continue
                
            # 캐시에서 먼저 확인
            cached_info = get_cached_book_info(book_key)
            if cached_info:
                cached_info["bookKey"] = book_key
                processed_books.append(cached_info)
                continue
                
            # 기본 정보 추출 - API 응답에서 가능한 많은 정보 추출
            title = book.get("title", "")
            if not title or title == "undefined":
                title = book.get("bookNm", f"도서 {book_key}")
                
            # 표지 이미지 URL 생성
            img_url = book.get("coverUrl")
            if not img_url or img_url == "undefined":
                # read365의 기본 표지 URL 형식 사용
                img_url = f"https://read365.edunet.net/fileStorage/textbook/textfront/{book_key}.jpg"
            elif img_url.startswith('/'):
                img_url = f"https://read365.edunet.net{img_url}"
                
            # 대출 상태 (API에서 제공하지 않는 경우 직접 가져와야 함)
            status = "상태 정보 로딩 중"
            
            book_info = {
                "bookKey": book_key,
                "title": title.strip() if title else f"도서 {book_key}",
                "status": status,
                "img": img_url
            }
            
            processed_books.append(book_info)
            
            # 결과 캐싱 (상태 정보는 나중에 업데이트)
            if title and img_url:
                cache_book_info(book_key, title.strip(), status, img_url)

        # Step 3: 필요한 경우에만 상태 정보 병렬로 업데이트
        # (이미 캐시된 항목은 건너뜀)
        uncached_books = [book for book in processed_books if book["status"] == "상태 정보 로딩 중"]
        if uncached_books:
            start_time = time.time()
            uncached_keys = [book["bookKey"] for book in uncached_books]
            details = fetch_book_info_parallel(uncached_keys, max_workers=10, limit=limit)
            
            # 상태 정보 업데이트
            for detail in details:
                book_key = detail.get("bookKey")
                if not book_key:
                    continue
                    
                # processed_books에서 해당 책 찾기
                for book in processed_books:
                    if book.get("bookKey") == book_key:
                        # 상태 정보 업데이트
                        if "status" in detail and detail["status"] and detail["status"] != "undefined":
                            book["status"] = detail["status"]
                        
                        # 더 나은 제목이나 이미지가 있으면 업데이트
                        if "title" in detail and detail["title"] and detail["title"] != "undefined" and detail["title"] != f"도서 {book_key}":
                            book["title"] = detail["title"]
                            
                        if "img" in detail and detail["img"] and detail["img"] != "undefined" and "undefined" not in detail["img"]:
                            book["img"] = detail["img"]
                            
                        # 캐시 업데이트
                        cache_book_info(book_key, book["title"], book["status"], book["img"])
                        break
            
            end_time = time.time()
            processing_time = end_time - start_time
        else:
            processing_time = 0

        # Step 4: 결과 딕셔너리 생성 및 반환
        result = {
            "keyword": book_name,
            "total_count": len(processed_books),
            "books": processed_books,
            "processing_time": f"{processing_time:.2f}초"
        }
        
        return result
    except Exception as e:
        import traceback
        return {
            "error": f"도서명 검색 중 오류 발생: {str(e)}",
            "trace": traceback.format_exc(),
            "books": []
        }

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

# 서버 종료 시 리소스 정리 함수
def cleanup():
    global browser, playwright
    if browser:
        browser.close()
    if playwright:
        playwright.stop()

# 프로그램 종료 시 리소스 정리를 위한 atexit 등록
import atexit
atexit.register(cleanup)

# 📌 [기존 기능 유지] - 서버 실행
if __name__ == '__main__':
    save_folder = "./data"
    app.run(host="0.0.0.0", port=44324, debug=True)
