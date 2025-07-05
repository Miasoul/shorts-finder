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
DB_LOCK = threading.Lock() 

def init_db(): 
    """SQLite 데이터베이스 초기화 (회원 테이블 및 대출 테이블 생성)""" 
    with DB_LOCK: 
        conn = sqlite3.connect(DB_PATH) 
        cursor = conn.cursor() 
        
        # 회원 테이블 생성
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS users ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                username TEXT UNIQUE NOT NULL, 
                password TEXT NOT NULL, 
                name TEXT NOT NULL, 
                created_at TEXT NOT NULL 
            ) 
        """) 
        
        # 대출 테이블 생성
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS borrowings ( 
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                user_id TEXT NOT NULL, 
                book_id TEXT NOT NULL, 
                book_title TEXT NOT NULL, 
                borrow_date TEXT NOT NULL, 
                return_date TEXT, 
                status TEXT NOT NULL DEFAULT 'borrowed', 
                FOREIGN KEY (user_id) REFERENCES users (username) 
            ) 
        """) 
        
        conn.commit() 
        conn.close() 

init_db() 

# CNN 모델 로드 
base_model = VGG16(weights='imagenet') 
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output) 

# 세션 재사용 
session = requests.Session() 

# DB 연결 컨텍스트 매니저 
class DatabaseConnection: 
    def __enter__(self): 
        self.conn = sqlite3.connect(DB_PATH) 
        return self.conn 
     
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.conn.close() 

# 📌 회원가입 API 
@app.route('/register', methods=['POST']) 
def register(): 
    data = request.json 
    username = data.get("username") 
    password = data.get("password") 
    name = data.get("name") 
    if not username or not password: 
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400 

    hashed_password = generate_password_hash(password) 
    created_at = time.strftime('%Y-%m-%d')  # 날짜만 저장 

    try: 
        with DB_LOCK: 
            with DatabaseConnection() as conn: 
                cursor = conn.cursor() 
                cursor.execute(""" 
                    INSERT INTO users (username, password, name, created_at) 
                    VALUES (?, ?, ?, ?)""", 
                    (username, hashed_password, name, created_at)) 
                conn.commit() 
        return jsonify({"message": "회원가입 성공!"}), 201 
    except sqlite3.IntegrityError: 
        return jsonify({"error": "이미 존재하는 아이디입니다."}), 400 
# 📌 [관리자 전용 대출 취소 API] - 새로 추가
@app.route('/admin_cancel_borrowing', methods=['POST'])
def admin_cancel_borrowing():
    """관리자가 강제로 대출을 취소하는 API"""
    data = request.json
    user_id = data.get("user_id")
    book_id = data.get("book_id")
    
    if not user_id or not book_id:
        return jsonify({"error": "사용자 ID와 도서 ID가 필요합니다."}), 400
    
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # 대출 기록 확인
            cursor.execute("""
                SELECT id, book_title FROM borrowings 
                WHERE user_id = ? AND book_id = ? AND status = 'borrowed'
            """, (user_id, book_id))
            borrowing = cursor.fetchone()
            
            if not borrowing:
                return jsonify({"error": "해당 대출 기록을 찾을 수 없습니다."}), 404
            
            # 대출 상태를 'admin_cancelled'로 변경 (관리자 취소임을 표시)
            cursor.execute("""
                UPDATE borrowings 
                SET status = 'admin_cancelled', return_date = ?
                WHERE user_id = ? AND book_id = ? AND status = 'borrowed'
            """, (time.strftime('%Y-%m-%d %H:%M:%S'), user_id, book_id))
            
            conn.commit()
            
            return jsonify({
                "message": f"'{borrowing[1]}' 도서의 대출이 관리자에 의해 취소되었습니다.",
                "book_title": borrowing[1],
                "user_id": user_id,
                "cancelled_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }), 200
            
    except Exception as e:
        return jsonify({"error": f"대출 취소 중 오류 발생: {str(e)}"}), 500

# 📌 [관리자 전용 대출 히스토리 조회 API] - 선택사항
@app.route('/admin_borrowing_history', methods=['GET'])
def get_admin_borrowing_history():
    """관리자가 모든 대출 히스토리를 조회하는 API (취소된 것도 포함)"""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # 모든 대출 기록 조회 (취소된 것도 포함)
            cursor.execute("""
                SELECT b.user_id, COALESCE(u.name, b.user_id) as user_name,
                       b.book_id, b.book_title, b.borrow_date, b.return_date, b.status
                FROM borrowings b
                LEFT JOIN users u ON b.user_id = u.username
                ORDER BY b.borrow_date DESC
            """)
            all_records = cursor.fetchall()
        
        history_list = []
        for record in all_records:
            history_list.append({
                "user_id": record[0],
                "user_name": record[1],
                "book_id": record[2],
                "book_title": record[3],
                "borrow_date": record[4],
                "return_date": record[5],
                "status": record[6]
            })
        
        # 상태별 집계
        status_summary = {}
        for record in history_list:
            status = record['status']
            if status not in status_summary:
                status_summary[status] = 0
            status_summary[status] += 1
        
        return jsonify({
            "total_records": len(history_list),
            "history": history_list,
            "status_summary": status_summary
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"히스토리 조회 중 오류 발생: {str(e)}",
            "history": []
        }), 500

# 📌 [관리자 전용 사용자 대출 현황 조회 API] - 선택사항
@app.route('/admin_user_detail/<user_id>', methods=['GET'])
def get_admin_user_detail(user_id):
    """관리자가 특정 사용자의 상세 대출 현황을 조회하는 API"""
    if not user_id:
        return jsonify({"error": "사용자 ID가 필요합니다."}), 400
    
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # 사용자 정보 조회
            cursor.execute("""
                SELECT name, created_at FROM users WHERE username = ?
            """, (user_id,))
            user_info = cursor.fetchone()
            
            # 사용자의 모든 대출 기록 조회
            cursor.execute("""
                SELECT book_id, book_title, borrow_date, return_date, status
                FROM borrowings 
                WHERE user_id = ?
                ORDER BY borrow_date DESC
            """, (user_id,))
            borrowings = cursor.fetchall()
        
        borrowings_list = []
        for borrowing in borrowings:
            borrowings_list.append({
                "book_id": borrowing[0],
                "book_title": borrowing[1],
                "borrow_date": borrowing[2],
                "return_date": borrowing[3],
                "status": borrowing[4]
            })
        
        # 상태별 집계
        status_count = {}
        for borrowing in borrowings_list:
            status = borrowing['status']
            if status not in status_count:
                status_count[status] = 0
            status_count[status] += 1
        
        return jsonify({
            "user_id": user_id,
            "user_name": user_info[0] if user_info else user_id,
            "member_since": user_info[1] if user_info else None,
            "total_borrowings": len(borrowings_list),
            "status_summary": status_count,
            "borrowings": borrowings_list
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"사용자 상세 조회 중 오류 발생: {str(e)}",
            "borrowings": []
        }), 500

# 📌 로그인 API 
@app.route('/login', methods=['POST']) 
def login(): 
    data = request.json 
    username = data.get("username") 
    password = data.get("password") 

    if not username or not password: 
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400 

    with DatabaseConnection() as conn: 
        cursor = conn.cursor() 
        cursor.execute("SELECT password, name, created_at FROM users WHERE username = ?", (username,)) 
        user = cursor.fetchone() 

    if user and check_password_hash(user[0], password): 
        return jsonify({ 
            "message": "로그인 성공!", 
            "name": user[1], 
            "created_at": user[2] 
        }), 200 
    else: 
        return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401 

# 📌 [대출 예약 API] - 수정: 2개 이상 대출 시 제한
@app.route('/borrow_book', methods=['POST'])
def borrow_book():
    data = request.json
    user_id = data.get("user_id")
    book_id = data.get("book_id")
    
    if not user_id or not book_id:
        return jsonify({"error": "사용자 ID와 도서 ID가 필요합니다."}), 400
    
    # 도서 정보 가져오기
    try:
        book_info = fetch_book_info(book_id)
        book_title = book_info.get("title", "Unknown")
    except Exception as e:
        return jsonify({"error": "도서 정보를 가져올 수 없습니다."}), 500
    
    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        
        # 현재 대출 중인 도서 수 확인
        cursor.execute("""
            SELECT COUNT(*) FROM borrowings 
            WHERE user_id = ? AND status = 'borrowed'
        """, (user_id,))
        current_borrowings = cursor.fetchone()[0]
        
        # 2개 이상 대출 중인 경우 제한
        if current_borrowings >= 2:
            return jsonify({"error": "이미 2개 이상의 도서를 대출예약 중입니다. 더 이상 예약할 수 없습니다."}), 400
        
        # 이미 대출 중인지 확인
        cursor.execute("""
            SELECT id FROM borrowings 
            WHERE user_id = ? AND book_id = ? AND status = 'borrowed'
        """, (user_id, book_id))
        existing_borrow = cursor.fetchone()
        
        if existing_borrow:
            return jsonify({"error": "이미 대출예약 중인 도서입니다."}), 400
        
        # 대출 예약 추가
        borrow_date = time.strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            INSERT INTO borrowings (user_id, book_id, book_title, borrow_date, status)
            VALUES (?, ?, ?, ?, 'borrowed')
        """, (user_id, book_id, book_title, borrow_date))
        conn.commit()
    
    return jsonify({"message": "대출 예약이 완료되었습니다."}), 200

# 📌 [사용자 대출 현황 조회 API] - 기존 유지
@app.route('/user_borrowings/<user_id>', methods=['GET'])
def get_user_borrowings(user_id):
    if not user_id:
        return jsonify({"error": "사용자 ID가 필요합니다."}), 400
    
    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT book_id, book_title, borrow_date, return_date, status
            FROM borrowings 
            WHERE user_id = ? AND status = 'borrowed'
            ORDER BY borrow_date DESC
        """, (user_id,))
        borrowings = cursor.fetchall()
    
    borrowings_list = []
    for borrowing in borrowings:
        borrowings_list.append({
            "book_id": borrowing[0],
            "book_title": borrowing[1],
            "borrow_date": borrowing[2],
            "return_date": borrowing[3],
            "status": borrowing[4]
        })
    
    return jsonify({"borrowings": borrowings_list}), 200

# 📌 [모든 학생의 대출 현황 조회 API] - 새로 추가
@app.route('/all_borrowings', methods=['GET'])
def get_all_borrowings():
    """모든 학생의 대출 현황을 조회하는 관리자용 API"""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # 먼저 borrowings 테이블에 데이터가 있는지 확인
            cursor.execute("SELECT COUNT(*) FROM borrowings")
            total_records = cursor.fetchone()[0]
            
            if total_records == 0:
                return jsonify({
                    "total_borrowings": 0,
                    "all_borrowings": [],
                    "user_summary": {},
                    "message": "현재 대출 중인 도서가 없습니다."
                }), 200
            
            # LEFT JOIN으로 변경하여 사용자 정보가 없어도 대출 기록을 가져올 수 있도록 함
            cursor.execute("""
                SELECT b.user_id, COALESCE(u.name, b.user_id) as user_name, 
                       b.book_id, b.book_title, b.borrow_date, b.return_date, b.status
                FROM borrowings b
                LEFT JOIN users u ON b.user_id = u.username
                WHERE b.status = 'borrowed'
                ORDER BY b.borrow_date DESC
            """)
            all_borrowings = cursor.fetchall()
        
        borrowings_list = []
        for borrowing in all_borrowings:
            borrowings_list.append({
                "user_id": borrowing[0],
                "user_name": borrowing[1],
                "book_id": borrowing[2],
                "book_title": borrowing[3],
                "borrow_date": borrowing[4],
                "return_date": borrowing[5],
                "status": borrowing[6]
            })
        
        # 사용자별 대출 수 집계
        user_summary = {}
        for borrowing in borrowings_list:
            user_id = borrowing['user_id']
            if user_id not in user_summary:
                user_summary[user_id] = {
                    "user_name": borrowing['user_name'],
                    "borrow_count": 0,
                    "books": []
                }
            user_summary[user_id]["borrow_count"] += 1
            user_summary[user_id]["books"].append({
                "book_id": borrowing['book_id'],
                "book_title": borrowing['book_title'],
                "borrow_date": borrowing['borrow_date']
            })
        
        return jsonify({
            "total_borrowings": len(borrowings_list),
            "all_borrowings": borrowings_list,
            "user_summary": user_summary
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"대출 현황 조회 중 오류 발생: {str(e)}",
            "total_borrowings": 0,
            "all_borrowings": [],
            "user_summary": {}
        }), 500

# 📌 [디버깅용 API] - 데이터베이스 상태 확인
@app.route('/debug/db_status', methods=['GET'])
def debug_db_status():
    """데이터베이스 상태를 확인하는 디버깅용 API"""
    try:
        with DatabaseConnection() as conn:
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # 사용자 수 확인
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            
            # 전체 대출 기록 수 확인
            cursor.execute("SELECT COUNT(*) FROM borrowings")
            total_borrowings = cursor.fetchone()[0]
            
            # 현재 대출 중인 기록 수 확인
            cursor.execute("SELECT COUNT(*) FROM borrowings WHERE status = 'borrowed'")
            active_borrowings = cursor.fetchone()[0]
            
            # 샘플 대출 기록 확인
            cursor.execute("SELECT * FROM borrowings LIMIT 5")
            sample_borrowings = cursor.fetchall()
            
        return jsonify({
            "tables": [table[0] for table in tables],
            "user_count": user_count,
            "total_borrowings": total_borrowings,
            "active_borrowings": active_borrowings,
            "sample_borrowings": sample_borrowings
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"데이터베이스 상태 확인 중 오류 발생: {str(e)}"
        }), 500
@app.route('/cancel_borrowing', methods=['POST'])
def cancel_borrowing():
    data = request.json
    user_id = data.get("user_id")
    book_id = data.get("book_id")
    
    if not user_id or not book_id:
        return jsonify({"error": "사용자 ID와 도서 ID가 필요합니다."}), 400
    
    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        
        # 대출 기록 확인
        cursor.execute("""
            SELECT id FROM borrowings 
            WHERE user_id = ? AND book_id = ? AND status = 'borrowed'
        """, (user_id, book_id))
        borrowing = cursor.fetchone()
        
        if not borrowing:
            return jsonify({"error": "대출 기록을 찾을 수 없습니다."}), 404
        
        # 대출 상태를 'cancelled'로 변경
        cursor.execute("""
            UPDATE borrowings 
            SET status = 'cancelled', return_date = ?
            WHERE user_id = ? AND book_id = ? AND status = 'borrowed'
        """, (time.strftime('%Y-%m-%d %H:%M:%S'), user_id, book_id))
        
        conn.commit()
    
    return jsonify({"message": "대출 예약이 취소되었습니다."}), 200

# 앱 실행은 따로 구성 (예: run.py 등에서) 
# if __name__ == '__main__': 
#     app.run(debug=True) 

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

    return {"title": title, "status": status, "img": img, "id": book_key} 

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
    if os.path.exists(save_folder):
        load_all_features(save_folder) 
        print("🚀 특징 벡터 미리 로드 완료")
    else:
        print("⚠️  ./data 폴더가 존재하지 않습니다. 영상 검색 기능은 사용할 수 없습니다.")

# 📌 서버 실행 
if __name__ == '__main__': 
    save_folder = "./data" 
    # 백그라운드에서 리소스 미리 로드 
    threading.Thread(target=preload_resources).start() 
    app.run(host="0.0.0.0", port=44324, debug=True)
