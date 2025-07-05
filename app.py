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
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
CORS(app)

book_cache = TTLCache(maxsize=1000, ttl=3600)

DB_PATH = "users.db"
DB_LOCK = threading.Lock()

def init_db():
    with DB_LOCK:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loans (
                loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                book_key TEXT NOT NULL,
                loan_date TEXT NOT NULL,
                return_date TEXT NOT NULL,
                status TEXT NOT NULL, -- '대출 중', '반납 완료', '연체'
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        conn.commit()
        conn.close()

init_db()

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

session = requests.Session()

class DatabaseConnection:
    def __enter__(self):
        self.conn = sqlite3.connect(DB_PATH)
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

book_descriptions = {}
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = None
book_keys_in_matrix = []

def update_tfidf_matrix():
    global tfidf_matrix, book_keys_in_matrix
    if book_descriptions:
        book_keys_in_matrix = list(book_descriptions.keys())
        descriptions_list = [book_descriptions[key] for key in book_keys_in_matrix]
        tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions_list)
    else:
        tfidf_matrix = None
        book_keys_in_matrix = []

def get_book_details_for_recommendation(book_key):
    try:
        book_info = fetch_book_info(book_key)
        if book_info and not book_info.get("error"):
            return f"{book_info.get('title', '')} {book_info.get('status', '')}"
    except Exception as e:
        print(f"Error fetching book details for recommendation {book_key}: {e}")
    return None

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    name = data.get("name")
    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력하세요."}), 400

    hashed_password = generate_password_hash(password)
    created_at = time.strftime('%Y-%m-%d')

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

@app.route('/profile', methods=['POST'])
def get_user_profile():
    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "사용자 이름을 제공해야 합니다."}), 400

    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, name, created_at FROM users WHERE username = ?", (username,))
        user_profile = cursor.fetchone()

    if user_profile:
        return jsonify({
            "username": user_profile[0],
            "name": user_profile[1],
            "created_at": user_profile[2]
        }), 200
    else:
        return jsonify({"error": "사용자를 찾을 수 없습니다."}), 404

@app.route('/profile/update', methods=['PUT'])
def update_user_profile():
    data = request.json
    username = data.get("username")
    new_name = data.get("new_name")
    
    if not username or not new_name:
        return jsonify({"error": "사용자 이름과 새 이름을 제공해야 합니다."}), 400

    try:
        with DB_LOCK:
            with DatabaseConnection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET name = ? WHERE username = ?", (new_name, username))
                conn.commit()
                if cursor.rowcount == 0:
                    return jsonify({"error": "사용자를 찾을 수 없거나 업데이트할 내용이 없습니다."}), 404
        return jsonify({"message": "프로필이 성공적으로 업데이트되었습니다."}), 200
    except Exception as e:
        return jsonify({"error": f"프로필 업데이트 중 오류 발생: {str(e)}"}), 500

@app.route('/loan/reserve', methods=['POST'])
def reserve_loan():
    data = request.json
    username = data.get("username")
    book_key = data.get("book_key")

    if not username or not book_key:
        return jsonify({"error": "사용자 이름과 도서 키를 제공해야 합니다."}), 400

    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_exists = cursor.fetchone()
        if not user_exists:
            return jsonify({"error": "존재하지 않는 사용자입니다."}), 404

        book_info = fetch_book_info(book_key)
        if book_info.get("error"):
            return jsonify({"error": "유효하지 않은 도서 키이거나 도서 정보를 가져올 수 없습니다."}), 404
        
        if book_info.get("status") != "이용가능":
            return jsonify({"error": f"현재 '{book_info.get('title')}' 도서는 대출이 불가능합니다. (현재 상태: {book_info.get('status')})"}), 400

        cursor.execute("SELECT * FROM loans WHERE username = ? AND book_key = ? AND status = '대출 중'", (username, book_key))
        existing_loan = cursor.fetchone()
        if existing_loan:
            return jsonify({"error": "이미 대출 중인 도서입니다."}), 400

        loan_date = datetime.now().strftime('%Y-%m-%d')
        return_date = (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d') # 2주 대출

        try:
            with DB_LOCK:
                cursor.execute("""
                    INSERT INTO loans (username, book_key, loan_date, return_date, status)
                    VALUES (?, ?, ?, ?, ?)""",
                    (username, book_key, loan_date, return_date, "대출 중"))
                conn.commit()
            return jsonify({"message": "도서 대출 예약이 완료되었습니다.", "loan_details": {
                "username": username,
                "book_key": book_key,
                "loan_date": loan_date,
                "return_date": return_date,
                "status": "대출 중"
            }}), 201
        except Exception as e:
            return jsonify({"error": f"대출 예약 중 오류 발생: {str(e)}"}), 500

@app.route('/loan/history', methods=['POST'])
def get_loan_history():
    data = request.json
    username = data.get("username")

    if not username:
        return jsonify({"error": "사용자 이름을 제공해야 합니다."}), 400

    with DatabaseConnection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT book_key, loan_date, return_date, status FROM loans WHERE username = ?", (username,))
        loans = cursor.fetchall()

    loan_history = []
    for loan in loans:
        book_key = loan[0]
        book_info = fetch_book_info(book_key)
        book_title = book_info.get("title", "알 수 없는 도서")
        loan_history.append({
            "book_key": book_key,
            "book_title": book_title,
            "loan_date": loan[1],
            "return_date": loan[2],
            "status": loan[3]
        })
    return jsonify({"username": username, "loan_history": loan_history}), 200


@functools.lru_cache(maxsize=1000)
def fetch_book_info(book_key):
    url = f'https://read365.edunet.net/alpasq/api/detail/info?bookKey={book_key}&speciesKey=34169559713&provCode=J10&neisCode=J100000477'
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        res = response.json()
        
        if res and res.get('data'):
            title = res['data'].get('title')
            img = res['data'].get('coverUrl')
            status = res['data'].get('status')
            return {"title": title, "status": status, "img": img, "bookKey": book_key}
        else:
            return {"error": "도서 정보가 없습니다. bookKey가 유효한지 확인하세요.", "bookKey": book_key}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching book info for {book_key}: {e}")
        return {"error": f"도서 정보 검색 중 네트워크 오류: {str(e)}", "bookKey": book_key}
    except Exception as e:
        print(f"Unexpected error in fetch_book_info for {book_key}: {e}")
        return {"error": f"도서 정보 처리 중 오류: {str(e)}", "bookKey": book_key}

@app.route('/search_book', methods=['POST'])
def search_book():
    data = request.json
    book_key = data.get("book_key")
    
    if not book_key:
        return jsonify({"error": "도서 키가 제공되지 않았습니다."}), 400
    
    try:
        book_info = fetch_book_info(book_key)
        if book_info.get("error"):
            return jsonify(book_info), 404
        return jsonify(book_info), 200
    except Exception as e:
        return jsonify({"error": f"도서 정보 검색 중 오류 발생: {str(e)}"}), 500

@app.route('/search_book_name', methods=['POST'])
def search_book_name_api():
    data = request.json
    book_name = data.get("book_name")
    author = data.get("author")
    publisher = data.get("publisher")
    
    if not book_name and not author and not publisher:
        return jsonify({"error": "도서명, 저자, 또는 출판사 중 하나 이상을 제공해야 합니다."}), 400
    
    cache_key = f"search_{book_name or ''}_{author or ''}_{publisher or ''}"
    if cache_key in book_cache:
        return jsonify(book_cache[cache_key]), 200
    
    try:
        results = search_book_name(book_name, author, publisher)
        book_cache[cache_key] = results
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": f"도서명 검색 중 오류 발생: {str(e)}"}), 500

def get_book_keys_for_page(search_term, page):
    search_url = "https://read365.edunet.net/alpasq/api/search"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "searchKeyword": search_term,
        "neisCode": ["J100000477"],
        "provCode": "J10",
        "page": str(page),
        "schoolName": "관양고등학교",
        "coverYn": "N"
    }

    try:
        response = session.post(search_url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json().get("data", {})
        book_list = data.get("bookList", [])
        if not book_list:
            return []
        return [book.get("bookKey") for book in book_list if "bookKey" in book]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results for page {page}, term '{search_term}': {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in get_book_keys_for_page: {e}")
        return []

def search_book_name(book_name, author=None, publisher=None):
    all_book_keys = []
    
    search_term_for_api = book_name if book_name else (author or publisher or "책") 

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(get_book_keys_for_page, search_term_for_api, i+1) for i in range(5)]
        for future in concurrent.futures.as_completed(futures):
            book_keys = future.result()
            if book_keys:
                all_book_keys.extend(book_keys)
    
    all_book_keys = list(set(all_book_keys))

    all_details = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_key = {executor.submit(fetch_book_info, key): key for key in all_book_keys}
        
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                book_detail = future.result()
                if book_detail and not book_detail.get("error"):
                    match = True
                    if author:
                        pass
                    if publisher:
                        pass
                    
                    if match:
                        all_details.append(book_detail)
            except Exception as e:
                all_details.append({
                    "bookKey": key,
                    "error": f"상세 정보 가져오기 실패: {str(e)}"
                })
    
    return {
        "keyword": book_name,
        "total_count": len(all_details),
        "books": all_details,
        "filtered_by": {
            "author": author,
            "publisher": publisher
        }
    }

@app.route('/recommend_books_content_based', methods=['POST'])
def recommend_books_content_based():
    data = request.json
    liked_book_key = data.get("liked_book_key")
    if not liked_book_key:
        return jsonify({"error": "좋아하는 도서 키(liked_book_key)를 제공해야 합니다."}), 400

    liked_book_desc = get_book_details_for_recommendation(liked_book_key)
    if not liked_book_desc or "error" in liked_book_desc:
        return jsonify({"error": f"도서 키 {liked_book_key}의 정보를 가져올 수 없습니다. 유효한 도서 키인지 확인하세요. {liked_book_desc.get('error', '')}"}), 404

    if liked_book_key not in book_descriptions:
        book_descriptions[liked_book_key] = liked_book_desc

    try:
        sample_search_results = search_book_name("소설") 
        for book in sample_search_results.get("books", []):
            key = book.get("bookKey")
            title = book.get("title")
            status = book.get("status")
            if key and title and key not in book_descriptions:
                book_descriptions[key] = f"{title} {status}"
        
        update_tfidf_matrix()
    except Exception as e:
        print(f"Error populating TF-IDF corpus: {e}")
        pass

    if tfidf_matrix is None or liked_book_key not in book_keys_in_matrix:
        return jsonify({"message": "추천을 위한 충분한 도서 정보가 없습니다. 잠시 후 다시 시도해 주세요."}), 500

    try:
        liked_book_idx = book_keys_in_matrix.index(liked_book_key)
        
        cosine_similarities = linear_kernel(tfidf_matrix[liked_book_idx:liked_book_idx+1], tfidf_matrix).flatten()

        related_book_indices = cosine_similarities.argsort()[:-6:-1] 
        
        recommended_books = []
        for i in related_book_indices:
            if i != liked_book_idx:
                recommended_book_key = book_keys_in_matrix[i]
                book_detail = fetch_book_info(recommended_book_key)
                if book_detail and not book_detail.get("error"):
                    recommended_books.append({
                        "bookKey": book_detail["bookKey"],
                        "title": book_detail["title"],
                        "img": book_detail["img"],
                        "status": book_detail["status"],
                        "similarity_score": float(cosine_similarities[i])
                    })
        
        return jsonify({
            "message": "AI 기반 도서 추천",
            "recommended_books": recommended_books
        }), 200

    except Exception as e:
        return jsonify({"error": f"도서 추천 중 오류 발생: {str(e)}"}), 500

def extract_cnn_features(image, model):
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError("이미지를 로드할 수 없습니다.")
    
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image, verbose=0)
    return features.flatten()

feature_cache = {}

def load_all_features(save_folder):
    global feature_cache
    
    if feature_cache:
        return feature_cache
        
    features = {}
    if not os.path.exists(save_folder):
        print(f"Warning: Feature data folder '{save_folder}' does not exist.")
        return features

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

def find_similar_video_from_saved_features(capture_image, save_folder):
    best_match_score = -1
    best_video = None
    best_frame_time = None

    try:
        capture_features = extract_cnn_features(capture_image, model)
    except Exception as e:
        print(f"Error extracting features from capture image: {e}")
        return None, -1, None

    all_features = load_all_features(save_folder)
    
    if not all_features:
        print("No features loaded from save_folder. Cannot find similar video.")
        return None, -1, None

    def process_video(video_name, feature_list):
        best_score_for_video = -1
        best_frame_for_video = None
        
        for i, frame_features in enumerate(feature_list):
            similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]
            
            if similarity_score > best_score_for_video:
                best_score_for_video = similarity_score
                best_frame_for_video = i * 3
        
        return (video_name, best_score_for_video, best_frame_for_video)
    
    num_features = len(all_features)
    max_workers_limit = min(10, num_features if num_features > 0 else 1)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_limit) as executor:
        futures = [executor.submit(process_video, video_name, feature_list) 
                   for video_name, feature_list in all_features.items()]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                video_name, score, frame_time = future.result()
                if score > best_match_score:
                    best_match_score = score
                    best_video = video_name
                    best_frame_time = frame_time
            except Exception as e:
                print(f"Error processing video in thread: {e}")

    return best_video, best_match_score, best_frame_time

@app.route('/find_similar_video', methods=['POST'])
def find_similar_video():
    img_url = request.json.get('image_url')
    if not img_url:
        return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400
    
    try:
        if not img_url.startswith('data:image'):
            return jsonify({'error': '유효하지 않은 이미지 URL 형식입니다. data URL이어야 합니다.'}), 400

        base64_string = img_url.split(',')[1]
        img_data = base64.b64decode(base64_string)
        
        temp_img_path = "temp_uploaded_image.jpg"
        with open(temp_img_path, "wb") as file:
            file.write(img_data)
        
        save_folder = "./data"
        best_video, best_match_score, best_frame_time = find_similar_video_from_saved_features(temp_img_path, save_folder)
        
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        if best_video is None or best_match_score <= 0 or best_frame_time is None:
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
            return jsonify({'message': '유사한 영상을 찾았으나, YouTube 링크를 생성할 수 없습니다 (비디오 이름에서 ID 추출 실패).'}), 404
            
    except IndexError:
        return jsonify({'error': 'Base64 이미지 데이터 파싱 오류입니다. 올바른 data URL 형식인지 확인하세요.'}), 400
    except Exception as e:
        print(f"Error in find_similar_video: {e}")
        return jsonify({'error': f"영상 검색 중 예기치 않은 오류 발생: {str(e)}"}), 500

def preload_resources():
    save_folder = "./data"
    print(f"특징 벡터 로드 중... (폴더: {save_folder})")
    try:
        load_all_features(save_folder)
        print("특징 벡터 미리 로드 완료")
    except Exception as e:
        print(f"특징 벡터 미리 로드 실패: {e}")

    print("추천 시스템을 위한 초기 도서 데이터 로드 중...")
    try:
        sample_book_keys = ["10100238801", "10100238802", "10100238803", "10100238804", "10100238805"]
        for key in sample_book_keys:
            desc = get_book_details_for_recommendation(key)
            if desc and "error" not in desc:
                book_descriptions[key] = desc
        update_tfidf_matrix()
        print("추천 시스템 초기 도서 데이터 로드 및 TF-IDF 매트릭스 생성 완료")
    except Exception as e:
        print(f"추천 시스템 초기 데이터 로드 실패: {e}")

if __name__ == '__main__':
    save_folder = "./data"
    threading.Thread(target=preload_resources).start()
    app.run(host="0.0.0.0", port=44324, debug=True)
