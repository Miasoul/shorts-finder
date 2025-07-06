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
from openai import OpenAI
import json

app = Flask(__name__) 
CORS(app) 

# OpenAI 클라이언트 초기화 (새로운 1.0+ 방식)
OPENAI_API_KEY = "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"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 캐시 설정 - 검색 결과를 저장하는 캐시 (TTL: 1시간) 
book_cache = TTLCache(maxsize=1000, ttl=3600) 
ai_response_cache = TTLCache(maxsize=500, ttl=1800)  # AI 응답 캐시 (30분)

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

# 📌 AI 도서 추천 시스템 - 핵심 함수들

def extract_keywords_from_message(user_message):
    """사용자 메시지에서 키워드를 추출하는 함수 (비용 최적화)"""
    cache_key = f"keywords_{hash(user_message)}"
    if cache_key in ai_response_cache:
        return ai_response_cache[cache_key]
    
    try:
        # 새로운 OpenAI 1.0+ API 방식
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "도서 검색용 핵심 키워드 1-3개를 쉼표로 구분해서 추출해줘. 한국어만 사용."
                },
                {
                    "role": "user", 
                    "content": user_message
                }
            ],
            max_tokens=50,  # 토큰 수 제한
            temperature=0.3
        )
        
        keywords = response.choices[0].message.content.strip()
        keyword_list = [k.strip() for k in keywords.split(',')]
        
        # 캐시에 저장
        ai_response_cache[cache_key] = keyword_list
        return keyword_list
        
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        # 백업 키워드 추출 (AI 실패 시)
        return extract_fallback_keywords(user_message)

def extract_fallback_keywords(message):
    """AI API 실패 시 백업 키워드 추출"""
    subjects = ['수학', '과학', '영어', '국어', '사회', '역사', '지리', '물리', '화학', '생물', '여행지리', '독서']
    careers = ['컴퓨터', '의학', '공학', '예술', '경영', '법학', '교육', '디자인', '음악', 'IT', '프로그래밍']
    activities = ['독서활동', '진로', '취미', '학습']
    
    keywords = []
    
    for subject in subjects:
        if subject in message:
            keywords.append(subject)
    
    for career in careers:
        if career in message:
            keywords.append(career)
            
    for activity in activities:
        if activity in message:
            keywords.append(activity)
    
    if not keywords:
        keywords = ['추천']
    
    return keywords[:3]  # 최대 3개

def search_books_by_keywords(keywords):
    """키워드 리스트로 도서를 검색하는 함수"""
    all_books = []
    
    for keyword in keywords:
        try:
            # 기존 search_book_name 함수 재사용
            result = search_book_name(keyword)
            if result.get('books'):
                all_books.extend(result['books'])
        except Exception as e:
            print(f"키워드 '{keyword}' 검색 실패: {e}")
            continue
    
    # 중복 제거 (도서 ID 기준)
    unique_books = {}
    for book in all_books:
        book_id = book.get('id')
        if book_id and book_id not in unique_books:
            unique_books[book_id] = book
    
    return list(unique_books.values())

def generate_ai_recommendation(user_message, keywords, books):
    """AI가 도서 추천 메시지를 생성하는 함수 (비용 최적화)"""
    cache_key = f"recommend_{hash(user_message + str(len(books)))}"
    if cache_key in ai_response_cache:
        return ai_response_cache[cache_key]
    
    try:
        # 도서 정보를 간단히 요약
        book_summary = []
        for book in books[:5]:  # 최대 5개만 처리
            book_summary.append(f"- {book.get('title', 'Unknown')}: {book.get('status', 'Unknown')}")
        
        book_info = "\n".join(book_summary) if book_summary else "검색된 도서가 없습니다."
        
        # 간단한 프롬프트 (토큰 수 최소화)
        prompt = f"""사용자: {user_message}
키워드: {', '.join(keywords)}
검색된 도서:
{book_info}

위 정보를 바탕으로 친근하게 도서를 추천해주세요. 100자 이내로 답변해주세요."""

        # 새로운 OpenAI 1.0+ API 방식
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "친근한 도서 추천 AI입니다. 간단명료하게 추천해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=150,  # 토큰 수 제한
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # 캐시에 저장
        ai_response_cache[cache_key] = ai_response
        return ai_response
        
    except Exception as e:
        print(f"AI 추천 생성 오류: {e}")
        return f"'{', '.join(keywords)}' 관련 도서를 찾았습니다! 아래 추천 도서들을 확인해보세요. 📚"

# 📌 AI 도서 추천 API
@app.route('/ai_book_recommendation', methods=['POST'])
def ai_book_recommendation():
    """AI가 스스로 도서를 검색하고 추천하는 API"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "메시지가 필요합니다."}), 400
        
        # 1단계: 키워드 추출
        keywords = extract_keywords_from_message(user_message)
        
        # 2단계: 키워드로 도서 검색
        books = search_books_by_keywords(keywords)
        
        # 3단계: AI 추천 메시지 생성
        ai_response = generate_ai_recommendation(user_message, keywords, books)
        
        # 4단계: 응답 구성
        return jsonify({
            "ai_response": ai_response,
            "keywords": keywords,
            "books": books[:6],  # 최대 6개 도서만 반환
            "total_found": len(books)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"AI 추천 처리 중 오류 발생: {str(e)}",
            "ai_response": "죄송합니다. 현재 AI 추천 서비스에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
            "books": [],
            "keywords": []
        }), 500

# 📌 기존 API들 (수정 없음)

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

# 📌 서버 실행 
if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=44324, debug=True)
