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
import webbrowser
from flask_cors import CORS
import base64
# Flask 앱 초기화
from PIL import Image

def crop_img(img_path):
    """
    이미지의 위아래 10%를 자르고 원본 파일에 덮어쓰는 함수
    
    Parameters:
    img_path (str): 원본 이미지 경로
    """
    try:
        # 이미지 열기
        img = Image.open(img_path)
        
        # 이미지 크기 확인
        width, height = img.size
        
        # 위아래 10% 계산
        top_crop = int(height * 0.1)
        bottom_crop = int(height * 0.9)  # 아래 10%를 자르기 위한 위치
        
        # 이미지 자르기 (left, top, right, bottom)
        cropped_img = img.crop((0, top_crop, width, bottom_crop))
        
        # 원본 파일에 덮어쓰기
        cropped_img.save(img_path)
        print(f"이미지가 성공적으로 처리되었습니다: {img_path}")
        
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
    return img_path

    

app = Flask(__name__)
CORS(app)  # CORS 허용


# CNN 모델 준비 (VGG16 모델 사용, FC 층 제거)
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# CNN을 이용하여 이미지의 특징 벡터를 추출하는 함수
def extract_cnn_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# 저장된 특징 벡터와 입력 이미지 비교
def find_similar_video_from_saved_features(capture_image, save_folder):
    best_match_score = -1
    best_video = None
    best_frame_time = None

    # 이미지 로드
    capture_img = cv2.imread(capture_image)
    if capture_img is None:
        return None, None, None

    # 특징 벡터 추출
    capture_features = extract_cnn_features(capture_img, model)

    # 저장된 특징 벡터 비교
    for feature_file in os.listdir(save_folder):
        if feature_file.endswith("_features.pkl"):
            video_name = feature_file.replace("_features.pkl", "")
            feature_path = os.path.join(save_folder, feature_file)

            # 특징 벡터 불러오기
            with open(feature_path, 'rb') as f:
                feature_list = pickle.load(f)

            # 유사도 비교
            for i, frame_features in enumerate(feature_list):
                similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]
                
                # 유사도 점수 업데이트
                if similarity_score > best_match_score:
                    best_match_score = similarity_score
                    best_video = video_name
                    best_frame_time = i * 3  # 프레임 간격 3초 가정

    return best_video, best_match_score, best_frame_time


# API 엔드포인트
@app.route('/find_similar_video', methods=['POST'])
def find_similar_video():
    # 요청에서 이미지 URL을 가져옵니다.
    img_url = request.json.get('image_url')
    base64_string = img_url.split(',')[1]

    # base64 데이터를 디코딩
    img_data = base64.b64decode(base64_string)

    # 이미지를 파일로 저장
    with open("output_image.jpg", "wb") as file:
        file.write(img_data)
   
    if not img_url:
        return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400

    try:
        # 이미지 다운로드
        

        # 이미지 데이터 처리
        img_data = crop_img('C:/Users/wkd18/Desktop/d/dest/py/output_image.jpg')
        best_video, best_match_score, best_frame_time = find_similar_video_from_saved_features('C:/Users/wkd18/Desktop/d/dest/py/output_image.jpg', save_folder)

        if best_video is None or best_frame_time is None:
            return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404
        
        # 유튜브 링크 생성
        matches = re.findall(r'\[([^\]]+)\]', best_video)
        if len(matches) >= 2:
            extracted_id = matches[1]  # 두 번째 대괄호의 내용을 추출
            youtube_link = f"https://www.youtube.com/watch?v={extracted_id}&t={best_frame_time}"

            # JSON 응답으로 변환할 때 float32를 기본 Python float로 변환
            return jsonify({
                'best_video': best_video,
                'best_match_score': float(best_match_score),  # float32 -> float
                'best_frame_time': best_frame_time,
                'youtube_link': youtube_link
            })
        else:
            return jsonify({'message': '유사한 영상을 찾을 수 없습니다.'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API 서버 실행
if __name__ == '__main__':
    save_folder = "./data"  # 미리 저장된 특징 벡터 파일 폴더
    app.run(debug=True, port = 8080)
