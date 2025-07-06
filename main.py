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
import base64
from flask_cors import CORS
import time
import concurrent.futures
import functools
import threading
from cachetools import TTLCache, cached

app = Flask(__name__)
CORS(app)

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

session = requests.Session()

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

    capture_features = extract_cnn_features(capture_image, model)
    
    all_features = load_all_features(save_folder)
    
    def process_video(video_name, feature_list):
        best_score = -1
        best_frame = None
        
        for i, frame_features in enumerate(feature_list):
            similarity_score = cosine_similarity([capture_features], [frame_features])[0][0]
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_frame = i * 3
        
        return (video_name, best_score, best_frame)
    
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

@app.route('/find_similar_video', methods=['POST'])
def find_similar_video():
    img_url = request.json.get('image_url')
    if not img_url:
        return jsonify({'error': '이미지 URL이 제공되지 않았습니다.'}), 400
    
    try:
        base64_string = img_url.split(',')[1]
        img_data = base64.b64decode(base64_string)
        
        temp_img_path = "output_image.jpg"
        with open(temp_img_path, "wb") as file:
            file.write(img_data)
        
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

def preload_resources():
    save_folder = "./data"
    if os.path.exists(save_folder):
        load_all_features(save_folder)
        print("특징 벡터 완료")
    else:
        print("./data 폴더가 존재하지 않습니다. 영상 검색 기능은 사용할 수 없습니다.")

if __name__ == '__main__':
    save_folder = "./data"
    threading.Thread(target=preload_resources).start()
    app.run(host="0.0.0.0", port=44324, debug=True)
