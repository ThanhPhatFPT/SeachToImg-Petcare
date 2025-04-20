from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import numpy as np
import torch
import clip
from PIL import Image
from fuzzywuzzy import fuzz
import urllib.parse

app = Flask(__name__)
CORS(app, resources={r"/search_similar": {"origins": "http://localhost:5173"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SPRING_API_URL = "http://localhost:8080/api/products/getAllProductss"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Cached data
PRODUCTS = []
FEATURE_MATRIX = None
PRODUCT_KEYS = []


def extract_clip_features_batch(image_paths):
    images = []
    for path in image_paths:
        try:
            img = preprocess(Image.open(path)).unsqueeze(0).to(device)
            images.append(img)
        except Exception as e:
            print(f"Error processing image {path}: {str(e)}")
    if not images:
        raise ValueError("No valid images to process")
    image_tensor = torch.cat(images, dim=0)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
    return features.cpu().numpy()


def extract_clip_features_from_memory(file):
    img = Image.open(file.stream)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img)
    return features.cpu().numpy().flatten()


def initialize_product_features():
    global PRODUCTS, FEATURE_MATRIX, PRODUCT_KEYS
    try:
        response = requests.get(SPRING_API_URL, timeout=10)
        response.raise_for_status()
        PRODUCTS = response.json()
        image_paths = []
        PRODUCT_KEYS = []
        for product in PRODUCTS:
            product_name = product.get("productName", "").lower()
            image_url = product.get("image")
            if not image_url:
                continue
            try:
                parsed_url = urllib.parse.urlparse(image_url)
                file_name = os.path.basename(parsed_url.path)
                temp_path = os.path.join(UPLOAD_FOLDER, file_name)
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                if os.path.exists(temp_path):
                    image_paths.append(temp_path)
                    PRODUCT_KEYS.append((product_name, image_url))
            except Exception as e:
                print(f"Error downloading {image_url}: {str(e)}")

        if image_paths:
            FEATURE_MATRIX = extract_clip_features_batch(image_paths)
            for path in image_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {str(e)}")
        else:
            print("Warning: No valid images downloaded from Spring API")
    except Exception as e:
        print(f"Error in initialize_product_features: {str(e)}")


initialize_product_features()


def get_category_from_name(product):
    # Ưu tiên sử dụng categoryName nếu có
    category = product.get("categoryName", "").strip().lower()
    if category:  # Nếu categoryName tồn tại và không rỗng
        return category

    # Nếu không có categoryName, suy ra từ tên sản phẩm
    name = product.get("productName", "").lower()
    if "iphone" in name or "samsung" in name or "phone" in name:
        return "electronics"
    elif "laptop" in name:
        return "computers"
    elif "thức ăn" in name:
        return "thức ăn"
    return "unknown"


@app.route('/search_similar', methods=['POST'])
def search_similar_products():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        file.stream.seek(0)
        query_features = extract_clip_features_from_memory(file)

        if FEATURE_MATRIX is None or len(PRODUCT_KEYS) == 0:
            return jsonify({'error': 'No product features available'}), 500

        dot_products = np.dot(FEATURE_MATRIX, query_features)
        norms = np.linalg.norm(FEATURE_MATRIX, axis=1) * np.linalg.norm(query_features)
        similarities = dot_products / norms
        similarities = np.nan_to_num(similarities, 0)

        results = []
        for i, (product_name, image_url) in enumerate(PRODUCT_KEYS):
            product = next((p for p in PRODUCTS if p.get("productName", "").lower() == product_name), None)
            if not product:
                continue
            image_similarity = similarities[i]
            if image_similarity >= 0.3:
                results.append({
                    'product': product["productName"],
                    'imageUrls': [product["image"]],
                    'matched_label': "Tương đồng hình ảnh",
                    'similarity': float(image_similarity)
                })
            elif any(keyword in product_name for keyword in ["iphone", "samsung", "laptop", "thức ăn"]):
                results.append({
                    'product': product["productName"],
                    'imageUrls': [product["image"]],
                    'matched_label': "Sản phẩm liên quan",
                    'similarity': 0.15
                })

        filtered_results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:10]

        if not filtered_results:
            fallback_results = [
                                   {
                                       'product': p["productName"],
                                       'imageUrls': [p["image"]],
                                       'matched_label': "Sản phẩm cùng loại",
                                       'similarity': 0.1
                                   }
                                   for p in PRODUCTS if get_category_from_name(p) == get_category_from_name(PRODUCTS[0])
                               ][:5]
            return jsonify({
                'results': fallback_results,
                'message': 'Không tìm thấy sản phẩm giống hệt, hiển thị sản phẩm cùng loại'
            })

        return jsonify({'results': filtered_results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)