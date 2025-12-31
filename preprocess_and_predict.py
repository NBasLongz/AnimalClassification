"""
Script để preprocess ảnh từ Google cho model
Cắt và resize ảnh giống dataset AFHQ
"""
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Thêm backend path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

def detect_and_crop_face(image_path, output_path=None):
    """
    Detect mặt động vật và crop để giống AFHQ dataset
    """
    # Load ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể load ảnh: {image_path}")
        return None
    
    # Convert sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect mặt động vật bằng Haar Cascade
    # Download từ: https://github.com/opencv/opencv/tree/master/data/haarcascades
    cascades = [
        'haarcascade_frontalcatface.xml',  # Mặt mèo
        'haarcascade_frontalface_default.xml',  # Có thể detect một số động vật
    ]
    
    faces = []
    for cascade_file in cascades:
        cascade_path = os.path.join(BASE_DIR, 'cascades', cascade_file)
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            detected = cascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            if len(detected) > 0:
                faces = detected
                break
    
    # Nếu không detect được, crop center
    if len(faces) == 0:
        print("⚠️ Không detect được mặt, crop center square...")
        h, w = img.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        cropped = img_rgb[start_h:start_h+size, start_w:start_w+size]
    else:
        # Lấy face lớn nhất
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        
        # Mở rộng crop một chút (như AFHQ)
        margin = int(max(w, h) * 0.3)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        
        cropped = img_rgb[y1:y2, x1:x2]
        print(f"✓ Detected face at ({x}, {y}, {w}, {h})")
    
    # Resize về 512x512 (như AFHQ)
    resized = cv2.resize(cropped, (512, 512))
    
    # Convert to PIL Image
    result = Image.fromarray(resized)
    
    # Save nếu cần
    if output_path:
        result.save(output_path)
        print(f"✓ Đã lưu ảnh đã xử lý tại: {output_path}")
    
    return result


def predict_with_preprocessing(image_path):
    """
    Predict với preprocessing giống AFHQ
    """
    from api import predict_animal
    
    print(f"\n📸 Xử lý ảnh: {image_path}")
    
    # Preprocess ảnh
    processed_img = detect_and_crop_face(image_path)
    
    if processed_img is None:
        return None
    
    # Predict
    print("\n🔮 Đang dự đoán...")
    result = predict_animal(processed_img)
    
    print("\n📊 Kết quả:")
    for class_name, confidence in sorted(result.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python preprocess_and_predict.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("  python preprocess_and_predict.py test_dog.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        sys.exit(1)
    
    # Tạo output path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_processed.jpg"
    
    # Detect, crop và predict
    result = detect_and_crop_face(image_path, output_path)
    
    if result:
        # Predict
        from api import predict_animal
        predictions = predict_animal(result)
        
        print("\n" + "="*60)
        print("KẾT QUẢ DỰ ĐOÁN")
        print("="*60)
        for class_name, confidence in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(confidence * 50)
            print(f"{class_name:30s} {confidence*100:5.2f}% {bar}")
        print("="*60)
        
        print(f"\n💡 Mẹo: Nếu kết quả không chính xác, thử:")
        print("   - Sử dụng ảnh close-up mặt động vật")
        print("   - Ảnh có background đơn giản")
        print("   - Ảnh sáng, rõ nét")
