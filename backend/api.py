import os
import sys
import pickle
import json
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog, local_binary_pattern

# Đường dẫn gốc của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến model và features đã lưu
MODELS_DIR = os.path.join(BASE_DIR, "saved_models", "HOG_LBP")
FEATURES_DIR = os.path.join(BASE_DIR, "saved_features", "HOG_LBP")

class AnimalClassifier:
    """
    Lớp để load model HOG+LBP SVM và thực hiện dự đoán động vật
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.target_size = (128, 128)
        self.classes = []
        self.load_model()
    
    def load_model(self):
        """Load SVM model, label encoder và config từ các file đã lưu"""
        
        # Load SVM model
        svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
        if not os.path.exists(svm_path):
            raise FileNotFoundError(f"Không tìm thấy model tại: {svm_path}")
        
        with open(svm_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Đã load SVM model từ: {svm_path}")
        
        # Load label encoder từ features file
        features_path = os.path.join(FEATURES_DIR, "hog_lbp_features.pkl")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Không tìm thấy features tại: {features_path}")
        
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
            self.label_encoder = features_data['label_encoder']
            self.target_size = features_data['target_size']
        
        self.classes = list(self.label_encoder.classes_)
        print(f"✓ Đã load label encoder. Classes: {self.classes}")
        print(f"✓ Target size: {self.target_size}")
        
        # Load config nếu có
        config_path = os.path.join(MODELS_DIR, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"✓ Đã load config: {config}")
    
    def extract_hog_lbp_features(self, image):
        """
        Trích xuất đặc trưng HOG+LBP từ một ảnh
        
        Args:
            image: PIL Image hoặc numpy array
            
        Returns:
            numpy array chứa đặc trưng HOG+LBP đã nối
        """
        # Chuyển PIL Image sang numpy array nếu cần
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Nếu ảnh là RGB, chuyển sang BGR cho cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize ảnh về target_size
        image = cv2.resize(image, self.target_size)
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cấu hình HOG (phải giống với quá trình training)
        hog_args = {
            'orientations': 9,
            'pixels_per_cell': (16, 16),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys',
            'feature_vector': True
        }
        
        # Trích xuất HOG features
        hog_feat = hog(gray, **hog_args)
        
        # Cấu hình LBP (phải giống với quá trình training)
        lbp_radius = 2
        lbp_points = 16
        lbp_method = 'uniform'
        
        # Trích xuất LBP features
        lbp = local_binary_pattern(gray, lbp_points, lbp_radius, lbp_method)
        
        # Tính histogram cho LBP
        n_bins = int(lbp.max() + 1)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # Chuẩn hóa histogram LBP (L1 norm)
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)
        
        # Nối HOG và LBP features
        fusion_feat = np.hstack([hog_feat, lbp_hist])
        
        return fusion_feat
    
    def predict(self, image):
        """
        Dự đoán loại động vật từ ảnh
        
        Args:
            image: PIL Image hoặc numpy array
            
        Returns:
            dict chứa confidence scores cho mỗi class
            Format: {"Chó (Dog)": 0.8, "Mèo (Cat)": 0.15, "Động vật hoang dã (Wild)": 0.05}
        """
        # Trích xuất features
        features = self.extract_hog_lbp_features(image)
        
        # Reshape để phù hợp với input của model (1 sample)
        features = features.reshape(1, -1)
        
        # Model là Pipeline với StandardScaler + SVM
        # Pipeline tự động scale features trước khi predict
        
        # Kiểm tra xem model có hỗ trợ predict_proba không
        if hasattr(self.model, 'predict_proba'):
            # Nếu model được train với probability=True
            probabilities = self.model.predict_proba(features)[0]
        else:
            # Nếu không có predict_proba, dùng decision_function
            # Lưu ý: Pipeline cũng có decision_function
            decision_scores = self.model.decision_function(features)[0]
            
            # Chuyển decision scores thành probabilities bằng softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / exp_scores.sum()
        
        # Map class names sang Vietnamese labels
        class_name_map = {
            'cat': 'Mèo (Cat)',
            'dog': 'Chó (Dog)',
            'wild': 'Động vật hoang dã (Wild)'
        }
        
        # Tạo dictionary confidence scores
        confidences = {}
        for i, class_name in enumerate(self.classes):
            vietnamese_label = class_name_map.get(class_name, class_name)
            confidences[vietnamese_label] = float(probabilities[i])
        
        return confidences
    
    def predict_class(self, image):
        """
        Dự đoán class của ảnh (chỉ trả về class có xác suất cao nhất)
        
        Args:
            image: PIL Image hoặc numpy array
            
        Returns:
            str: Tên class với confidence cao nhất
        """
        confidences = self.predict(image)
        predicted_class = max(confidences, key=confidences.get)
        return predicted_class


# Khởi tạo classifier global để tái sử dụng
_classifier = None

def get_classifier():
    """
    Lấy instance của classifier (singleton pattern)
    """
    global _classifier
    if _classifier is None:
        _classifier = AnimalClassifier()
    return _classifier


def predict_animal(image):
    """
    Hàm prediction chính để sử dụng trong frontend
    
    Args:
        image: PIL Image
        
    Returns:
        dict: Confidence scores cho mỗi class
    """
    classifier = get_classifier()
    return classifier.predict(image)


if __name__ == "__main__":
    # Test code
    print("="*60)
    print("Testing Animal Classifier API")
    print("="*60)
    
    try:
        # Load model
        classifier = get_classifier()
        print("\n✓ Model loaded successfully!")
        print(f"Classes: {classifier.classes}")
        
        # Test với một ảnh nếu có
        test_image_path = os.path.join(BASE_DIR, "Data", "afhq_split_80_20", "test", "cat")
        if os.path.exists(test_image_path):
            test_images = [f for f in os.listdir(test_image_path) if f.lower().endswith(('.jpg', '.png'))]
            if test_images:
                test_img_path = os.path.join(test_image_path, test_images[0])
                print(f"\n📸 Testing with image: {test_images[0]}")
                
                # Load image
                from PIL import Image
                img = Image.open(test_img_path)
                
                # Predict
                result = predict_animal(img)
                print("\n🔮 Prediction Results:")
                for class_name, confidence in sorted(result.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print("\n" + "="*60)
        print("✓ API is ready to use!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
