# Backend API - Hệ thống Phân Loại Động Vật

## Mô tả
Backend API sử dụng HOG+LBP features và SVM model để phân loại động vật thành 3 loại:
- **Mèo (Cat)**
- **Chó (Dog)** 
- **Động vật hoang dã (Wild)**

## Cấu trúc thư mục

```
backend/
├── api.py              # API module chính
└── README.md          # File này

saved_models/HOG_LBP/
├── svm_model.pkl       # SVM model đã huấn luyện
├── config.json         # Cấu hình model

saved_features/HOG_LBP/
└── hog_lbp_features.pkl # Features và label encoder
```

## Yêu cầu

### Thư viện Python
```bash
pip install numpy opencv-python pillow scikit-image scikit-learn
```

### Dữ liệu cần thiết
- Model đã được huấn luyện: `saved_models/HOG_LBP/svm_model.pkl`
- Features file: `saved_features/HOG_LBP/hog_lbp_features.pkl`

## Cách sử dụng

### 1. Import API module

```python
import sys
sys.path.insert(0, 'path/to/backend')

from api import predict_animal, get_classifier
```

### 2. Dự đoán từ PIL Image

```python
from PIL import Image
from api import predict_animal

# Load ảnh
image = Image.open("path/to/animal.jpg")

# Dự đoán
confidences = predict_animal(image)

# Kết quả
print(confidences)
# Output: {
#     'Mèo (Cat)': 0.7572,
#     'Động vật hoang dã (Wild)': 0.1792,
#     'Chó (Dog)': 0.0637
# }
```

### 3. Sử dụng classifier trực tiếp

```python
from api import get_classifier

# Lấy classifier instance (singleton)
classifier = get_classifier()

# Dự đoán với confidence scores
confidences = classifier.predict(image)

# Chỉ lấy class có xác suất cao nhất
predicted_class = classifier.predict_class(image)
print(predicted_class)  # 'Mèo (Cat)'
```

### 4. Test API

Chạy file `api.py` trực tiếp để test:

```bash
python backend/api.py
```

Output mẫu:
```
============================================================
Testing Animal Classifier API
============================================================
✓ Đã load SVM model từ: .../saved_models/HOG_LBP/svm_model.pkl
✓ Đã load label encoder. Classes: ['cat', 'dog', 'wild']
✓ Target size: (128, 128)
✓ Model loaded successfully!

📸 Testing with image: 0008.png

🔮 Prediction Results:
  Mèo (Cat): 0.7572 (75.72%)
  Động vật hoang dã (Wild): 0.1792 (17.92%)
  Chó (Dog): 0.0637 (6.37%)

============================================================
✓ API is ready to use!
============================================================
```

## Chi tiết kỹ thuật

### Feature Extraction

API sử dụng phương pháp kết hợp HOG và LBP:

1. **HOG (Histogram of Oriented Gradients)**
   - Orientations: 9
   - Pixels per cell: (16, 16)
   - Cells per block: (2, 2)
   - Block normalization: L2-Hys

2. **LBP (Local Binary Pattern)**
   - Radius: 2
   - Points: 16
   - Method: uniform
   - Histogram normalization: L1

### Image Processing Pipeline

1. Chuyển PIL Image sang numpy array
2. Convert RGB → BGR (cho cv2)
3. Resize về (128, 128)
4. Convert sang grayscale
5. Trích xuất HOG features
6. Trích xuất LBP histogram
7. Nối HOG + LBP features
8. Feed vào SVM model

### Model

- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: RBF
- **Feature Vector Size**: ~1782 dimensions
  - HOG: ~1764 features
  - LBP: 18 features

## API Reference

### `AnimalClassifier` Class

#### `__init__()`
Khởi tạo và load model

#### `load_model()`
Load SVM model, label encoder và config

#### `extract_hog_lbp_features(image)`
Trích xuất HOG+LBP features từ ảnh
- **Args**: `image` (PIL.Image hoặc numpy.ndarray)
- **Returns**: numpy array chứa features

#### `predict(image)`
Dự đoán với confidence scores
- **Args**: `image` (PIL.Image hoặc numpy.ndarray)
- **Returns**: dict {class_name: confidence}

#### `predict_class(image)`
Dự đoán class với xác suất cao nhất
- **Args**: `image` (PIL.Image hoặc numpy.ndarray)
- **Returns**: str (tên class)

### Helper Functions

#### `get_classifier()`
Lấy singleton instance của classifier
- **Returns**: AnimalClassifier instance

#### `predict_animal(image)`
Wrapper function cho frontend
- **Args**: `image` (PIL.Image)
- **Returns**: dict {class_name: confidence}

## Xử lý lỗi

API có xử lý các lỗi phổ biến:

```python
try:
    result = predict_animal(image)
except FileNotFoundError as e:
    print(f"Không tìm thấy model file: {e}")
except Exception as e:
    print(f"Lỗi khi dự đoán: {e}")
```

## Performance

- **Load time**: ~1-2 giây (load model + features)
- **Inference time**: ~50-100ms per image
- **Memory**: ~100-200 MB

## Tích hợp với Frontend

Frontend Gradio đã được cấu hình tự động sử dụng API:

```python
# frontend/demo.py
from api import predict_animal as api_predict_animal

def predict_animal(image):
    return api_predict_animal(image)
```

Để chạy frontend demo:
```bash
python frontend/demo.py
```

## Troubleshooting

### Model không load được
```
FileNotFoundError: Không tìm thấy model tại: ...
```
**Giải pháp**: Chạy notebook `HOG_LBP.ipynb` để train và lưu model

### Import error
```
ImportError: No module named 'api'
```
**Giải pháp**: Thêm backend vào sys.path:
```python
import sys
sys.path.insert(0, 'path/to/backend')
```

### Feature dimension mismatch
```
ValueError: X has ... features but model is expecting ...
```
**Giải pháp**: Đảm bảo feature extraction config giống với training (target_size, HOG params, LBP params)

## Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue hoặc liên hệ qua email.
