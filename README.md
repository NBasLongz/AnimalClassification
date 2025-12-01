
# Hướng dẫn sử dụng Demo Phân loại Động vật

## 🎯 Tổng quan

Hệ thống demo phân loại động vật sử dụng:
- **Backend**: HOG+LBP features + SVM model
- **Frontend**: Gradio web interface
- **Classes**: Chó (Dog), Mèo (Cat), Động vật hoang dã (Wild)

## 📁 Cấu trúc Project

```
AnimalClassfication/
├── backend/
│   ├── api.py              # Backend API (HOG+LBP + SVM)
│   └── README.md          # Documentation
├── frontend/
│   └── demo.py            # Gradio web interface
├── Data/
│   └── afhq_split_80_20/  # Dataset
├── Notebook/
│   ├── SIFT.ipynb         # SIFT feature extraction
│   └── HOG_LBP.ipynb      # HOG+LBP feature extraction
├── saved_models/
│   ├── SIFT/              # SIFT models
│   └── HOG_LBP/           # HOG+LBP models (SVM)
├── saved_features/
│   ├── SIFT/              # SIFT features
│   └── HOG_LBP/           # HOG+LBP features
└── test_integration.py    # Integration test script
```

## 🚀 Hướng dẫn chạy Demo

### Bước 1: Cài đặt thư viện

```bash
pip install numpy opencv-python pillow scikit-image scikit-learn gradio
```

### Bước 2: Training Model (nếu chưa có)

Chạy notebook `HOG_LBP.ipynb` để train và lưu model:
1. Mở `Notebook/HOG_LBP.ipynb`
2. Run tất cả cells
3. Model sẽ được lưu tại `saved_models/HOG_LBP/svm_model.pkl`

### Bước 3: Test Backend API

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

### Bước 4: Test Integration

```bash
python test_integration.py
```

### Bước 5: Chạy Web Demo

```bash
python frontend/demo.py
```

Demo sẽ mở trong browser tại `http://localhost:7860`

## 🎨 Giao diện Demo

Demo có các tính năng:
- ✅ Upload ảnh động vật
- ✅ Hiển thị kết quả dự đoán với confidence scores
- ✅ Giao diện đẹp với CSS tùy chỉnh
- ✅ Nút xóa ảnh và phân loại lại
- ✅ Responsive design

## 🔧 Chi tiết kỹ thuật

### Backend API (`backend/api.py`)

**Chức năng chính:**
```python
from api import predict_animal

# Input: PIL Image
# Output: {'Mèo (Cat)': 0.75, 'Chó (Dog)': 0.06, 'Động vật hoang dã (Wild)': 0.18}
result = predict_animal(image)
```

**Feature Extraction:**
- HOG: orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2)
- LBP: radius=2, points=16, method='uniform'
- Target size: 128x128 pixels

**Model:**
- Algorithm: Support Vector Machine (SVM)
- Kernel: RBF
- Features: ~1782 dimensions (HOG: 1764 + LBP: 18)

### Frontend Demo (`frontend/demo.py`)

**Tích hợp với Backend:**
```python
# Tự động load backend API
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
from api import predict_animal as api_predict_animal

# Sử dụng trong Gradio
def predict_animal(image):
    return api_predict_animal(image)
```

**Giao diện:**
- Theme: Gradio Soft theme
- Custom CSS: Full-screen white background
- Banner header với thông tin project
- Upload frame với viền nét đứt màu xanh

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model load time | ~1-2 giây |
| Inference time | ~50-100ms per image |
| Memory usage | ~100-200 MB |
| Model accuracy | ~90%+ (trên test set) |

## 🎯 Workflow

```
1. User uploads image
         ↓
2. Frontend (Gradio) receives image
         ↓
3. Call backend API: predict_animal(image)
         ↓
4. Backend API:
   - Resize image to 128x128
   - Extract HOG features
   - Extract LBP features
   - Concatenate features
   - Feed to SVM model
   - Return confidence scores
         ↓
5. Frontend displays results with confidence %
```

## 📝 Ví dụ sử dụng

### Sử dụng API trong code

```python
from PIL import Image
from api import predict_animal

# Load ảnh
image = Image.open("test_cat.jpg")

# Dự đoán
result = predict_animal(image)

# In kết quả
for class_name, confidence in sorted(result.items(), key=lambda x: x[1], reverse=True):
    print(f"{class_name}: {confidence:.2%}")
```

Output:
```
Mèo (Cat): 75.72%
Động vật hoang dã (Wild): 17.92%
Chó (Dog): 6.37%
```

### Batch prediction

```python
from api import get_classifier
import os

classifier = get_classifier()

# Predict nhiều ảnh
image_dir = "path/to/images"
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)
        result = classifier.predict(img)
        predicted = max(result, key=result.get)
        print(f"{filename}: {predicted} ({result[predicted]:.2%})")
```

## 🐛 Troubleshooting

### 1. Model không load được

**Lỗi:**
```
FileNotFoundError: Không tìm thấy model tại: saved_models/HOG_LBP/svm_model.pkl
```

**Giải pháp:**
- Chạy notebook `HOG_LBP.ipynb` để train model
- Đảm bảo đã run đến cell "Save Model"

### 2. Import error

**Lỗi:**
```
ImportError: No module named 'api'
```

**Giải pháp:**
```python
import sys
sys.path.insert(0, 'backend/')
from api import predict_animal
```

### 3. Gradio không cài

**Lỗi:**
```
No module named 'gradio'
```

**Giải pháp:**
```bash
pip install gradio
```

### 4. Feature dimension mismatch

**Lỗi:**
```
ValueError: X has ... features but model is expecting ...
```

**Giải pháp:**
- Đảm bảo config trong `api.py` giống với training
- Check target_size, HOG params, LBP params

## 📚 Tài liệu tham khảo

- [Backend API README](backend/README.md)
- [HOG Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)
- [LBP Documentation](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern)
- [Gradio Documentation](https://gradio.app/docs/)

## 🎓 Credits

- **Dataset**: AFHQ (Animal Faces HQ)
- **Framework**: scikit-learn, scikit-image, OpenCV, Gradio
- **GVHD**: [Tên Giảng Viên]
- **SVTH**: [Tên Của Bạn]

## 📞 Liên hệ

Nếu có thắc mắc hoặc gặp vấn đề, vui lòng tạo issue hoặc liên hệ qua email.

---

**Chúc bạn demo thành công! 🎉**
