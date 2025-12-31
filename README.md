# Animal Classification Project

Dự án phân loại động vật (Chó, Mèo, Động vật hoang dã) sử dụng các kỹ thuật Trích xuất đặc trưng truyền thống (HOG, LBP) kết hợp với mô hình máy học SVM.

##  Tổng quan

Hệ thống bao gồm:
- **Backend**: Xử lý ảnh, trích xuất đặc trưng (HOG + LBP) và dự đoán sử dụng SVM.
- **Frontend**: Giao diện web tương tác được xây dựng bằng Gradio.
- **Notebooks**: Các file Jupyter Notebook dùng để nghiên cứu, trích xuất đặc trưng và huấn luyện mô hình.

**Classes**:
- 🐶 Chó (Dog)
- 🐱 Mèo (Cat)
- 🦁 Động vật hoang dã (Wild)

##  Cấu trúc Dự án

```
AnimalClassfication/
├── backend/
│   ├── api.py              # API xử lý logic chính (Feature Extraction + Prediction)
│   └── README.md           # Tài liệu chi tiết cho Backend
├── frontend/
│   └── demo.py             # Giao diện web (Gradio)
├── Notebook/
│   ├── SIFT.ipynb          # Nghiên cứu đặc trưng SIFT
│   └── HOG_LBP.ipynb       # Huấn luyện model với HOG + LBP
├── saved_models/           # Nơi lưu trữ model đã train
├── saved_features/         # Nơi lưu trữ đặc trưng đã trích xuất
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Tài liệu dự án
```

##  Cài đặt

1. **Clone repository** (nếu có):
   ```bash
   git clone <your-repo-url>
   cd AnimalClassfication
   ```

2. **Cài đặt các thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

##  Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)

Nếu bạn chưa có model trong thư mục `saved_models/`, hãy chạy notebook để huấn luyện:

1. Mở `Notebook/HOG_LBP.ipynb`.
2. Chạy lần lượt các cells để thực hiện:
   - Load dữ liệu.
   - Trích xuất đặc trưng HOG và LBP.
   - Huấn luyện SVM model.
   - Lưu model vào `saved_models/HOG_LBP/svm_model.pkl`.

### 2. Chạy Demo (Web Interface)

Để khởi động giao diện web:

```bash
python frontend/demo.py
```

Truy cập vào đường dẫn hiển thị trên terminal (thường là `http://localhost:7860`) để trải nghiệm.

### 3. Sử dụng Backend API

Bạn có thể test riêng phần backend:

```bash
python backend/api.py
```

##  Chi tiết kỹ thuật

### Feature Extraction
Dự án sử dụng sự kết hợp của hai loại đặc trưng:
- **HOG (Histogram of Oriented Gradients)**: Mô tả hình dạng và biên cạnh của đối tượng.
- **LBP (Local Binary Patterns)**: Mô tả kết cấu (texture) của bề mặt.

### Model
- **Algorithm**: Support Vector Machine (SVM).
- **Kernel**: RBF (Radial Basis Function).

##  Yêu cầu hệ thống

- Python 3.8+
- Các thư viện: numpy, opencv-python, scikit-learn, scikit-image, gradio, pillow.

##  Tác giả

- **Sinh viên thực hiện**: [Nguyễn Bá Long - Nguyễn Công Thiết]
- **Môn học**: Computer Vision (Thị giác máy tính)
- **Trường**: Đại học Công nghệ Thông tin (UIT)

---
*Dự án này là một phần của bài tập môn học Thị giác máy tính.*

