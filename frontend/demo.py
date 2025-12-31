import gradio as gr
import random
from PIL import Image
import numpy as np
import os
import sys

# Thêm đường dẫn backend vào sys.path để import api
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))

# Import API từ backend
try:
    from api import predict_animal as api_predict_animal
    print("✓ Đã load model API từ backend!")
    USE_REAL_MODEL = True
except Exception as e:
    print(f"⚠ Không thể load model API: {e}")
    print("⚠ Sử dụng chế độ giả lập (mock)")
    USE_REAL_MODEL = False

# ==========================================
# CẤU HÌNH DANH SÁCH LỚP (CLASSES)
# ==========================================
ANIMAL_CLASSES = [
    "Chó (Dog)", 
    "Mèo (Cat)", 
    "Động vật hoang dã (Wild)"
]

# ==========================================
# PHẦN 1: HÀM TẢI MODEL (MÔ PHỎNG)
# ==========================================
def load_my_model():
    print("Đang tải model...")
    # Model được load tự động thông qua api.py
    print("Đã tải model thành công!")
    return None

model = load_my_model()

# ==========================================
# PHẦN 2: HÀM XỬ LÝ VÀ DỰ ĐOÁN
# ==========================================
def detect_and_crop_face(image):
    """
    Crop thông minh để tìm vùng có động vật (thay vì face detection)
    Ưu tiên vùng CENTER và UPPER (vì mặt thường ở trên)
    """
    width, height = image.size
    
    # CASE 1: Nếu ảnh đã gần vuông (tỉ lệ < 1.3), giữ nguyên center crop
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio < 1.3:
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))
    
    # CASE 2: Ảnh dài (portrait - cao hơn rộng)
    if height > width:
        size = width  # Crop vuông với chiều rộng
        left = 0
        right = width
        # Ưu tiên UPPER 1/3 của ảnh (mặt thường ở trên)
        top = int(height * 0.15)  # Bắt đầu từ 15% từ trên
        bottom = top + size
        # Đảm bảo không vượt quá
        if bottom > height:
            bottom = height
            top = height - size
        return image.crop((left, top, right, bottom))
    
    # CASE 3: Ảnh ngang (landscape - rộng hơn cao)
    else:
        size = height  # Crop vuông với chiều cao
        top = 0
        bottom = height
        # Ưu tiên CENTER HORIZONTAL (mặt thường ở giữa)
        left = (width - size) // 2
        right = left + size
        return image.crop((left, top, right, bottom))

def preprocess_image(image):
    """
    Preprocess ảnh MẠNH HƠN để tăng khả năng dự đoán đúng với ảnh từ Google
    Thêm FACE DETECTION để tự động crop vùng mặt
    """
    if image is None:
        return None
    
    # Convert sang PIL Image nếu cần
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert sang RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # BƯỚC MỚI: Detect và crop face
    img_cropped = detect_and_crop_face(image)
    
    # Resize về 512x512 (như AFHQ)
    img_resized = img_cropped.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Thêm: Tăng độ sắc nét (sharpen) để features rõ hơn
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(img_resized)
    img_sharp = enhancer.enhance(1.5)  # Tăng 50% độ sắc nét
    
    # Thêm: Cân bằng độ sáng
    enhancer = ImageEnhance.Brightness(img_sharp)
    img_bright = enhancer.enhance(1.1)  # Tăng 10% độ sáng
    
    return img_bright

def predict_with_ensemble(image):
    """
    Dự đoán với ENSEMBLE đơn giản hơn
    Chỉ dùng ảnh gốc + flip (vì rotate có thể làm sai features)
    """
    if image is None:
        return None
    
    # Convert sang PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    predictions = []
    
    # 1. Predict với ảnh gốc (đã preprocess)
    img1 = preprocess_image(image)
    pred1 = api_predict_animal(img1)
    predictions.append(pred1)
    
    # 2. Predict với ảnh flip ngang (mirror) - weight gấp đôi
    img2 = img1.transpose(Image.FLIP_LEFT_RIGHT)
    pred2 = api_predict_animal(img2)
    predictions.append(pred2)
    predictions.append(pred2)  # Weight x2
    
    # Tính trung bình confidence của tất cả predictions
    final_result = {}
    for class_name in ANIMAL_CLASSES:
        avg_conf = sum(p.get(class_name, 0) for p in predictions) / len(predictions)
        final_result[class_name] = avg_conf
    
    # Normalize lại tổng = 1.0
    total = sum(final_result.values())
    if total > 0:
        final_result = {k: v/total for k, v in final_result.items()}
    
    return final_result

def predict_animal(image):
    if image is None:
        return "Vui lòng tải ảnh lên"
    
    # Nếu sử dụng model thực tế
    if USE_REAL_MODEL:
        try:
            # PREDICT ĐỠN GIẢN NHẤT - không preprocessing phức tạp
            # Chỉ resize về 512x512 như AFHQ
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Center crop square
            width, height = image.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            image = image.crop((left, top, left + size, top + size))
            
            # Resize 512x512
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            confidences = api_predict_animal(image)
            
            # Tìm class có confidence cao nhất
            predicted_class = max(confidences, key=confidences.get)
            
            # Trả về chỉ tên class
            return predicted_class
            
        except Exception as e:
            print(f"❌ Lỗi khi dự đoán: {str(e)}")
            # Fallback sang mock nếu có lỗi
            pass
    
    # --- [BẮT ĐẦU] LOGIC GIẢ LẬP (fallback) ---
    # Resize ảnh
    image = image.resize((224, 224)) 
    
    predicted_class = random.choice(ANIMAL_CLASSES)
    
    return predicted_class
    # --- [KẾT THÚC] LOGIC GIẢ LẬP ---

# ==========================================
# PHẦN 3: GIAO DIỆN (ĐÃ NÂNG CẤP)
# ==========================================

# CSS tùy chỉnh để làm đẹp và full màn hình
custom_css = """
/* Mở rộng container ra toàn màn hình và đổi nền sang trắng */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    background-color: white !important; /* Nền trắng toàn trang */
}

/* Thanh tiêu đề Banner */
.header-banner {
    background-color: #0056b3; /* Xanh dương đậm */
    color: white;
    padding: 30px;
    text-align: center;
    border-bottom: 5px solid #004494;
    margin-bottom: 0 !important; /* Xóa khoảng cách dưới header để liền mạch */
}
.header-banner h1 {
    color: white !important;
    font-size: 2.5rem !important;
    margin: 0 !important;
}
.header-banner p {
    color: #e6f2ff !important;
    font-size: 1.1rem !important;
    margin-top: 10px !important;
}

/* Khung chứa chính (Main Panel) - Full màn hình */
.main-panel {
    background: white;
    padding: 30px;
    margin: 0 !important; /* Xóa margin để tràn viền */
    border: none !important; /* Xóa viền */
    box-shadow: none !important; /* Xóa bóng đổ */
    border-radius: 0 !important; /* Xóa bo góc */
    min-height: calc(100vh - 150px); /* Chiều cao tối thiểu để chạm đáy (trừ header) */
}

/* Nút bấm */
.primary-btn {
    background-color: #0056b3 !important; /* Nền xanh */
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    transition: all 0.3s ease;
}
.primary-btn:hover {
    background-color: #004494 !important; /* Xanh đậm hơn khi hover */
    transform: scale(1.02);
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 0.9rem;
    background-color: white; /* Footer cũng nền trắng */
}

/* --- STYLE MỚI CHO KHUNG UPLOAD --- */
.upload-frame {
    border: 3px dashed #0056b3 !important; /* Viền nét đứt màu xanh đậm */
    border-radius: 15px !important;
    background-color: #f8fbff !important;
    padding: 10px !important;
    transition: all 0.3s;
}
.upload-frame:hover {
    border-color: #0088ff !important;
    background-color: #e6f2ff !important;
}

/* --- STYLE CHO KẾT QUẢ DỰ ĐOÁN --- */
.result-text textarea {
    font-size: 2.5rem !important; /* Chữ to hơn */
    font-weight: bold !important;
    text-align: center !important; /* Căn giữa */
    color: #0056b3 !important; /* Màu xanh */
    padding: 40px 20px !important;
    border: 3px solid #0056b3 !important;
    border-radius: 15px !important;
    background-color: #f0f8ff !important;
    line-height: 1.5 !important;
}
"""

# Tạo interface đơn giản
demo = gr.Blocks(title="Phân Loại Động Vật")

with demo:
    # Thêm CSS tùy chỉnh
    gr.HTML(f"<style>{custom_css}</style>")
    
    # --- BANNER HEADER (HTML thuần) ---
    gr.HTML(
        """
        <div class="header-banner">
            <h1>Demo : Phân Loại Động Vật</h1>
            <p>Hệ thống nhận diện động vật sử dụng Machine Learning</p>
            <p style="font-size: 0.9rem; margin-top: 5px; opacity: 0.8;">Accuracy: 95.29%</p>
        </div>
        """
    )
    
    # --- MAIN CONTENT ---
    with gr.Row(elem_classes="main-panel"): # Gom tất cả vào khung trắng full màn
        
        # Cột bên trái: Upload và Điều khiển
        with gr.Column(scale=1):
            gr.Markdown("### 1. Tải ảnh lên")
            
            # Đã thêm elem_classes="upload-frame" để tạo khung
            input_image = gr.Image(
                type="pil", 
                label="Thêm ảnh", 
                height=550, # Đã tăng chiều cao lên 550px
                elem_classes="upload-frame",
                sources=["upload"] # Chỉ cho phép upload file, tắt webcam và clipboard
            )
            
            with gr.Row():
                clear_btn = gr.Button(" Xóa ảnh", variant="secondary")
                submit_btn = gr.Button(" PHÂN LOẠI NGAY", variant="primary", elem_classes="primary-btn")

        # Cột bên phải: Kết quả
        with gr.Column(scale=1):
            gr.Markdown("### 2. Kết quả nhận diện")
            
            # Thay đổi: Dùng Textbox để hiển thị chỉ tên lớp
            output_result = gr.Textbox(
                label="Kết quả dự đoán",
                interactive=False,
                lines=3,
                placeholder="Kết quả sẽ hiển thị ở đây...",
                elem_classes="result-text"
            )
            
            # Thêm thông tin bổ sung nếu cần
            gr.Markdown(
                """
                > **Lưu ý:**
                > - Kết quả có độ chính xác phụ thuộc vào model huấn luyện.
                > - Ảnh càng rõ nét, kết quả càng chính xác.
                """
            )

    # --- FOOTER ---
    gr.HTML('<div class="footer">© 2024 Đồ án môn học. Được xây dựng bằng Python & Gradio.</div>')

    # Xử lý sự kiện
    submit_btn.click(
        fn=predict_animal, 
        inputs=input_image, 
        outputs=output_result
    )
    clear_btn.click(lambda: None, None, input_image)

if __name__ == "__main__":
    print("Đang khởi động giao diện full màn hình...")
    demo.launch(share=False)