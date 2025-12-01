"""
Script kiểm tra model và so sánh accuracy
"""
import pickle
import numpy as np
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(BASE_DIR, "saved_models", "HOG_LBP", "svm_model.pkl")
features_path = os.path.join(BASE_DIR, "saved_features", "HOG_LBP", "hog_lbp_features.pkl")

print("="*70)
print("KIỂM TRA MODEL VÀ SO SÁNH ACCURACY")
print("="*70)

print("\n1. Load model...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"   ✓ Model type: {type(model)}")
print(f"   ✓ Has predict_proba: {hasattr(model, 'predict_proba')}")

if hasattr(model, 'named_steps'):
    print(f"   ✓ Pipeline steps: {list(model.named_steps.keys())}")
    svm = model.named_steps.get('svm')
    if svm:
        print(f"   ✓ SVM probability: {svm.probability}")

print("\n2. Load features đã lưu...")
with open(features_path, 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test']
le = data['label_encoder']

print(f"   ✓ Test set size: {X_test.shape}")
print(f"   ✓ Classes: {list(le.classes_)}")

print("\n3. Đánh giá accuracy trên test set...")
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"   ✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n4. Kiểm tra từng class...")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

print("\n5. Test với 1 ảnh từ test set...")
test_idx = 0  # Ảnh đầu tiên
true_label = le.inverse_transform([y_test[test_idx]])[0]
pred_label = le.inverse_transform([y_pred[test_idx]])[0]

print(f"   True label: {true_label}")
print(f"   Predicted: {pred_label}")
print(f"   Feature vector shape: {X_test[test_idx].shape}")

# So sánh với API prediction
print("\n6. So sánh với API prediction...")
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'backend'))
from api import predict_animal

# Load ảnh từ test set
test_img_path = data['test_paths'][test_idx]
print(f"   Image path: {test_img_path}")

if os.path.exists(test_img_path):
    img = Image.open(test_img_path)
    api_result = predict_animal(img)
    
    print(f"\n   API Prediction:")
    for cls, conf in sorted(api_result.items(), key=lambda x: x[1], reverse=True):
        print(f"      {cls}: {conf:.4f} ({conf*100:.2f}%)")
    
    # So sánh với model prediction trực tiếp
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_test[test_idx:test_idx+1])[0]
        print(f"\n   Model predict_proba:")
        for i, cls in enumerate(le.classes_):
            print(f"      {cls}: {prob[i]:.4f} ({prob[i]*100:.2f}%)")
    else:
        print(f"\n   ⚠ Model không có predict_proba (cần retrain với probability=True)")

print("\n" + "="*70)
print("KẾT LUẬN")
print("="*70)
print(f"• Model accuracy trên test set: {accuracy*100:.2f}%")
if hasattr(model, 'named_steps'):
    svm = model.named_steps.get('svm')
    if svm and not svm.probability:
        print(f"• ⚠ CẢNH BÁO: Model SVM KHÔNG được train với probability=True")
        print(f"• → API đang dùng decision_function + softmax (kém chính xác hơn)")
        print(f"• → Khuyến nghị: Retrain model với probability=True trong HOG_LBP.ipynb")
    else:
        print(f"• ✓ Model SVM có predict_proba - kết quả chính xác")
else:
    print(f"• ⚠ Model không phải Pipeline")
print("="*70)
