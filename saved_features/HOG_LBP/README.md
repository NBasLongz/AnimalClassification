# Saved Features - HOG_LBP

Thư mục này chứa features đã trích xuất từ dataset AFHQ.

## Files được tạo sau khi chạy `HOG_LBP.ipynb`:

1. **hog_lbp_features.pkl** (~217 MB)
   - X_train, X_test: HOG+LBP features (shape: ~1782 dimensions)
   - y_train, y_test: Labels
   - label_encoder: LabelEncoder object
   - train_paths, test_paths: Đường dẫn ảnh
   - target_size: (128, 128)

## Cách tạo lại features:

```bash
# Chạy notebook HOG_LBP.ipynb từ đầu
jupyter notebook Notebook/HOG_LBP.ipynb
```

Chạy tất cả cells từ đầu đến phần "Trích xuất đặc trưng".

## Lưu ý:

⚠️ File .pkl KHÔNG được commit lên Git vì quá lớn (> 100MB).
✅ Cần có dataset `Data/afhq_split_80_20/` để tạo lại features.
⏱️ Thời gian trích xuất: ~10-15 phút cho toàn bộ dataset.
