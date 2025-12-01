# Saved Models - HOG_LBP

Thư mục này chứa các model đã train cho HOG+LBP features.

## Models được tạo sau khi chạy `HOG_LBP.ipynb`:

1. **svm_model.pkl** (~93 MB)
   - SVM model với RBF kernel
   - Accuracy: ~95.29%
   - Parameters: C=100, gamma=0.001, probability=True

2. **random_forest_model.pkl** (~79 MB)
   - Random Forest model
   - Parameters: n_estimators=500, max_depth=25

3. **config.json**
   - Cấu hình và tham số tốt nhất

## Cách tạo lại models:

```bash
# Chạy notebook HOG_LBP.ipynb
jupyter notebook Notebook/HOG_LBP.ipynb
```

Hoặc chạy cell retrain SVM trong notebook để tạo model mới.

## Lưu ý:

⚠️ Các file .pkl KHÔNG được commit lên Git vì quá lớn (> 50MB).
✅ Chỉ commit code và config.json.
