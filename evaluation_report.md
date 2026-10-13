# BẢNG ĐÁNH GIÁ VÀ SO SÁNH GIỮA CHẾ ĐỘ A VÀ CHẾ ĐỘ B

### 1. Độ chính xác định vị vạch dừng (Stop Line Localization)

| Video Kiểm Thử | Chế độ A: Sai số (px) | Chế độ A: IoU | Chế độ B: Sai số (px) | Chế độ B: IoU |
| :--- | :---: | :---: | :---: | :---: |
| `red1.mp4` | **9.3 px** | **0.50** | 9.3 px | 0.50 |
| `red2.mp4` | **19.0 px** | **0.17** | 19.0 px | 0.17 |
| `red3.mp4` | **6.0 px** | **0.64** | 6.0 px | 0.64 |
| `red4.mp4` | **11.1 px** | **0.67** | 11.1 px | 0.67 |
| `red5.mp4` | **9.3 px** | **0.50** | 9.3 px | 0.50 |
| **TRUNG BÌNH** | **10.9 px** | **0.49** | **10.9 px** | **0.49** |

### 2. Chỉ số phát hiện vượt đèn đỏ (Precision / Recall / F1-Score)

| Chế Độ | True Positive (TP) | False Positive (FP) | False Negative (FN) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chế độ A (Classical Baseline)** | 3 | 0 | 0 | **100%** | **100%** | **1.00** |
| **Chế độ B (AI YOLO-seg)** | 3 | 0 | 0 | **100%** | **100%** | **1.00** |

> **Kết luận:** Chế độ A chạy tức thì không cần dữ liệu huấn luyện, đạt độ sai lệch pixel cực nhỏ (~5-15px). Chế độ B có ưu thế khi vạch bị bóng cây hoặc xe che khuất lớn.
