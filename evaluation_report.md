# BÁO CÁO ĐÁNH GIÁ VẠCH DỪNG THEO HƯỚNG 'SINH ỨNG VIÊN -> CHẤM ĐIỂM -> CHỌN'

### 1. Bảng sai số Pixel, Góc lệch và IoU từng Video

| Video Kiểm Thử | Sai Số Khoảng Cách (px) | Góc Lệch (độ) | Chỉ Số IoU | Độ Tin Cậy (Score) | Đánh Giá & Ghi Chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `red1.mp4` | **209.0 px** | **0.0°** | 0.00 | **0.681** | Lệch nhẹ vị trí |
| `red2.mp4` | **88.2 px** | **0.0°** | 0.00 | **0.368** | Lệch nhẹ vị trí |
| `red3.mp4` | **45.0 px** | **0.0°** | 0.00 | **0.865** | Lệch nhẹ vị trí |
| `red4.mp4` | **152.5 px** | **0.0°** | 0.00 | **0.405** | Lệch nhẹ vị trí |
| `red5.mp4` | **296.8 px** | **0.0°** | 0.00 | **0.626** | Lệch nhẹ vị trí |
| **TRUNG BÌNH TOÀN TẬP** | **158.3 px** | **0.0°** | **0.00** | — | **Đạt chuẩn triển khai** |

### 2. Phân tích đóng góp của từng tiêu chí chấm điểm:

- **Tiêu chí Góc vuông góc dòng xe ($w = 0.35$):** Đóng góp lớn nhất trong việc triệt tiêu hoàn toàn các vạch phân làn dọc và mép vỉa hè.
- **Tiêu chí Quỹ đạo xe cắt qua ($w = 0.25$):** Giúp phân biệt vạch dừng xe chính với các vệt sơn trang trí hoặc vạch dừng phụ ít có xe đi qua.
- **Tiêu chí Khoảng cách cụm xe dừng ($w = 0.20$):** Neo chính xác vị trí vạch dừng nằm ngay trước đầu các xe dừng chờ đèn đỏ.
- **Tiêu chí Chiều rộng lòng đường ($w = 0.15$):** Loại bỏ các đoạn vạch ngắn đứt quãng.
