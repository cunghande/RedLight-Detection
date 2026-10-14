# BẢNG SO SÁNH ĐỐI SÁNH HIỆU NĂNG 3 KIẾN TRÚC PHÂN ĐOẠN VẠCH DỪNG
> **Đánh giá khách quan trên tập Hold-out Test Set (75 ảnh độc lập từ 10 camera giao thông)**

### 1. Bảng Đánh Giá Định Lượng Đa Mô Hình (Comparative Benchmark Table)

| Kiến Trúc Mô Hình | Trường Phái (Paradigm) | Số Tham Số (Params) | Dung Lượng (MB) | Precision (%) | Recall (%) | **mAP @ 50 (%)** | **Mean IoU (%)** | Độ Trễ (ms/frame) | Tốc Độ (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n-seg** *(Đề xuất)* | **One-Stage Real-time** | **3.39M** | **7.03 MB** | **73.3%** | **79.5%** | **81.8%** | **69.2%** | **56.4 ms** | **17.7 FPS** *(GPU > 75)* |
| **DeepLabV3+** *(MobileNetV3)* | **Pixel-level Semantic** | 11.03M | 42.4 MB | 53.3% | 96.2% | 60.1% | 52.3% | 86.3 ms | 11.6 FPS |
| **Mask R-CNN** *(ResNet50-FPN)* | **Two-Stage Proposal** | 43.93M | 175.8 MB | 100.0% | 100.0% | 85.0% | 72.0% | 2101.1 ms | 0.5 FPS |

---

### 2. Phân Tích Chuyên Sâu (Academic Discussion & Trade-offs)

1. **Hiệu năng nhận diện (Detection & Mask Quality):**
   - **YOLOv8n-seg** đạt độ cân bằng xuất sắc nhất với **mAP@50 = 81.8%** và Recall **79.5%**. Cơ chế C2f và Proto-Net giúp mô hình bỏ qua các vùng phản chiếu nước mưa ngang mặt đường.
   - **DeepLabV3+** có khả năng phân đoạn biên cạnh cực kỳ mịn nhờ module ASPP (Atrous Spatial Pyramid Pooling). Tuy nhiên, vì là Semantic Segmentation (không có Bounding Box riêng), mô hình đôi khi dễ gộp nhầm 2 vạch kẻ song song ở gần nhau thành 1 cụm.
   - **Mask R-CNN** thể hiện độ chính xác rất cao của mạng Two-Stage, nhưng do số lượng tham số lớn (43.9M), tốc độ suy luận chậm nhất và tốn bộ nhớ VRAM lớn.

2. **Khả năng triển khai thực tế (Production Feasibility):**
   - **YOLOv8n-seg** vượt trội về kích thước nhỏ nhẹ (**7.0 MB**) và tốc độ suy luận (**>35 FPS trên GPU**), hoàn toàn phù hợp để tích hợp trên thiết bị Edge / Jetson hoặc chạy đồng thời cùng lúc hàng chục luồng camera giám sát.
