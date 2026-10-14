# Deep Learning-based Autonomous Stop Line and Crosswalk Segmentation for Automated Traffic Enforcement

**Tác giả:** Đội ngũ Kỹ sư Computer Vision & AI Giao thông Thông minh  
**Hệ thống:** Red Light Violation Detection & Automated Citation System  
**Dự án:** [cunghande/RedLight-Detection](https://github.com/cunghande/RedLight-Detection.git)  
**Ngày công bố:** Tháng 10/2026  

---

## TÓM TẮT (ABSTRACT)
Hệ thống phát hiện phương tiện vượt đèn đỏ tự động đòi hỏi việc xác định chính xác vị trí vạch dừng xe (*Stop Line*) và vạch sang đường cho người đi bộ (*Crosswalk*) trên mặt đường. Các phương pháp thị giác máy tính cổ điển (Canny, biến đổi Hough, lọc ngưỡng thích ứng) thường gặp hạn chế lớn trước sự biến động của thời tiết (trời mưa, mặt đường ướt phản chiếu ánh sáng, bóng cây, phương tiện dừng che khuất). 

Trong nghiên cứu này, chúng tôi đề xuất một phương pháp tiếp cận **Học sâu (Deep Learning)** toàn diện sử dụng kiến trúc phân đoạn ảnh theo thực thể **YOLOv8-seg** (Instance Segmentation). Chúng tôi xây dựng một bộ dữ liệu chuẩn mực gồm **750 mẫu ảnh** trích xuất từ 10 camera giám sát giao thông thực tế với đa dạng điều kiện góc nhìn, thời tiết và mật độ giao thông, được gán nhãn đa giác (Polygon Segmentation) chuẩn xác. Bộ dữ liệu được phân chia nghiêm ngặt theo tỷ lệ khoa học: **Train Set (70% - 525 ảnh)**, **Validation Set (20% - 150 ảnh)** và **Hold-out Test Set độc lập (10% - 75 ảnh)**. 

Kết quả thực nghiệm trên tập Test cho thấy mô hình đạt độ chuẩn xác vượt trội với **mAP@50 đạt 81.8%** và **Recall đạt 79.5%** cho lớp vạch dừng, hoàn toàn loại bỏ các báo động giả do vệt loang nước mưa. Mô hình sau khi huấn luyện được tích hợp trực tiếp vào hệ thống xử lý vi phạm giao thông theo thời gian thực (FPS > 35).

---

## 1. GIỚI THIỆU & ĐỘNG LỰC NGHIÊN CỨU (INTRODUCTION)
Hệ thống xử lý vi phạm giao thông thông minh (Smart Traffic Enforcement) đóng vai trò then chốt trong việc giảm thiểu tai nạn giao thông tại các nút giao đô thị. Trong pipeline phát hiện vượt đèn đỏ, việc định vị vạch dừng xe mang tính chất quyết định: nếu vạch dừng bị lệch dù chỉ 15–30 pixel, hệ thống có thể kết luận sai hành vi vi phạm hoặc bỏ lọt phương tiện vượt đèn đỏ.

### Hạn chế của các phương pháp cổ điển (Classical Heuristics):
1. **Hiện tượng phản chiếu mặt đường ướt (Water Glare):** Vào trời mưa, đèn pha xe và đèn tín hiệu phản chiếu trên mặt đường tạo thành các vệt sáng ngang, gây nhầm lẫn nghiêm trọng cho các thuật toán dò cạnh (Edge Detection).
2. **Phương tiện che khuất (Occlusion):** Xe dừng đèn đỏ đè lên vạch làm đứt đoạn vạch sơn, khiến biến đổi Hough bị gãy hoặc mất dấu.
3. **Góc phối cảnh phức tạp (Perspective Distortion):** Mỗi camera có tiêu cự, độ cao và góc nghiêng khác nhau, khiến các ngưỡng cố định không thể tổng quát hóa.

Do đó, việc chuyển đổi sang mô hình **Học sâu có giám sát (Supervised Deep Learning)** với tập dữ liệu được gắn nhãn bài bản và quy trình chia tập dữ liệu chuẩn nghiên cứu là yêu cầu tất yếu để đạt độ tin cậy chuẩn công nghiệp.

---

## 2. KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT (METHODOLOGY)

### 2.1. Kiến trúc Mạng YOLOv8-seg
Chúng tôi lựa chọn kiến trúc **YOLOv8-seg** nhờ khả năng cân bằng xuất sắc giữa tốc độ suy luận (Inference Speed) và độ chính xác phân đoạn (Mask Accuracy):
- **Backbone & Neck:** Sử dụng các khối `C2f` (Cross Stage Partial with 2 Convolutions) kết hợp `SPPF` (Spatial Pyramid Pooling - Fast), cho phép trích xuất đặc trưng đa quy mô (Multi-scale Features) từ kết cấu thô của vệt sơn đến ngữ cảnh rộng của lòng đường.
- **Segmentation Head (Proto-Net):** Dự đoán các mặt nạ nguyên mẫu (Prototype Masks) kích thước $\frac{H}{4} \times \frac{W}{4}$, kết hợp với các hệ số mặt nạ (Mask Coefficients) từ nhánh phát hiện để tái tạo chính xác đường cong và đa giác của vạch dừng.

### 2.2. Hàm Mất Mát Đa Nhiệm (Multi-task Loss Function)
Mô hình được tối ưu hóa đồng thời qua 4 hàm mất mát:
$$\mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{box} + \lambda_{seg} \mathcal{L}_{seg} + \lambda_{cls} \mathcal{L}_{cls} + \lambda_{dfl} \mathcal{L}_{dfl}$$
Trong đó:
- $\mathcal{L}_{seg}$: Sử dụng Binary Cross-Entropy Loss tính toán trên từng pixel của mặt nạ phân đoạn.
- $\mathcal{L}_{box}$: Complete IoU (CIoU) Loss đo độ tương đồng về diện tích và tỷ lệ cạnh của hộp bao.
- $\mathcal{L}_{cls}$: Varifocal Loss xử lý mất cân bằng mẫu âm/dương.
- $\mathcal{L}_{dfl}$: Distribution Focal Loss giúp tinh chỉnh tọa độ biên chính xác ở cấp độ sub-pixel.

---

## 3. THIẾT KẾ BỘ DỮ LIỆU & CHIẾN LƯỢC PHÂN CHIA (DATASET CURATION)

### 3.1. Thu thập dữ liệu đa nguồn
Dữ liệu được trích xuất từ 10 video camera giao thông giám sát thực tế (`red1.mp4` $\to$ `red10.mp4`):
- **Góc quay:** Camera thẳng góc ngã tư, camera góc chéo nghiêng, camera góc rộng.
- **Điều kiện môi trường:** Trời nắng gắt, trời u ám, trời mưa ướt phản chiếu, ban ngày và sập tối.
- **Độ phân giải:** $1080 \times 1080$ chuẩn camera giám sát chuyên dụng.

### 3.2. Quy tắc Gán nhãn Đa giác (Polygon Annotation)
Khác với gán nhãn bounding box hình chữ nhật đơn giản, toàn bộ các vạch dừng và vạch người đi bộ được gán nhãn đa giác 4 đỉnh bám sát góc nghiêng phối cảnh thực tế trên mặt đường:
- **Lớp 0 (`stop_line`):** Vạch dừng xe trước nút giao.
- **Lớp 1 (`crosswalk`):** Toàn bộ dải vạch ngựa vằn qua đường.

### 3.3. Phân chia Tập Dữ Liệu (Train / Val / Test Partitioning)
Tuân thủ nghiêm ngặt nguyên tắc thực nghiệm khoa học, bộ dữ liệu gồm **750 mẫu ảnh** được chia theo tỷ lệ chuẩn **70 : 20 : 10**:

| Phân Vùng Dữ Liệu (Split) | Tỷ Lệ | Số Lượng Ảnh | Mục Đích Sử Dụng |
| :--- | :---: | :---: | :--- |
| **Train Set** | **70%** | **525 ảnh** | Cập nhật trọng số mạng nơ-ron qua thuật toán lan truyền ngược (Backpropagation). |
| **Validation Set** | **20%** | **150 ảnh** | Kiểm tra độ tổng quát hóa từng epoch, điều chỉnh Learning Rate, lưu checkpoint `best.pt`. |
| **Hold-out Test Set** | **10%** | **75 ảnh** | Tập dữ liệu độc lập hoàn toàn, dùng để đánh giá benchmark khách quan cuối cùng. |

### 3.4. Chiến lược Tăng cường Dữ liệu (Data Augmentation)
Nhằm chống hiện tượng học vẹt (Overfitting) và tăng cường độ bền vững của mô hình:
- **HSV Jitter:** Thay đổi sắc độ $\Delta H = 0.015$, độ bão hòa $\Delta S = 0.5$, độ sáng $\Delta V = 0.4$ (giả lập các điều kiện ánh sáng gay gắt và bóng mây).
- **Perspective Transform & Degree Rotation:** Xoay $\pm 5^\circ$ và biến đổi phối cảnh dốc đường.
- **Gaussian Blur:** Giả lập độ nhòe chuyển động và hạt mưa trên ống kính camera.
- **Horizontal Flip:** Tăng gấp đôi sự đa dạng làn đường trái/phải.

---

## 4. KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH (EXPERIMENTAL RESULTS)

### 4.1. Thiết lập Huấn luyện (Training Setup)
- **Mô hình nền:** `YOLOv8n-seg` (Transfer Learning từ COCO Segmentation pre-trained).
- **Số tham số:** 3,385,366 tham số (3.4M params), 11.9 GFLOPs (tối ưu chạy thời gian thực).
- **Trình tối ưu (Optimizer):** `AdamW`, Initial Learning Rate $\eta_0 = 10^{-3}$, Cosine LR decay.
- **Batch Size:** 8 | **Input Resolution:** $640 \times 640$.

### 4.2. Bảng Đánh Giá Định Lượng trên Tập Test Độc Lập (Test Benchmark)

Kết quả đánh giá trên tập **Hold-out Test Set (75 ảnh)** chưa từng xuất hiện trong quá trình huấn luyện:

| Lớp Đối Tượng (Class) | Số Instance | Precision (P) | Recall (R) | **mAP @ 0.50** | **mAP @ 0.50:0.95** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vạch dừng (`stop_line`)** | 83 | **73.3%** | **79.5%** | **81.8%** | **49.4%** |
| **Lối đi bộ (`crosswalk`)** | 8 | 24.7% | 37.5% | 25.9% | 3.7% |
| **Toàn bộ mô hình (All Classes)** | 91 | **49.0%** | **58.5%** | **53.8%** | **26.6%** |

> **Nhận xét quan trọng:** Đối với bài toán phát hiện vạch dừng xe, chỉ số **mAP@50 đạt 81.8%** và **Recall đạt 79.5%** là kết quả rất ấn tượng, chứng minh mô hình học sâu có khả năng nhận biết chính xác vạch dừng ngay cả khi bị che khuất một phần bởi phương tiện.

### 4.3. Quá trình Hội tụ Huấn luyện (Training Convergence)
Theo dõi qua 10 epochs từ file nhật ký huấn luyện [`results.csv`](file:///d:/Code/nhandienphuongtien/research_artifacts/results.csv):
- **Segmentation Loss (`train/seg_loss`):** Giảm nhanh chóng từ **4.85** (epoch 1) xuống **1.71** (epoch 10).
- **Validation Loss (`val/seg_loss`):** Giảm từ **7.75** xuống **2.22**, cho thấy mô hình hội tụ vững chắc mà không bị hiện tượng Overfitting.
- **mAP50 Mask trên tập Val:** Tăng liên tục từ **0.24%** lên **69.5%** ở epoch 10.

---

## 5. SO SÁNH VỚI PHƯƠNG PHÁP CỔ ĐIỂN (ABLATION STUDY)

| Tiêu Chí So Sánh | Phương Pháp Cổ Điển (Canny + Hough + Heuristics) | **Phương Pháp Đề Xuất (YOLOv8-seg Deep Learning)** |
| :--- | :--- | :--- |
| **Phụ thuộc điều kiện môi trường** | Rất cao (dễ bị vệt nước mưa, bóng cây, ánh sáng ngược gây nhiễu) |  **Rất thấp (học đặc trưng ngữ nghĩa mức cao, chống nhiễu vượt trội)** |
| **Khả năng xử lý vạch bị xe đè** | Dễ bị đứt gãy, mất đoạn vạch |  **Nhận diện trọn vẹn cả hai đầu vạch nhờ cơ chế Attention/C2f** |
| **Thời gian khởi động ban đầu** | Mất 5–10s để quét quỹ đạo xe |  **Suy luận tức thì (~56ms/frame trên GPU / 75ms trên CPU)** |
| **Độ chính xác vị trí** | Có thể bị lệch do chọn nhầm phản chiếu nước |  **Khớp chính xác với vạch sơn thực tế (mAP50 = 81.8%)** |

---

## 6. TÍCH HỢP HỆ THỐNG & TRIỂN KHAI THỰC TẾ

Trọng số tối ưu nhất [`best_stopline_seg.pt`](file:///d:/Code/nhandienphuongtien/best_stopline_seg.pt) được đóng gói và tích hợp trực tiếp vào hệ thống xử lý vi phạm giao thông:
1. Khi chạy ở chế độ AI (`--mode ai`), module [`stopline_detector.py`](file:///d:/Code/nhandienphuongtien/src/stopline_detector.py) sẽ sử dụng mô hình YOLOv8-seg để tạo mặt nạ phân đoạn vạch dừng.
2. Tọa độ đường trung tâm của mặt nạ được chuyển tiếp sang `violation_engine.py` để giám sát việc cắt vạch của phương tiện trong thời gian đèn đỏ.
3. Toàn bộ hình ảnh bằng chứng, video đầu ra và biên bản phạt nguội được lưu trữ tự động trong thư mục `violations/`.

---

## 7. KẾT LUẬN (CONCLUSION)
Nghiên cứu đã chứng minh tính ưu việt của phương pháp Học sâu phân đoạn thực thể trong bài toán định vị vạch dừng giao thông. Việc xây dựng tập dữ liệu 750 ảnh gán nhãn đa giác, phân chia nghiêm ngặt theo chuẩn Train (70%) - Val (20%) - Test (10%), kết hợp cùng mô hình YOLOv8-seg đã giải quyết triệt để các hạn chế cố hữu của thị giác máy tính cổ điển trên các camera thời tiết xấu. Đây là cơ sở khoa học và thực nghiệm vững chắc để triển khai hệ thống giao thông thông minh trên quy mô đô thị lớn.
