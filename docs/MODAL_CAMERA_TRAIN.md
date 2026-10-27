# Fine-tune YOLO11x bằng ảnh camera trên Modal

## 1. Chuẩn bị và gán nhãn

Đã trích 67 frame, cách nhau 1 giây, từ 10 clip trong `dataset_videos/` vào
`training_data/to_label/`. File `frames.csv` ghi video và thời điểm gốc của mỗi ảnh.
Các clip hiện chỉ dài tổng cộng khoảng một phút, nên đây là dữ liệu mẫu để kiểm tra
quy trình, chưa đủ cho lượt huấn luyện chính thức. Bổ sung video ở nhiều ngày,
thời điểm, ánh sáng, thời tiết và mật độ xe khác nhau. Trích thêm bằng:

```powershell
python training/prepare_camera_dataset.py extract --videos duong_dan_video --output training_data/to_label_2 --interval 2
```

Đưa ảnh vào công cụ gán nhãn (ví dụ CVAT), vẽ bounding box cho **mọi** phương tiện
thuộc năm lớp: `bicycle` (0), `car` (1), `motorcycle` (2), `bus` (3), `truck` (4).
Xuất ở định dạng YOLO Detection, mỗi ảnh có một file `.txt` cùng tên. Ảnh không có
phương tiện vẫn cần file `.txt` rỗng. Kiểm tra tên lớp và thứ tự ID trong bản xuất;
không dùng nhãn của `dataset_stopline/` vì đó là bài toán vạch dừng.

## 2. Chia tập và tạo ZIP

Chia theo video gốc hoặc phiên/ngày quay trước khi sắp ảnh: khoảng 70% train,
15% val, 15% test. Không chia các frame gần nhau của cùng video ra nhiều tập.
Nếu nhiều clip cắt từ cùng một buổi quay, giữ cả buổi trong cùng một tập. Giữ
nguyên tên frame (ví dụ `red1_0000.jpg`) để công cụ bắt được trường hợp một video
bị chia lẫn. Với tập 2.000 ảnh, mục tiêu là khoảng 1.400/300/300 ảnh.

Sắp ảnh và nhãn đã duyệt vào:

```text
training_data/labeled/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Ví dụ `images/train/red1_0000.jpg` phải có `labels/train/red1_0000.txt`.
Chạy lệnh sau để kiểm tra class ID, tọa độ, file nhãn thiếu, tập rỗng và video bị
chia lẫn, rồi đóng gói `data.yaml` cùng ảnh/nhãn:

```powershell
python training/prepare_camera_dataset.py package --labeled training_data/labeled --output training_data/traffic_dataset.zip
```

Nếu lệnh báo lỗi, sửa dữ liệu rồi chạy lại. Chưa có nhãn thì lệnh sẽ dừng;
không tạo nhãn rỗng cho ảnh có xe.

## 3. Chạy Modal

1. Mở [Modal Notebooks](https://modal.com/notebooks), upload
   `notebooks/finetune-camera-yolo11x-modal.ipynb`.
2. Trong sidebar chọn GPU H100, gắn Volume `traffic-training` tại
   `/mnt/traffic-training` (tạo Volume nếu chưa có).
3. Trong Files của Volume, tạo thư mục `datasets` và upload
   `training_data/traffic_dataset.zip` thành
   `/mnt/traffic-training/datasets/traffic_dataset.zip`.
4. Chạy các cell theo thứ tự. Cell cấu hình mặc định `PILOT=True`: train 3 epoch
   để kiểm tra dữ liệu và môi trường. Checkpoint pilot nằm ở
   `/mnt/traffic-training/runs/camera_yolo11x_v1_pilot/weights/best.pt`.
5. Nếu pilot chạy đúng, đổi `PILOT=False` rồi chạy lại từ cell cấu hình.
   Lượt chính thức bắt đầu từ `yolo11x.pt` pretrained COCO và train 80 epoch;
   `best.pt` nằm ở
   `/mnt/traffic-training/runs/camera_yolo11x_v1_full/weights/best.pt`.
   Đổi `RUN_NAME` khi thay dataset hoặc cấu hình. Cell cuối đánh giá test đúng
   một lần sau lượt chính thức.
6. Tải `best.pt` về máy và dừng kernel Modal. Volume giữ dữ liệu qua các phiên,
   còn kernel đang chạy tiếp tục tính phí compute.

Notebook này chỉ fine-tune YOLO11x cho phương tiện. Nó dùng cùng trọng số gốc
COCO với `weights/coco/yolo11x.pt` trong dự án, không dùng bản đã fine-tune
`weights/yolo11x.pt` làm điểm khởi đầu.

## 4. Chạy thử trọng số mới ở máy cá nhân

Lưu file tải về, ví dụ `weights/camera_yolo11x_best.pt`, rồi chạy cùng video đã
dùng để so sánh model cũ và mới:

```powershell
python main.py --video dataset_videos/red3.mp4 --vehicle-weights weights/camera_yolo11x_best.pt --headless
```

Giữ nguyên cấu hình mặc định cho đến khi kiểm tra bản mới trên video camera
thực tế và đối chiếu số hộp sai, xe bị bỏ sót và sự kiện vi phạm. Tập test dùng
để báo cáo kết quả cuối; chọn checkpoint dựa trên val.
