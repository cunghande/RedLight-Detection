# Phát hiện phương tiện vượt đèn đỏ

Pipeline video: **YOLO11x → ByteTrack → vạch dừng + trạng thái đèn → sự kiện nghi vượt đèn đỏ → video/báo cáo**.
`weights/yolo11x.pt` là model phương tiện đã fine-tune trên tập con COCO với năm lớp
`bicycle`, `car`, `motorcycle`, `bus`, `truck`. `weights/yolov8s.pt` là model COCO
đã train sẵn, dùng để tìm cụm đèn (class 9); màu đèn được phân tích bằng HSV.
Notebook phương tiện không huấn luyện model đèn hay vạch dừng. Vạch được lấy từ
cache theo camera hoặc đề xuất bằng CV và quỹ đạo xe; nên xác nhận trước khi chạy.

## Chạy video

```powershell
pip install -r requirements.txt
python preview_stopline.py --video dataset_videos/red1.mp4 --camera-id cam_red1
python main.py --video dataset_videos/red1.mp4 --camera-id cam_red1 --headless --output output_violation.mp4
```

Ở cửa sổ preview, chỉnh vạch bằng hai lần nhấp rồi nhấn `s` để lưu. Nếu thay góc
camera, dùng `--redetect` và xác nhận lại vạch. Cache từ Mask R-CNN cũ được bỏ qua.
Không tái sử dụng cùng `camera-id` cho hai góc nhìn khác nhau.

Video đầu ra được ghi theo `--output`; ảnh và báo cáo sự kiện nằm trong thư mục
`violations/run_.../` riêng cho mỗi lượt chạy.
Đây là kết quả thử nghiệm: cần xem lại vạch, trạng thái đèn và ảnh sự kiện trước khi
kết luận có vi phạm. Khi không nhìn thấy đèn đủ lâu, trạng thái chuyển thành `UNKNOWN`
và hệ thống không kết luận xe vượt đèn đỏ ở những frame đó.

## Huấn luyện và kết quả

Notebook đã chạy trên Modal H100: [`notebooks/train-traffic-3-models-modal-h100.ipynb`](notebooks/train-traffic-3-models-modal-h100.ipynb).
Notebook và `weights/yolo11x.pt` đã được giữ nguyên sau lượt train. ZIP đầy đủ do Modal
xuất chứa SSD300-VGG16 (baseline), YOLOv8s và YOLO11x đã fine-tune,
cùng checkpoint và hình/bảng dùng cho paper. Lưu ZIP gốc ở nơi lưu kết quả ngoài repo.
YOLOv8s trong ZIP là **model 5 lớp phương tiện**, khác với `weights/yolov8s.pt` COCO
80 lớp đang dùng để tìm đèn.

Trên 200 ảnh test COCO subset, mAP50–95 lần lượt là **0,257 (SSD)**,
**0,318 (YOLOv8s)** và **0,309 (YOLO11x)**. Vì vậy kết quả hiện tại chưa cho thấy
YOLO11x chính xác hơn YOLOv8s. Pipeline vẫn chọn YOLO11x theo cấu hình hiện tại;
cần đánh giá trên video camera thực tế trước khi quyết định model triển khai.
Xem [hướng dẫn Modal](docs/MODAL_GUIDE.md), [dataset](docs/DATASET_GUIDE.md),
[paper](docs/PAPER_GUIDE.md).

## Các phần chính

| Đường dẫn | Vai trò |
|---|---|
| `main.py` | Điều phối video, tracking, đèn, vạch và báo cáo |
| `config.yaml` | Trọng số và tham số pipeline |
| `src/stopline_detector.py`, `src/trajectory_analyzer.py` | Đề xuất vạch dừng từ ảnh và quỹ đạo xe |
| `src/traffic_light_detector.py` | Nhận diện cụm đèn và màu đèn |
| `src/violation_engine.py`, `src/report_generator.py` | Kiểm tra xe qua vạch khi đỏ và xuất sự kiện |
| `preview_stopline.py` | Xem, chỉnh và lưu vạch theo camera |
| `dataset_videos/` | Video mẫu; `configs/` là cache vạch theo camera |
| `training/`, `tests/` | Mã huấn luyện tham chiếu và kiểm tra cục bộ |

Kiểm tra mã và pipeline mẫu: `python -m pytest tests -q`.
