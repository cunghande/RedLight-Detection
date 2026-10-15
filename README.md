# Phát hiện phương tiện vượt đèn đỏ

Dự án nhận diện phương tiện, theo dõi ID theo thời gian, đọc màu đèn giao thông và ghi nhận xe cắt qua vạch dừng khi đèn đỏ. Kết quả dùng để kiểm tra video; cần xem lại bằng chứng trước khi đưa ra kết luận thực tế.

## Cài đặt

```powershell
pip install -r requirements.txt
python download_models.py
```

Kiểm tra GPU NVIDIA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Chạy video mới: tự phát hiện

Chế độ mặc định tự tìm vạch dừng, tự tìm đèn và theo dõi xe. Nó phù hợp để thử nhanh một video chưa từng có cấu hình.

```powershell
python main.py --video dataset_videos/red1.mp4 --device 0 --scene-calibrated --output output_red1_new.mp4
```

Thêm `--headless` nếu chỉ muốn xuất file mà không mở cửa sổ xem trực tiếp:

```powershell
python main.py --video dataset_videos/red1.mp4 --device 0 --headless --output output_red1_new.mp4
```

Lần chạy đầu có thể chậm do quét ứng viên vạch. Cache vạch tự tìm được lưu ở `configs/<tên-video>.json`. Dùng `--redetect` để quét lại:

```powershell
python main.py --video dataset_videos/red1.mp4 --device 0 --redetect --headless
```

## Chạy chính xác hơn: hiệu chỉnh 4 điểm

Với một góc camera cụ thể, hiệu chỉnh thủ công giúp loại bỏ lỗi chọn nhầm vạch hoặc nhầm cụm đèn. Chạy:

```powershell
python main.py --video dataset_videos/red1.mp4 --device 0 --calibrate --output output_red1_new.mp4
```

Trong cửa sổ hiệu chỉnh:

1. Bấm hai đầu của vạch dừng.
2. Bấm hai góc bao sát cụm đèn đang điều khiển luồng xe đó.
3. Nhấn `S` để lưu; `U` bỏ điểm gần nhất; `R` làm lại; `Q` hủy.

Cấu hình gồm đúng bốn điểm. Không cần gán nhãn xe hoặc gán từng khung hình. File cấu hình lưu tại `configs/scenes/` và gắn với đúng video.

Sau khi lưu, chạy bằng cấu hình đã hiệu chỉnh:

```powershell
python main.py --video dataset_videos/red1.mp4 --device 0 --scene-calibrated --output output_red1_new.mp4
```

Thêm `--headless` vào lệnh trên nếu không cần xem video khi xử lý. Nếu có nhiều video mới, thay `red1.mp4` bằng `red2.mp4` đến `red10.mp4` và hiệu chỉnh riêng khi muốn dùng chế độ chính xác hơn.

## Kết quả

- Video đã vẽ bbox, ID, màu đèn, vạch và số vi phạm được lưu theo `--output`.
- Mỗi lần chạy tạo `violations/run_.../` chứa `events.json`, `events.csv`, `events.md`, ảnh bằng chứng và chẩn đoán theo frame.
- Mỗi ID xe chỉ được ghi một lần trong một lượt chạy.

## Thành phần chính

| Đường dẫn | Vai trò |
|---|---|
| `main.py` | Điều phối pipeline video và tham số dòng lệnh |
| `calibrate_scene.py` | Giao diện hiệu chỉnh nhanh vạch và cụm đèn |
| `src/stopline_detector.py` | Tự đề xuất vạch dừng cho video mới |
| `src/traffic_light_detector.py` | Tìm đèn và đọc màu đèn |
| `src/light_classifier.py` | Classifier màu đèn pretrained |
| `src/crossing_tracker.py` | Theo dõi xe cắt đoạn vạch |
| `src/violation_engine.py` | Ghi nhận sự kiện vượt đèn đỏ |
| `weights/` | Trọng số model |

## Kiểm thử

```powershell
python -m pytest tests -q
```
