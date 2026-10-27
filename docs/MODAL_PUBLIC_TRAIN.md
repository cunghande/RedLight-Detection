# Fine-tune YOLO11x với dataset công khai trên Modal

Đây là quy trình dùng dữ liệu **đã có bounding box**; không cần tự gán nhãn video
trong dự án. Nguồn là [Accident_Detect](https://github.com/K1ethoang/Accident_Detect),
dataset giao thông Việt Nam được công bố với giấy phép CC BY 4.0. Bản gốc có
2.919 ảnh và bảy lớp `accident, bicycle, bus, car, motorcycle, person, truck`.
Nguồn ảnh nằm ở `training_data/public_accident_detect/dataset/` sau khi tải Git LFS.

Script `training/prepare_camera_dataset.py public` đã chuyển về đúng năm lớp
phương tiện trong dự án, bỏ nhãn `accident` và `person`, rồi chia lại **theo nhóm
video** để giảm trùng cảnh giữa train/val/test. ZIP đã sẵn sàng tại
`training_data/public_traffic_dataset_balanced.zip`:

| Tập | Ảnh |
|---|---:|
| Train | 2.047 |
| Validation | 436 |
| Test | 436 |

Thứ tự lớp: `bicycle=0, car=1, motorcycle=2, bus=3, truck=4`. Tổng số bounding
box tương ứng là 55, 2.241, 1.802, 240 và 892. **Xe đạp chỉ có 55 box** và các
ảnh nguồn chủ yếu phục vụ bài toán tai nạn; vì vậy đây là lần fine-tune thử nghiệm,
chưa bảo đảm tốt hơn trên video vượt đèn đỏ. So sánh trên video camera mục tiêu
trước khi thay model đang dùng. Một số clip gốc có thể vẫn gần nhau dù tên nhóm
khác nhau; phép chia theo tên file không chứng minh độc lập hoàn toàn.

## Chạy trên Modal

1. Mở [Modal Notebooks](https://modal.com/notebooks), upload
   `notebooks/finetune-camera-yolo11x-modal.ipynb`.
2. Chọn GPU H100 và gắn Volume `traffic-training` tại `/mnt/traffic-training`.
3. Trong Files của Volume, tạo thư mục `datasets`; upload
   `training_data/public_traffic_dataset_balanced.zip` vào đó. Notebook đã đặt
   sẵn đường dẫn `/mnt/traffic-training/datasets/public_traffic_dataset_balanced.zip`.
   Mở đúng thư mục `datasets` **trước khi** bấm Upload; upload ở thư mục khác sẽ
   khiến notebook không thấy đường dẫn mặc định. Nếu upload rồi, kiểm tra vị trí
   bằng một cell Python:

   ```python
   from pathlib import Path
   print('Volume:', list(Path('/mnt/traffic-training').glob('*.zip')))
   print('Datasets:', list(Path('/mnt/traffic-training/datasets').glob('*.zip')))
   print('Notebook folder:', list(Path.cwd().glob('*.zip')))
   ```

   Nếu ZIP nằm ở Volume root, sửa `DATASET_ZIP` thành
   `/mnt/traffic-training/public_traffic_dataset_balanced.zip`, hoặc chuyển file
   vào thư mục `datasets`. Bản notebook mới tự tìm file cùng tên trong Volume.
4. Chạy các cell theo thứ tự. Mặc định `PILOT=True`: train 3 epoch để kiểm tra.
5. Nếu pilot ổn, đổi `PILOT=False` và chạy lại từ cell cấu hình. Lượt chính thức
   train 80 epoch, bắt đầu từ `yolo11x.pt` pretrained COCO. Checkpoint tốt nhất
   nằm ở `/mnt/traffic-training/runs/public_traffic_yolo11x_v1_full/weights/best.pt`.
6. Tải `best.pt` về máy, dừng kernel Modal. Trọng số mới chỉ được chọn theo val;
   cell cuối đánh giá trên test sau khi hoàn tất lượt chính thức.

Nếu cần tạo lại bộ dữ liệu từ nguồn công khai:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone --depth 1 https://github.com/K1ethoang/Accident_Detect.git training_data/public_accident_detect
git -C training_data/public_accident_detect lfs pull --include="dataset/**" --exclude=""
python training/prepare_camera_dataset.py public --output training_data/public_labeled_balanced --archive training_data/public_traffic_dataset_balanced.zip
```

Đánh giá trên máy sau khi tải checkpoint:

```powershell
python main.py --video dataset_videos/red3.mp4 --vehicle-weights weights/public_traffic_yolo11x_best.pt --headless
```

Giữ các video camera thực tế ngoài tập train để so sánh với
`weights/coco/yolo11x.pt`. Nếu lớp xe đạp hoặc cảnh bình thường nhận diện kém,
cần thêm một nguồn dữ liệu phù hợp rồi huấn luyện/đánh giá lại.
