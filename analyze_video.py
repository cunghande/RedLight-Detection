import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("dataset_videos/red1.mp4")

tracks = {}
f = 0
while cap.isOpened() and f < 100:
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, persist=True, verbose=False)
    if res and len(res) > 0 and res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy.cpu().numpy()
        ids = res[0].boxes.id.int().cpu().numpy()
        cls_ids = res[0].boxes.cls.int().cpu().numpy()
        for box, tid, cid in zip(boxes, ids, cls_ids):
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])
            tracks.setdefault(tid, []).append((f, cx, cy, box))
    f += 1

cap.release()

print("\n=== QUY DAO CAC PHUONG TIEN DI CHUYEN TRONG RED1.MP4 ===")
for tid, pts in sorted(tracks.items()):
    if len(pts) >= 10:
        p_start = pts[0]
        p_end = pts[-1]
        dx = p_end[1] - p_start[1]
        dy = p_end[2] - p_start[2]
        print(f"Xe ID {tid}: Bat dau (f={p_start[0]}, x={p_start[1]}, y={p_start[2]}) -> Ket thuc (f={p_end[0]}, x={p_end[1]}, y={p_end[2]}) | dx={dx}, dy={dy}")
