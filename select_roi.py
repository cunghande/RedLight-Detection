import cv2
import argparse
from pathlib import Path
import config

def select_points(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo video: {video_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[LOI] Khong the doc frame tu video.")
        return

    clone = frame.copy()
    points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(clone, points[0], points[1], (0, 255, 255), 2)
            cv2.imshow("Chon Vach Dung (Stop Line)", clone)

    print("\n=== HUONG DAN CHON VACH DUNG ===")
    print("1. Click chuot trai 2 diem de tao vach dung (diem bat dau & diem ket thuc).")
    print("2. Nhan phim 's' de LUU toa do.")
    print("3. Nhan phim 'r' de CHON LAI.")
    print("4. Nhan phim 'q' de THOAT.\n")

    cv2.imshow("Chon Vach Dung (Stop Line)", clone)
    cv2.setMouseCallback("Chon Vach Dung (Stop Line)", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clone = frame.copy()
            points.clear()
            cv2.imshow("Chon Vach Dung (Stop Line)", clone)
            print("[INFO] Da reset, vui long click lai 2 diem.")
        elif key == ord("s"):
            if len(points) >= 2:
                p1, p2 = points[0], points[1]
                print(f"[THANH CONG] Toa do vach dung moi: STOP_LINE = [({p1[0]}, {p1[1]}), ({p2[0]}, {p2[1]})]")
                print("Ban co the copy toa do nay vao file config.py")
                break
            else:
                print("[CANH BAO] Can click du 2 diem truoc khi luu!")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cong cu chon toa do vach dung tren video")
    parser.add_argument("--source", type=str, default="demo.mp4", help="Duong dan den video can chon vach dung")
    args = parser.parse_args()
    select_points(args.source)
