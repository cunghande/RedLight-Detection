import cv2
import argparse
from pathlib import Path
import config

def select_multi_lines(video_path):
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
    current_line = []
    lines = []
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

    def redraw():
        nonlocal clone
        clone = frame.copy()
        # Ve cac vach da chon xong
        for i, (p1, p2) in enumerate(lines):
            c = colors[i % len(colors)]
            cv2.line(clone, p1, p2, c, 3)
            cv2.putText(clone, f"VACH {i+1}: {p1}->{p2}", (p1[0], max(30, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        # Ve diem dang chon do dang
        if len(current_line) == 1:
            cv2.circle(clone, current_line[0], 6, (0, 255, 255), -1)
        cv2.imshow("Chon Da Vach Dung (Stop Lines)", clone)

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_line.append((x, y))
            if len(current_line) == 2:
                lines.append((current_line[0], current_line[1]))
                print(f"[+] Da chon Vach {len(lines)}: {current_line[0]} -> {current_line[1]}")
                current_line.clear()
            redraw()

    print("\n" + "="*60)
    print("CONG CU CHON DA VACH DUNG (HO TRO 2 HOAC NHIEU VACH)")
    print("="*60)
    print(" 1. Click chuot trai 2 diem de tao Vach 1.")
    print(" 2. Tiep tuc click chuot trai 2 diem de tao Vach 2.")
    print(" 3. Nhan phim [s] de XAC NHAN va LUU toa do.")
    print(" 4. Nhan phim [r] de RESET chon lai tu dau.")
    print(" 5. Nhan phim [q] de THOAT.\n")

    cv2.imshow("Chon Da Vach Dung (Stop Lines)", clone)
    cv2.setMouseCallback("Chon Da Vach Dung (Stop Lines)", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            lines.clear()
            current_line.clear()
            redraw()
            print("[INFO] Da reset, vui long chon lai.")
        elif key == ord("s"):
            if len(lines) > 0:
                print("\n[THANH CONG] Danh sach toa do cac vach dung ban vua chon:")
                for i, (p1, p2) in enumerate(lines):
                    print(f"   Vach {i+1}: [({p1[0]}, {p1[1]}), ({p2[0]}, {p2[1]})]")
                break
            else:
                print("[CANH BAO] Ban chua chon vach nao (can it nhat 2 click chuot).")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cong cu chon da vach dung tren video")
    parser.add_argument("--source", type=str, default="dataset_videos/red1.mp4", help="Duong dan video can chon vach dung")
    args = parser.parse_args()
    select_multi_lines(args.source)
