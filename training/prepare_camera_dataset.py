"""Extract frames for annotation and package a labeled YOLO dataset for Modal."""

import argparse
import csv
import math
import random
import shutil
import zipfile
from pathlib import Path

import cv2


CLASSES = ("bicycle", "car", "motorcycle", "bus", "truck")
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
PUBLIC_CLASS_MAP = {1: 0, 3: 1, 4: 2, 2: 3, 6: 4}


def extract(videos: Path, output: Path, interval: float) -> None:
    if interval <= 0:
        raise ValueError("interval must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "frames.csv"
    if manifest.exists():
        raise FileExistsError(f"{manifest} exists; choose a new output directory")
    rows = []
    for video in sorted(videos.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not fps or not total:
            cap.release()
            raise ValueError(f"Cannot read {video}")
        count = math.ceil(total / fps / interval)
        for index in range(count):
            second = index * interval
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            image = output / f"{video.stem}_{index:04d}.jpg"
            if image.exists():
                raise FileExistsError(image)
            if not cv2.imwrite(str(image), frame):
                raise OSError(f"Cannot write {image}")
            rows.append((image.name, video.name, f"{second:.2f}"))
        cap.release()
    if not rows:
        raise ValueError(f"No MP4 frames found in {videos}")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "video", "second"))
        writer.writerows(rows)
    print(f"Extracted {len(rows)} frames to {output}")


def check_label(path: Path) -> list[int]:
    classes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected class cx cy width height")
        try:
            class_id = int(parts[0])
            cx, cy, width, height = map(float, parts[1:])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: invalid number") from error
        if not 0 <= class_id < len(CLASSES):
            raise ValueError(f"{path}:{line_number}: invalid class {class_id}")
        if not all(math.isfinite(x) for x in (cx, cy, width, height)):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        if not (0 < width <= 1 and 0 < height <= 1 and
                0 <= cx - width / 2 and cx + width / 2 <= 1 and
                0 <= cy - height / 2 and cy + height / 2 <= 1):
            raise ValueError(f"{path}:{line_number}: box is outside image")
        classes.append(class_id)
    return classes


def package(labeled: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    files = []
    by_split = {}
    video_splits = {}
    counts = [0] * len(CLASSES)
    for split in SPLITS:
        image_dir = labeled / "images" / split
        label_dir = labeled / "labels" / split
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f"No images in {image_dir}")
        by_split[split] = len(images)
        for image in images:
            relative = image.relative_to(image_dir)
            label = (label_dir / relative).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Missing label {label}; use an empty file for a negative image")
            for class_id in check_label(label):
                counts[class_id] += 1
            # Extracted names begin with the source video stem.
            source = (image.stem.split("_frame_", 1)[0] if "_frame_" in image.stem
                      else image.stem.rsplit("_", 1)[0])
            if source in video_splits and video_splits[source] != split:
                raise ValueError(f"Frames from {source} occur in multiple splits")
            video_splits[source] = split
            files.extend((image, label))
        actual_labels = {p.relative_to(label_dir).with_suffix("") for p in label_dir.rglob("*.txt")}
        image_stems = {p.relative_to(image_dir).with_suffix("") for p in images}
        if actual_labels != image_stems:
            raise ValueError(f"Extra or mismatched labels in {label_dir}")
    yaml_text = (
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names: [bicycle, car, motorcycle, bus, truck]\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("traffic_dataset/data.yaml", yaml_text)
        for path in files:
            archive.write(path, "traffic_dataset/" + path.relative_to(labeled).as_posix())
    print(f"Created {output}; images: {by_split}; boxes: {dict(zip(CLASSES, counts))}")


def public(source: Path, output: Path, archive: Path) -> None:
    """Convert the seven-class Vietnam traffic dataset to five vehicle classes."""
    if output.exists() or archive.exists():
        raise FileExistsError("Output exists; choose new --output and --archive paths")
    expected = "names: ['accident', 'bicycle', 'bus', 'car', 'motorcycle', 'person', 'truck']"
    if expected not in (source / "data.yaml").read_text(encoding="utf-8"):
        raise ValueError("Unexpected source class order; check mapping before conversion")
    groups = {}
    group_classes = {}
    for split in ("train", "valid", "test"):
        image_dir = source / split / "images"
        for image in sorted(image_dir.glob("*.jpg")):
            if image.stat().st_size < 1000:
                raise ValueError(f"Image is still a Git LFS pointer: {image}")
            label = source / split / "labels" / (image.stem + ".txt")
            if not label.is_file():
                raise ValueError(f"Missing source label: {label}")
            group = image.stem.split("_frame_", 1)[0] if "_frame_" in image.stem else image.stem
            groups.setdefault(group, []).append((image, label))
            class_counts = group_classes.setdefault(group, [0] * len(CLASSES))
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Unexpected source label: {label}")
                original_id = int(parts[0])
                if original_id in PUBLIC_CLASS_MAP:
                    class_counts[PUBLIC_CLASS_MAP[original_id]] += 1
    if not groups:
        raise ValueError(f"No source images in {source}")
    total = sum(len(items) for items in groups.values())
    targets = {"train": 0.70 * total, "val": 0.15 * total, "test": 0.15 * total}
    class_totals = [sum(group_classes[g][c] for g in groups) for c in range(len(CLASSES))]
    best = None
    for seed in range(1000):
        names = sorted(groups)
        random.Random(seed).shuffle(names)
        split_counts = {split: 0 for split in SPLITS}
        class_split_counts = {split: [0] * len(CLASSES) for split in SPLITS}
        assignment = {}
        for group in names:
            split = max(SPLITS, key=lambda candidate: (targets[candidate] - split_counts[candidate]) / targets[candidate])
            assignment[group] = split
            split_counts[split] += len(groups[group])
            for class_id, count in enumerate(group_classes[group]):
                class_split_counts[split][class_id] += count
        if any(class_split_counts[split][c] == 0 for split in SPLITS for c in range(len(CLASSES))):
            continue
        score = sum(
            abs(class_split_counts[split][c] - class_totals[c] * targets[split] / total)
            / max(1, class_totals[c] * targets[split] / total)
            for split in SPLITS for c in range(len(CLASSES))
        )
        if best is None or score < best[0]:
            best = (score, assignment, class_split_counts)
    if best is None:
        raise ValueError("Cannot create splits with every vehicle class represented")
    assignment = best[1]
    split_counts = {split: 0 for split in SPLITS}
    for group in sorted(groups):
        split = assignment[group]
        for image, label in groups[group]:
            dest_image = output / "images" / split / image.name
            dest_label = output / "labels" / split / label.name
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            dest_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dest_image)
            lines = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                original_id = int(parts[0])
                if original_id in PUBLIC_CLASS_MAP:
                    lines.append(" ".join((str(PUBLIC_CLASS_MAP[original_id]), *parts[1:])))
            dest_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_counts[split] += 1
    package(output, archive)
    print(f"Source groups: {len(groups)}; converted split: {split_counts}; class split: {best[2]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--videos", type=Path, default=Path("dataset_videos"))
    extract_parser.add_argument("--output", type=Path, default=Path("training_data/to_label"))
    extract_parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    package_parser = sub.add_parser("package")
    package_parser.add_argument("--labeled", type=Path, default=Path("training_data/labeled"))
    package_parser.add_argument("--output", type=Path, default=Path("training_data/traffic_dataset.zip"))
    public_parser = sub.add_parser("public")
    public_parser.add_argument("--source", type=Path,
                               default=Path("training_data/public_accident_detect/dataset"))
    public_parser.add_argument("--output", type=Path, default=Path("training_data/public_labeled"))
    public_parser.add_argument("--archive", type=Path,
                               default=Path("training_data/public_traffic_dataset.zip"))
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.videos, args.output, args.interval)
    elif args.command == "package":
        package(args.labeled, args.output)
    else:
        public(args.source, args.output, args.archive)


if __name__ == "__main__":
    main()
