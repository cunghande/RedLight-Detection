import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("ReportGenerator")


class ReportGenerator:
    """
    Module Tạo Báo Cáo Phạt Nguội (JSON, CSV, Markdown).
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_reports(self, violations: List[Dict]):
        if not violations:
            logger.info("Không có vi phạm nào được ghi nhận để xuất báo cáo.")
            return

        json_path = self.output_dir / "traffic_citations.json"
        csv_path = self.output_dir / "traffic_citations.csv"
        md_path = self.output_dir / "traffic_citations_report.md"

        # 1. Xuất JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(violations, f, ensure_ascii=False, indent=2)

        # 2. Xuất CSV
        fields = [
            "citation_code", "track_id", "vehicle_type", "confidence",
            "frame_index", "timestamp", "stop_line_violated",
            "fine_range", "license_suspension", "legal_basis"
        ]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for v in violations:
                writer.writerow(v)

        # 3. Xuất Markdown
        md_content = [
            "# BÁO CÁO KẾT QUẢ XỬ PHẠT NGUỘI PHƯƠNG TIỆN VƯỢT ĐÈN ĐỎ",
            f"\n**Tổng số vi phạm ghi nhận:** {len(violations)}\n",
            "| Mã Biên Bản | ID Xe | Loại Xe | Vị Trí Vạch Dừng | Thời Gian | Mức Phạt Dự Kiến | Tước GPLX |",
            "| :--- | :---: | :--- | :--- | :--- | :--- | :--- |"
        ]
        for v in violations:
            md_content.append(
                f"| `{v['citation_code']}` | **{v['track_id']}** | {v['vehicle_type']} | {v['stop_line_violated']} | {v['timestamp']} | **{v['fine_range']}** | {v['license_suspension']} |"
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

        logger.info(f"Đã xuất báo cáo phạt nguội ra: JSON, CSV, Markdown tại '{self.output_dir}'")
