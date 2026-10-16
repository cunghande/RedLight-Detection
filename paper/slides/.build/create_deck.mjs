import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "D:\\Code\\nhandienphuongtien";
const buildDir = path.join(workspaceDir, "paper", "slides", ".build");
const previewDir = path.join(buildDir, "previews");
const outputDir = path.join(workspaceDir, "output", "pptx");
const draftPath = path.join(buildDir, "red_light_detection_draft.pptx");
const finalPath = path.join(outputDir, "Nguyen_Tuan_Anh_Phat_hien_phuong_tien_vuot_den_do_10_slides.pptx");
const skillDir = "C:\\Users\\Admin\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.1013.11914\\skills\\presentations";
const pythonExecutable = "C:\\Users\\Admin\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const { makeNativeBulletParagraphs, applyPresentationChartFont } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href
);

const W = 1280;
const H = 720;
const FONT = "Arial";
const C = {
  navy: "#172A3A",
  blue: "#236B8E",
  blue2: "#5E9DBA",
  ink: "#17212B",
  gray: "#5F6B76",
  light: "#EEF2F4",
  line: "#CFD8DD",
  white: "#FFFFFF",
  red: "#B23A35",
  amber: "#B67A22",
  green: "#3B7D61",
};

const img = {
  cover: path.join(workspaceDir, "violations", "run_20261016_074005_587285", "EV-20261016-074030-552313-0025_scene.jpg"),
  problem: path.join(workspaceDir, ".debug_red1.jpg"),
  vehicle: path.join(workspaceDir, "violations", "run_20261016_074005_587285", "EV-20261016-074023-502504-0015_vehicle.jpg"),
  signal: path.join(workspaceDir, ".debug_red2.jpg"),
  sequence: path.join(workspaceDir, ".test-tmp", "accuracy_audit", "red3_tracks.jpg"),
  evidence1: path.join(workspaceDir, "violations", "VIOLATION_20261013_152403_ID76_scene.jpg"),
  evidence2: path.join(workspaceDir, "violations", "VIOLATION_20261013_152428_ID61_scene.jpg"),
  closing: path.join(workspaceDir, ".debug_red3.jpg"),
};

const imageData = {};
for (const imagePath of Object.values(img)) {
  const bytes = await fs.readFile(imagePath);
  imageData[imagePath] = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });
presentation.theme.colorScheme = {
  name: "Minimal Traffic",
  themeColors: {
    accent1: C.blue,
    accent2: C.blue2,
    accent3: C.green,
    accent4: C.amber,
    accent5: C.red,
    accent6: C.gray,
    bg1: C.white,
    bg2: C.light,
    tx1: C.ink,
    tx2: C.gray,
    dk1: "#000000",
    dk2: C.navy,
    lt1: C.white,
    lt2: C.light,
    hlink: C.blue,
    folHlink: C.navy,
  },
};

function addText(slide, text, pos, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: pos,
    fill: opts.fill ?? "none",
    line: opts.line ?? { style: "solid", fill: "none", width: 0 },
    borderRadius: opts.borderRadius,
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    alignment: opts.align ?? "left",
  };
  shape.text.insets = opts.insets ?? 0;
  shape.text.verticalAlignment = opts.valign ?? "top";
  shape.text.autoFit = opts.autoFit ?? "shrinkText";
  return shape;
}

function addRule(slide, left, top, width, color = C.blue, height = 4) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: color,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addImage(slide, imagePath, pos, opts = {}) {
  return slide.images.add({
    blob: imageData[imagePath],
    contentType: imagePath.toLowerCase().endsWith(".png") ? "image/png" : "image/jpeg",
    alt: opts.alt ?? "Ảnh minh họa từ kết quả của hệ thống",
    fit: opts.fit ?? "cover",
    position: pos,
    geometry: opts.geometry ?? "rect",
    borderRadius: opts.borderRadius,
    crop: opts.crop,
  });
}

function addSlideTitle(slide, title, number, subtitle = "") {
  addText(slide, title, { left: 72, top: 42, width: 1010, height: 50 }, { fontSize: 34, bold: true, color: C.navy });
  if (subtitle) addText(slide, subtitle, { left: 74, top: 95, width: 1000, height: 28 }, { fontSize: 17, color: C.gray });
  addRule(slide, 72, 124, 1136, C.line, 2);
  addText(slide, String(number).padStart(2, "0"), { left: 1150, top: 44, width: 58, height: 32 }, { fontSize: 17, bold: true, color: C.gray, align: "right" });
}

function addFooter(slide) {
  addText(slide, "Nguyễn Tuấn Anh  ·  Lớp 124231", { left: 72, top: 680, width: 450, height: 20 }, { fontSize: 12, color: "#7D8790" });
}

function addBulletList(slide, items, pos, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: 18,
    hangingPoints: 9,
    spaceAfterPoints: 7,
  });
  shape.text.style = { typeface: FONT, fontSize: opts.fontSize ?? 22, color: opts.color ?? C.ink };
  shape.text.insets = 0;
  shape.text.autoFit = "shrinkText";
  return shape;
}

function addProcessBox(slide, text, pos, accent = C.blue) {
  const box = slide.shapes.add({
    geometry: "roundRect",
    position: pos,
    fill: C.white,
    line: { style: "solid", fill: C.line, width: 1.2 },
    borderRadius: 10,
  });
  addRule(slide, pos.left, pos.top, 7, accent, pos.height);
  addText(slide, text, { left: pos.left + 18, top: pos.top + 11, width: pos.width - 30, height: pos.height - 20 }, { fontSize: 19, bold: true, color: C.ink, valign: "middle" });
  return box;
}

// 1. Cover
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addImage(slide, img.cover, { left: 760, top: 0, width: 520, height: 720 }, { alt: "Ảnh bằng chứng một phương tiện đi qua nút giao khi đèn đỏ" });
  slide.shapes.add({ geometry: "rect", position: { left: 720, top: 0, width: 80, height: 720 }, fill: C.white, line: { style: "solid", fill: "none", width: 0 } });
  addRule(slide, 76, 96, 86, C.blue, 6);
  addText(slide, "PHÁT HIỆN PHƯƠNG TIỆN\nVƯỢT ĐÈN ĐỎ", { left: 76, top: 144, width: 620, height: 190 }, { fontSize: 45, bold: true, color: C.navy });
  addText(slide, "Hệ thống thị giác máy tính cho camera giao thông cố định", { left: 78, top: 355, width: 570, height: 64 }, { fontSize: 22, color: C.gray });
  addText(slide, "Nguyễn Tuấn Anh\nLớp 124231", { left: 78, top: 552, width: 380, height: 80 }, { fontSize: 22, bold: true, color: C.ink });
}

// 2. Problem and objective
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Bài toán và mục tiêu", 2);
  addText(slide, "Phát hiện một xe chưa đủ để kết luận vi phạm", { left: 72, top: 164, width: 535, height: 80 }, { fontSize: 32, bold: true, color: C.navy });
  addText(slide, "Hệ thống phải liên kết đúng phương tiện, trạng thái đèn, vạch dừng và thời điểm cắt vạch.", { left: 72, top: 258, width: 520, height: 94 }, { fontSize: 23, color: C.gray });
  addText(slide, "ĐẦU VÀO", { left: 72, top: 410, width: 160, height: 24 }, { fontSize: 15, bold: true, color: C.blue });
  addText(slide, "Video từ camera cố định", { left: 72, top: 440, width: 430, height: 34 }, { fontSize: 23, bold: true });
  addText(slide, "ĐẦU RA", { left: 72, top: 515, width: 160, height: 24 }, { fontSize: 15, bold: true, color: C.blue });
  addText(slide, "Sự kiện nghi vấn kèm ảnh bằng chứng và dữ liệu truy vết", { left: 72, top: 545, width: 500, height: 70 }, { fontSize: 23, bold: true });
  addImage(slide, img.problem, { left: 650, top: 160, width: 558, height: 466 }, { alt: "Khung hình nút giao dùng để hiệu chỉnh vạch dừng và vùng đèn" });
  addFooter(slide);
}

// 3. Architecture
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Kiến trúc hệ thống", 3, "Hai nhánh nhận diện hợp nhất tại bộ luật sự kiện");
  const video = addProcessBox(slide, "Video + cấu hình camera", { left: 72, top: 180, width: 230, height: 76 }, C.navy);
  const detect = addProcessBox(slide, "Nhận diện xe\nYOLO", { left: 370, top: 155, width: 210, height: 86 }, C.blue);
  const light = addProcessBox(slide, "Nhận diện đèn\nROI + MobileNetV2", { left: 370, top: 282, width: 210, height: 86 }, C.amber);
  const track = addProcessBox(slide, "Theo dõi ID\nByteTrack", { left: 650, top: 155, width: 210, height: 86 }, C.green);
  const state = addProcessBox(slide, "Ổn định trạng thái\n3 khung hình", { left: 650, top: 282, width: 210, height: 86 }, C.amber);
  const rule = addProcessBox(slide, "Cắt vạch có hướng\nvà xác nhận sự kiện", { left: 930, top: 215, width: 250, height: 98 }, C.red);
  slide.shapes.connect(video, detect, { kind: "elbow", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(video, light, { kind: "elbow", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(detect, track, { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(light, state, { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(track, rule, { kind: "elbow", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(state, rule, { kind: "elbow", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  addText(slide, "Kết quả", { left: 72, top: 455, width: 180, height: 34 }, { fontSize: 24, bold: true, color: C.navy });
  addText(slide, "Video đã gắn nhãn  ·  JSON/CSV  ·  ảnh toàn cảnh  ·  ảnh cắt phương tiện", { left: 72, top: 500, width: 1040, height: 44 }, { fontSize: 24, color: C.ink });
  addText(slide, "Mỗi lượt chạy lưu cấu hình, hash checkpoint và trạng thái hoàn tất để có thể kiểm tra lại.", { left: 72, top: 566, width: 1040, height: 54 }, { fontSize: 20, color: C.gray });
  addFooter(slide);
}

// 4. Models
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Các mô hình đang sử dụng", 4);
  addText(slide, "Phương tiện", { left: 72, top: 162, width: 240, height: 38 }, { fontSize: 27, bold: true, color: C.navy });
  addText(slide, "YOLO11x pretrained COCO", { left: 72, top: 218, width: 410, height: 38 }, { fontSize: 25, bold: true });
  addText(slide, "Phát hiện bicycle, car, motorcycle, bus và truck. Ảnh đầu vào 960, confidence 0,10.", { left: 72, top: 266, width: 480, height: 86 }, { fontSize: 20, color: C.gray });
  addText(slide, "ByteTrack", { left: 72, top: 382, width: 240, height: 34 }, { fontSize: 25, bold: true });
  addText(slide, "Duy trì ID khi xe bị che khuất ngắn và giữ các hộp có độ tin cậy thấp cho bước liên kết.", { left: 72, top: 427, width: 480, height: 78 }, { fontSize: 20, color: C.gray });
  addImage(slide, img.vehicle, { left: 395, top: 540, width: 160, height: 90 }, { alt: "Ảnh cắt phương tiện do hệ thống ghi nhận", fit: "contain" });
  addRule(slide, 620, 160, 2, C.line, 470);
  addText(slide, "Đèn giao thông", { left: 680, top: 162, width: 300, height: 38 }, { fontSize: 27, bold: true, color: C.navy });
  addText(slide, "YOLOv8s + ROI cố định", { left: 680, top: 218, width: 430, height: 38 }, { fontSize: 25, bold: true });
  addText(slide, "YOLO đề xuất vùng đèn. Cấu hình camera xác nhận vùng đèn điều khiển đúng làn.", { left: 680, top: 266, width: 470, height: 76 }, { fontSize: 20, color: C.gray });
  addText(slide, "MobileNetV2 ONNX", { left: 680, top: 370, width: 340, height: 34 }, { fontSize: 25, bold: true });
  addText(slide, "Phân loại trạng thái từ crop 224 × 224. Ngưỡng confidence 0,80; đầu ra mơ hồ chuyển thành UNKNOWN.", { left: 680, top: 415, width: 470, height: 88 }, { fontSize: 20, color: C.gray });
  addImage(slide, img.signal, { left: 1015, top: 523, width: 165, height: 110 }, { alt: "Khung hình camera có cụm đèn giao thông", fit: "cover" });
  addFooter(slide);
}

// 5. Dataset
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Bộ dữ liệu và thiết lập thực nghiệm", 5, "Tập con COCO, cùng phép chia cho ba detector");
  const datasetChart = slide.charts.add("bar", {
    position: { left: 72, top: 170, width: 700, height: 410 },
    categories: ["Train", "Validation", "Test"],
    series: [{ name: "Số ảnh", values: [1000, 200, 200], fill: C.blue }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    xAxis: { textStyle: { fill: C.gray, fontSize: 17 }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { min: 0, max: 1100, majorUnit: 200, majorGridlines: { style: "solid", fill: C.light, width: 1 }, textStyle: { fill: C.gray, fontSize: 15 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: C.ink, fontSize: 16, bold: true } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: "none", width: 0 },
  });
  applyPresentationChartFont(datasetChart, { fontFamily: FONT });
  addText(slide, "1.400", { left: 850, top: 175, width: 280, height: 76 }, { fontSize: 48, bold: true, color: C.navy });
  addText(slide, "ảnh tổng cộng", { left: 850, top: 250, width: 280, height: 34 }, { fontSize: 21, color: C.gray });
  addText(slide, "3.539 / 685 / 762", { left: 850, top: 330, width: 330, height: 54 }, { fontSize: 31, bold: true, color: C.blue });
  addText(slide, "đối tượng trong train / validation / test", { left: 850, top: 390, width: 330, height: 58 }, { fontSize: 19, color: C.gray });
  addText(slide, "5 lớp", { left: 850, top: 505, width: 150, height: 38 }, { fontSize: 26, bold: true, color: C.navy });
  addText(slide, "bicycle, car, motorcycle, bus, truck", { left: 850, top: 550, width: 330, height: 64 }, { fontSize: 19, color: C.gray });
  addFooter(slide);
}

// 6. Processing steps
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Các bước xử lý", 6, "Luồng xử lý một video từ hiệu chỉnh đến xuất bằng chứng");
  const steps = [
    ["01", "Nạp video và\ncấu hình camera"],
    ["02", "Phát hiện xe,\ncập nhật ID"],
    ["03", "Đọc trạng thái\nđèn theo làn"],
    ["04", "Tính điểm neo và\nquỹ đạo có hướng"],
    ["05", "Kiểm tra cắt vạch\nvà xác nhận 3 frame"],
    ["06", "Xuất video, sự kiện\nvà ảnh bằng chứng"],
  ];
  const boxes = [];
  for (let i = 0; i < steps.length; i++) {
    const row = Math.floor(i / 3);
    const col = i % 3;
    const left = 72 + col * 380;
    const top = 170 + row * 220;
    const box = slide.shapes.add({ geometry: "roundRect", position: { left, top, width: 330, height: 150 }, fill: C.white, line: { style: "solid", fill: C.line, width: 1.4 }, borderRadius: 12 });
    addText(slide, steps[i][0], { left: left + 22, top: top + 22, width: 70, height: 42 }, { fontSize: 28, bold: true, color: C.blue });
    addText(slide, steps[i][1], { left: left + 22, top: top + 76, width: 280, height: 62 }, { fontSize: 22, bold: true, color: C.ink });
    boxes.push(box);
  }
  slide.shapes.connect(boxes[0], boxes[1], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(boxes[1], boxes[2], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(boxes[2], boxes[3], { kind: "elbow", fromSide: "bottom", toSide: "top", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(boxes[3], boxes[4], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  slide.shapes.connect(boxes[4], boxes[5], { kind: "straight", fromSide: "right", toSide: "left", line: { style: "solid", fill: C.gray, width: 2 }, tail: { type: "triangle", width: "sm", length: "sm" } });
  addFooter(slide);
}

// 7. Violation logic
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Logic xác nhận vi phạm", 7);
  addImage(slide, img.sequence, { left: 72, top: 155, width: 635, height: 500 }, { alt: "Chuỗi sáu khung hình theo dõi xe cắt vạch khi đèn đỏ", fit: "contain" });
  addText(slide, "Sự kiện chỉ được ghi khi đủ bằng chứng", { left: 760, top: 172, width: 430, height: 72 }, { fontSize: 29, bold: true, color: C.navy });
  addBulletList(slide, [
    "ID xe đã xuất hiện ở phía trước vạch.",
    "Đoạn quỹ đạo cắt đúng đoạn vạch theo hướng giám sát.",
    "Đèn ở trạng thái RED tại thời điểm cắt.",
    "Xe tiếp tục ở phía sau vạch trong 3 frame.",
    "Xe quay lại trước khi đủ xác nhận thì ứng viên bị hủy.",
  ], { left: 760, top: 270, width: 430, height: 280 }, { fontSize: 20 });
  addText(slide, "Frame cắt và frame xác nhận được lưu riêng để giữ đúng mốc thời gian.", { left: 760, top: 580, width: 430, height: 64 }, { fontSize: 19, bold: true, color: C.blue });
  addFooter(slide);
}

// 8. Evaluation results
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Kết quả đánh giá detector", 8, "200 ảnh test, confidence 0,25 và IoU 0,50");
  const accuracyChart = slide.charts.add("bar", {
    position: { left: 60, top: 160, width: 790, height: 440 },
    categories: ["SSD300-VGG16", "YOLOv8s", "YOLO11x"],
    series: [
      { name: "mAP@0.50:0.95", values: [0.257, 0.318, 0.309], fill: C.blue },
      { name: "F1", values: [0.445, 0.529, 0.514], fill: C.amber },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 50 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fill: C.gray, fontSize: 15 } },
    xAxis: { textStyle: { fill: C.gray, fontSize: 15 }, line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { min: 0, max: 0.6, majorUnit: 0.1, numberFormatCode: "0.0", majorGridlines: { style: "solid", fill: C.light, width: 1 }, textStyle: { fill: C.gray, fontSize: 14 } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.000", textStyle: { fill: C.ink, fontSize: 14, bold: true } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: "none", width: 0 },
  });
  applyPresentationChartFont(accuracyChart, { fontFamily: FONT });
  addText(slide, "YOLOv8s", { left: 900, top: 185, width: 260, height: 52 }, { fontSize: 36, bold: true, color: C.blue });
  addText(slide, "mAP cao nhất", { left: 900, top: 242, width: 260, height: 30 }, { fontSize: 18, color: C.gray });
  addText(slide, "0,318", { left: 900, top: 286, width: 240, height: 72 }, { fontSize: 48, bold: true, color: C.navy });
  addText(slide, "F1 = 0,529", { left: 900, top: 370, width: 250, height: 38 }, { fontSize: 26, bold: true });
  addText(slide, "119,9 FPS", { left: 900, top: 458, width: 250, height: 40 }, { fontSize: 28, bold: true, color: C.green });
  addText(slide, "Đo batch 1, FP32 trên NVIDIA H100. FPS chỉ tính detector.", { left: 900, top: 510, width: 275, height: 76 }, { fontSize: 17, color: C.gray });
  addFooter(slide);
}

// 9. Speed and evidence
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addSlideTitle(slide, "Tốc độ và bằng chứng sự kiện", 9);
  const speedChart = slide.charts.add("bar", {
    position: { left: 55, top: 170, width: 560, height: 360 },
    categories: ["SSD300-VGG16", "YOLOv8s", "YOLO11x"],
    series: [{ name: "FPS", values: [166.8, 119.9, 65.2], fill: C.green }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 45 },
    hasLegend: false,
    xAxis: { min: 0, max: 180, majorUnit: 30, majorGridlines: { style: "solid", fill: C.light, width: 1 }, textStyle: { fill: C.gray, fontSize: 14 } },
    yAxis: { textStyle: { fill: C.gray, fontSize: 15 }, line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0", textStyle: { fill: C.ink, fontSize: 15, bold: true } },
    chartFill: C.white,
    chartLine: { style: "solid", fill: "none", width: 0 },
  });
  applyPresentationChartFont(speedChart, { fontFamily: FONT });
  addText(slide, "SSD nhanh nhất; YOLOv8s có độ chính xác cao hơn và chỉ 11,1 triệu tham số.", { left: 72, top: 560, width: 530, height: 70 }, { fontSize: 20, bold: true, color: C.navy });
  addImage(slide, img.evidence1, { left: 680, top: 165, width: 245, height: 330 }, { alt: "Ảnh bằng chứng sự kiện thứ nhất", fit: "cover" });
  addImage(slide, img.evidence2, { left: 955, top: 165, width: 245, height: 330 }, { alt: "Ảnh bằng chứng sự kiện thứ hai", fit: "cover" });
  addText(slide, "Ảnh toàn cảnh giữ đồng thời phương tiện, vạch và trạng thái nút giao để người vận hành kiểm tra lại.", { left: 680, top: 530, width: 520, height: 78 }, { fontSize: 20, color: C.gray });
  addFooter(slide);
}

// 10. Conclusion
{
  const slide = presentation.slides.add({ background: { fill: C.white } });
  addImage(slide, img.closing, { left: 760, top: 0, width: 520, height: 720 }, { alt: "Khung hình camera cố định tại nút giao", fit: "cover" });
  slide.shapes.add({ geometry: "rect", position: { left: 720, top: 0, width: 80, height: 720 }, fill: C.white, line: { style: "solid", fill: "none", width: 0 } });
  addRule(slide, 72, 72, 84, C.blue, 6);
  addText(slide, "Kết luận", { left: 72, top: 110, width: 500, height: 64 }, { fontSize: 42, bold: true, color: C.navy });
  addText(slide, "YOLOv8s đạt cân bằng tốt nhất trên tập thử", { left: 72, top: 210, width: 570, height: 72 }, { fontSize: 29, bold: true, color: C.ink });
  addText(slide, "mAP@0.50:0.95 = 0,318 · F1 = 0,529 · 119,9 FPS", { left: 72, top: 292, width: 570, height: 48 }, { fontSize: 21, color: C.blue });
  addRule(slide, 72, 378, 540, C.line, 2);
  addText(slide, "Giới hạn cần xử lý", { left: 72, top: 410, width: 360, height: 38 }, { fontSize: 25, bold: true, color: C.navy });
  addText(slide, "Chưa có ground truth sự kiện độc lập. Dữ liệu COCO chưa đại diện đầy đủ cho camera và giao thông Việt Nam.", { left: 72, top: 460, width: 555, height: 90 }, { fontSize: 21, color: C.gray });
  addText(slide, "Bước tiếp theo", { left: 72, top: 580, width: 250, height: 34 }, { fontSize: 23, bold: true, color: C.navy });
  addText(slide, "Gán nhãn theo camera, đánh giá tracking, trạng thái đèn và F1 ở mức sự kiện.", { left: 72, top: 620, width: 555, height: 58 }, { fontSize: 20, color: C.gray });
}

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await layout.text());
}

const montage = await presentation.export({ format: "webp", montage: { columns: 2, slideWidth: 520, gap: 14, padding: 14, background: "#DDE3E7" } });
await fs.writeFile(path.join(previewDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const draft = await PresentationFile.exportPptx(presentation);
await draft.save(draftPath);

if (process.env.PRESENTATION_FINALIZE === "1") {
  const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);
  const stagingDir = path.join(buildDir, ".codex-finalizer");
  await fs.mkdir(stagingDir, { recursive: true });
  const candidatePath = path.join(stagingDir, "candidate.pptx");
  const candidate = await PresentationFile.exportPptx(presentation);
  await candidate.save(candidatePath);
  const result = await finalizePresentation({
    explicitTotalSlideCount: 10,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [5, 8, 9],
    requiredEmbeddedWorkbookChartOwnerSlides: [],
    materializeLiteralChartWorkbooks: true,
    workspaceDir,
    candidatePath,
    finalPath,
    pythonExecutable,
    integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", "12192000,6858000",
      "--validate-bullet-geometry",
      "--validate-heading-fit",
    ],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(stagingDir, "validation.json"),
  });
  console.log(JSON.stringify(result, null, 2));
}

console.log(`Draft: ${draftPath}`);
console.log(`Slides: ${presentation.slides.items.length}`);
