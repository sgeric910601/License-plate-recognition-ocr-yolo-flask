import cv2
import os
import re
import datetime
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR
from flask import Flask, render_template, Response, send_from_directory, jsonify
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# 避免多進程錯誤
multiprocessing.set_start_method('fork', force=True)

# Flask 應用初始化
app = Flask(__name__)

# 加載 YOLO 模型並設置設備
model = YOLO('license_plate_detector2.pt')
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"使用的設備是: {device}")
model.to(device)

# 初始化 PaddleOCR
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=device in ['cuda', 'mps'])

# 設置保存圖像的目錄
save_dir = 'static/captured_images'
os.makedirs(save_dir, exist_ok=True)

# 全局變量
skip_frames = 10
saved_license_plates = set()
frame_count = 0
plates_file = "plates.txt"
last_license_plate = ""
executor = ThreadPoolExecutor(max_workers=2)

# 在程序啟動時初始化攝像頭
class Camera:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("無法啟動攝像頭，請檢查設備。")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("無法從攝像頭讀取影像。")
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

camera = Camera()

# 檢查車牌號是否有效
def is_valid_license_plate(text):
    car_plate_pattern = r'^(?:[A-Z]{2,3}-\d{4}|\d{4}-[A-Z]{2}|\d{4}-[A-Z]\d|\d{4}-\d[A-Z]|[A-Z]\d-\d{4}|\d[A-Z]-\d{4})$'
    bike_plate_pattern = r'^(?:\d{3}-[A-Z]{3}|[A-Z]{3}-\d{3}|[A-Z0-9]{3}-[A-Z0-9]{3})$'
    return re.match(car_plate_pattern, text) or re.match(bike_plate_pattern, text)

# 處理檢測和識別
def detect_and_recognize(frame):
    results = model(frame)
    return results

# 視頻流生成器
def process_video_stream():
    global frame_count, saved_license_plates, last_license_plate
    while True:
        frame = camera.get_frame()
        try:
            results = detect_and_recognize(frame)
            detected_text = ""

            for result in results:
                boxes = result.boxes
                if boxes is not None and boxes.data is not None and len(boxes.data) > 0:
                    data = boxes.data.cpu().numpy()
                    for row in data:
                        x1, y1, x2, y2, conf, cls = row
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        conf = float(conf)

                        if conf > 0.5:
                            h, w, _ = frame.shape
                            if 0 <= x1 < w and 0 <= x2 <= w and 0 <= y1 < h and 0 <= y2 <= h:
                                license_plate = frame[y1:y2, x1:x2]
                                ocr_result = ocr.ocr(license_plate, cls=True)

                                if ocr_result:
                                    for line in ocr_result:
                                        if line:
                                            for word_info in line:
                                                if word_info and len(word_info) > 1:
                                                    detected_text += word_info[1][0] + ' '

                                license_text = detected_text.strip().replace(' ', '')

                                if is_valid_license_plate(license_text):
                                    if license_text not in saved_license_plates:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, license_text, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        file_name = f"captured_frame_{frame_count}.jpg"
                                        save_path = os.path.join(save_dir, file_name)
                                        cv2.imwrite(save_path, frame)
                                        frame_count += 1
                                        saved_license_plates.add(license_text)
                                        last_license_plate = license_text
                                        with open(plates_file, "a", encoding="utf-8") as file:
                                            file.write(license_text + "\n")
        except Exception as e:
            print(f"處理影像時發生錯誤: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    images = os.listdir(save_dir)
    return render_template('index-v1.html', images=images, last_license_plate=last_license_plate)

@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/captured_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(save_dir, filename)

@app.route('/get_dynamic_data')
def get_dynamic_data():
    images = os.listdir(save_dir)
    image_urls = [f"/static/captured_images/{img}" for img in images]
    return jsonify({
        "last_license_plate": last_license_plate,
        "images": image_urls,
        "video_feed": "/video_feed"
    })

@app.route('/shutdown')
def shutdown():
    camera.release()
    return "攝像頭已釋放"

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', debug=True, port=5001)
    finally:
        camera.release()
