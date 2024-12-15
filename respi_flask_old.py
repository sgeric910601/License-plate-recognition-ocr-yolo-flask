import cv2
import os
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR
from flask import Flask, render_template, Response, send_from_directory
import re
import datetime
import torch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from flask import jsonify

# 加载预训练的车牌检测模型
model = YOLO('license_plate_detector2.pt')
# 自动选择设备：CUDA > MPS > CPU
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"使用的设备是: {device}")
# device = 'cpu'  # 如果需要 GPU 支援，可改為 'cuda'
model.to(device)
# 初始化 PaddleOCR（英文模式，使用GPU）
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu= device in ['cuda', 'mps'])

skip_frames = 20  # 每 5 幀處理一次 OCR
current_frame = 0
executor = ThreadPoolExecutor(max_workers=2)

# 設定保存圖像的目錄
save_dir = 'static/captured_images'
os.makedirs(save_dir, exist_ok=True)

# 存儲已檢測到的車牌號集合
saved_license_plates = set()
frame_count = 0     # 用於命名保存的圖像文件

plates_file = "plates.txt"

# 獲取當前日期和時間
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 將日期和時間寫入檔案
with open(plates_file, "a", encoding="utf-8") as file:
    file.write(f"程式執行時間：{current_time}\n")

print(f"執行時間已記錄在 {plates_file}")

last_license_plate = ""     # 用于保存最新识别的车牌号
# 初始化攝影機
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法啟動攝影機，請檢查設備。")
    exit()

# 定義函數以驗證車牌格式
def is_valid_license_plate(text):
    # 汽車車牌格式
    car_plate_pattern = r'^(?:[A-Z]{2,3}-\d{4}|\d{4}-[A-Z]{2}|\d{4}-[A-Z]\d|\d{4}-\d[A-Z]|[A-Z]\d-\d{4}|\d[A-Z]-\d{4})$'
    # 機車車牌格式
    bike_plate_pattern = r'^(?:\d{3}-[A-Z]{3}|[A-Z]{3}-\d{3}|[A-Z0-9]{3}-[A-Z0-9]{3})$'

    
    # 驗證車牌是否符合任一格式
    return re.match(car_plate_pattern, text) or re.match(bike_plate_pattern, text)


def detect_and_recognize(frame):
    results = model(frame)
    # 處理 YOLO 和 OCR
    return results

def process_video_stream():
    global frame_count, saved_license_plates, last_license_plate, current_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法從攝影機讀取影像。")
            break

        try:
            # if current_frame % skip_frames == 0:  # 控制 YOLO 檢測頻率
                # 使用 YOLO 模型檢測車牌
                results = model(frame, imgsz=320)  # 降低输入分辨率加速
                detected_text = ''

                # 遍历每个检测结果
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and boxes.data is not None and len(boxes.data) > 0:
                        data = boxes.data.cpu().numpy()

                        # 遍历每个检测框
                        for row in data:
                            x1, y1, x2, y2, conf, cls = row
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            conf = float(conf)

                            if conf > 0.5:
                                # 提取检测框区域
                                h, w, _ = frame.shape
                                if 0 <= x1 < w and 0 <= x2 <= w and 0 <= y1 < h and 0 <= y2 <= h:
                                    license_plate = frame[y1:y2, x1:x2]
                                    ocr_result = ocr.ocr(license_plate, cls=True)

                                    # OCR 结果处理
                                    if ocr_result:
                                        for line in ocr_result:
                                            if line:
                                                for word_info in line:
                                                    if word_info and len(word_info) > 1:
                                                        detected_text += word_info[1][0] + ' '

                                    license_text = detected_text.strip().replace(' ', '')  # 清理文本

                                    # 验证车牌格式
                                    if is_valid_license_plate(license_text):
                                        if license_text not in saved_license_plates:
                                            # 用 OpenCV 绘制框框和文字
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(frame, license_text, (x1, y1 - 10),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                                            # 保存图像
                                            file_name = f"captured_frame_{frame_count}.jpg"
                                            save_path = os.path.join(save_dir, file_name)
                                            cv2.imwrite(save_path, frame)
                                            print(f"檢測到車牌 '{license_text}'，保存圖像: {save_path}")
                                            frame_count += 1
                                            saved_license_plates.add(license_text)

                                            # 更新最后一个识别的车牌号
                                            last_license_plate = license_text

                                            # 将车牌号保存到文件
                                            with open(plates_file, "a", encoding="utf-8") as file:
                                                file.write(license_text + "\n")
                                                lines = [line.strip() for line in file.readlines() if line.strip()]
                                            print("已儲存車牌")

                                    else:
                                        print(f"無效車牌號碼: {license_text}")

        except Exception as e:
            print(f"處理影像時發生錯誤: {e}")

        # 将视频帧转换为 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 使用生成器返回视频帧
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



app = Flask(__name__)
@app.route('/')
def index():
    global last_license_plate
    # 在 HTML 页面显示摄像头画面和保存的图片以及最后识别的车牌号
    images = os.listdir(save_dir)
    return render_template('index-v1.html', images=images, last_license_plate=last_license_plate)

@app.route('/video_feed')
def video_feed():
    # 摄像头视频流路由
    return Response(process_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/captured_images/<filename>')
def uploaded_file(filename):
    # 提供保存的图像
    return send_from_directory(save_dir, filename)



@app.route('/get_dynamic_data')
def get_dynamic_data():
    global last_license_plate
    images = os.listdir(save_dir)
    # 回傳 JSON 格式的最新車牌號以及當前已保存的圖片 URL
    # 前端可透過 url_for('uploaded_file', filename=image) 動態產生圖片路徑
    image_urls = [f"/static/captured_images/{img}" for img in images]

    return jsonify({
        "last_license_plate": last_license_plate,
        "images": image_urls,
        "video_feed": "/video_feed"  # 假設前端需要更新攝像機畫面 URL
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port = 5001)
    
