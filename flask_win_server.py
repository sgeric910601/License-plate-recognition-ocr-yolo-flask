import cv2
import socket
import struct
import pickle
import os
import datetime
import torch
import re
from flask import Flask, render_template, Response, jsonify, url_for, send_from_directory
from ultralytics import YOLO
from paddleocr import PaddleOCR
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading


# 初始化 Flask
app = Flask(__name__)

# 初始化 YOLO 和 PaddleOCR
model = YOLO('license_plate_detector2.pt')
# if torch.backends.mps.is_available():
#     device = 'mps'
# elif torch.cuda.is_available():
#     device = 'cuda'
# else:
    # device = 'cpu'
device = 'cpu'
print(f"使用的設備是: {device}")
model.to(device)

ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=device in ['cuda', 'mps'])

# 全局變量
last_frame = None
last_license_plate = ""
saved_license_plates = set()
save_dir = 'static/captured_images'
plates_file = "plates.txt"
os.makedirs(save_dir, exist_ok=True)

# 獲取當前日期和時間
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 將日期和時間寫入檔案
with open(plates_file, "a", encoding="utf-8") as file:
    file.write(f"程式執行時間：{current_time}\n")

print(f"執行時間已記錄在 {plates_file}")



def receive_video():
    """接收樹莓派的視頻流"""
    global last_frame
    server_socket = None
    conn = None
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('0.0.0.0', 5680))
        server_socket.listen(1)
        print("等待樹莓派連接...")

        conn, addr = server_socket.accept()
        print(f"樹莓派已連接: {addr}")
        data = b""
        payload_size = struct.calcsize(">L")

        while True:
            # 接收消息大小
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionResetError("樹莓派連接中斷")
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            # 接收幀數據
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                frame = pickle.loads(frame_data)
                last_frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"解碼錯誤: {e}")
    except Exception as e:
        print(f"視頻接收錯誤: {e}")
    finally:
        if conn:
            conn.close()
        if server_socket:
            server_socket.close()
        print("已釋放視頻接收資源")
        
def is_valid_license_plate(text):
    """檢查車牌號是否符合格式"""
    car_plate_pattern = r'^(?:[A-Z]{2,3}-\d{4}|\d{4}-[A-Z]{2})$'
    bike_plate_pattern = r'^(?:\d{3}-[A-Z]{3}|[A-Z]{3}-\d{3})$'
    return re.match(car_plate_pattern, text) or re.match(bike_plate_pattern, text)

# 處理檢測和識別
def detect_and_recognize(frame):
    results = model(frame)
    return results

# 視頻流生成器
def process_video_stream():
    """处理视频流"""
    global last_frame, saved_license_plates, last_license_plate
    while True:
        if last_frame is not None:
            frame = last_frame.copy()
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
                                            file_name = f"captured_frame_{len(saved_license_plates)}.jpg"
                                            save_path = os.path.join(save_dir, file_name)
                                            cv2.imwrite(save_path, frame)
                                            saved_license_plates.add(license_text)
                                            last_license_plate = license_text
                                            with open(plates_file, "a", encoding="utf-8") as file:
                                                file.write(license_text + "\n")
            except Exception as e:
                print(f"处理图像时出错: {e}")

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# def generate_frames():
#     """產生實時視頻幀"""
#     global last_frame
#     while True:
#         if last_frame is not None:
#             _, buffer = cv2.imencode('.jpg', last_frame)
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
@app.route('/')
def index():
    """主頁"""
    images = os.listdir(save_dir)
    return render_template('index-v1.html', images=images, last_license_plate=last_license_plate)

# @app.route('/video_feed')
# def video_feed():
#     """實時視頻流"""
#     def generate():
#         for frame in process_frame():  # 確保 process_frame() 產生正確的視頻幀
#             try:
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             except Exception as e:
#                 print(f"視頻幀生成錯誤: {e}")  # 日誌輸出
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/captured_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(save_dir, filename)

# @app.route('/get_dynamic_data')
# def get_dynamic_data():
#     # 獲取保存的圖片列表
#     images = os.listdir(save_dir)

#     # 返回 JSON 格式的數據
#     return jsonify({
#         "last_license_plate": last_license_plate,  # 最後檢測到的車牌號
#         "images": [f"/static/captured_images/{img}" for img in images],  # 圖片的路徑
#         "video_feed": url_for('video_feed')  # 視頻流的完整 URL
#     })
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
    """釋放資源並關閉伺服器"""
    print("正在釋放資源...")
    os._exit(0)  # 强制终止程序
    return "伺服器已停止並釋放資源。"

# if __name__ == "__main__":
#     import threading
#     threading.Thread(target=receive_video, daemon=True).start()
#     app.run(host='0.0.0.0', debug=True)
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        # 開啟接收視頻的線程
        threading.Thread(target=receive_video, daemon=True).start()
        # 啟動 Flask
        app.run(host='0.0.0.0', debug=True, port=5000)
    finally:
        print("程序即將退出，釋放所有資源...")
