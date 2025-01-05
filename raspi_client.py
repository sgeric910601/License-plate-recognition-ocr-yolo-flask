import socket
import struct
import pickle
import subprocess
import numpy as np

def send_video():
    # 設置樹莓派上的 libcamera 命令
    libcamera_command = [
        "libcamera-vid",
        "--width", "640",
        "--height", "480",
        "--framerate", "15",
        "-t","0",
	"--codec", "mjpeg",
        "-o", "-"
    ]
    
    # 啟動 libcamera-vid 並捕獲視頻流
    process = subprocess.Popen(libcamera_command, stdout=subprocess.PIPE, bufsize=10**8)

    # 伺服器 IP 和端口
    server_address = ('192.168.2.221', 5680)  # 替換為伺服器的 IP 地址
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)

    try:
        buffer = b""
        while True:
            buffer += process.stdout.read(1024)
            start = buffer.find(b'\xff\xd8')  # JPEG 開始標記
            end = buffer.find(b'\xff\xd9')   # JPEG 結束標記
            if start != -1 and end != -1:
                frame_data = buffer[start:end+2]
                buffer = buffer[end+2:]

                # 將 JPEG 數據打包並發送
                frame = np.frombuffer(frame_data, np.uint8)
                data = pickle.dumps(frame)
                size = len(data)
                client_socket.sendall(struct.pack(">L", size) + data)
    except Exception as e:
        print(f"傳輸錯誤: {e}")
    finally:
        process.terminate()
        client_socket.close()

if __name__ == "__main__":
    send_video()
