import socket
import os
import sys
import json

def send_command(socket_file, command):
    # 获取当前环境变量并转为字典
    env_vars = {key: os.environ[key] for key in os.environ}
    
    # 创建发送数据的字典
    data = {
        'command': command,
        'env': env_vars
    }
    
    # 连接到 Unix socket
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_file)
        # 将数据转换为 JSON 格式并发送
        sock.sendall(json.dumps(data).encode('utf-8'))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python client.py <socket_file> <command>")
        sys.exit(1)

    socket_file = sys.argv[1]
    command = " ".join(sys.argv[2:])
    send_command(socket_file, command)