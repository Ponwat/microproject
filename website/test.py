import redis

import base64
import numpy as np
import cv2 as cv

from flask import Flask, render_template
from flask_socketio import SocketIO

def main() -> None:
    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")
    r = redis.Redis(host="localhost", port=6379, db=0)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/capture")
    def capture() -> str:
        frame_data = r.get("latest_frame")

        decoded_data = base64.b64decode(frame_data)

        image_array = np.frombuffer(decoded_data, dtype=np.uint8)

        decoded_image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        cv.imwrite(f"./a.jpg", decoded_image)
        return frame_data.decode('utf-8')

    @socketio.on("connect")
    def handle_connect() -> None:
        def send_frames():
            while True:
                frame_data = r.get("latest_frame")  # Get the latest frame from Redis
                if frame_data:
                    socketio.emit('video_frame', {'data': frame_data.decode('utf-8')})
                socketio.sleep(0.03)

        socketio.start_background_task(send_frames)

    socketio.run(app, host='0.0.0.0', port=5000)

def test() -> None:
    r = redis.Redis()
    while True:
        print("a", r.get("latest_frame"))

if __name__ == '__main__':
    main()
    # test()
