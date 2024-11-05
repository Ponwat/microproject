import time
import datetime
import base64

from numpy._core.defchararray import decode
import redis

import cv2 as cv
from cv2.typing import MatLike

import numpy as np

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_draw

IMAGE_SCALER = 4

def finger_angle(landmark, index1: int, index2: int, index3: int) -> float:
    point1 = point_to_numpy(landmark[index1])
    point2 = point_to_numpy(landmark[index2])
    point3 = point_to_numpy(landmark[index3])
    vector1 = point2 - point1
    vector2 = point2 - point3

    angle = angle_3d(vector1, vector2) 
    return angle

def point_to_numpy(point) -> np.ndarray:
    return np.array([point.x, point.y, point.z])

def angle_3d(vector1: np.ndarray, vector2: np.ndarray) -> float:
    dot_product = np.dot(vector1, vector2)
    magnitude_vector = np.linalg.norm(vector1)
    magnitude_reference = np.linalg.norm(vector2)

    angle = np.arccos(dot_product / (magnitude_vector * magnitude_reference))
    return angle

def capture(image: MatLike) -> None:
    current_date_time = datetime.datetime.fromtimestamp(time.time()).strftime("%y%m%d-%H%M%S")
    cv.imwrite(
        f"../images/{current_date_time}.jpg",
        image
    )

def main() -> None:
    today = datetime.datetime.fromtimestamp(time.time())
    print(today.strftime("%y-%m-%d %H:%M:%S"))

    cap = cv.VideoCapture(0)
    last_time: float = time.time()
    cap.set(cv.CAP_PROP_FRAME_WIDTH, int(1280))
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(720))
    print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    last_capture_time = -1

    r = redis.Redis(host='localhost', port=6379, db=0)

    while True:
        ret, image = cap.read()

        # scaled_image = cv.resize(image, (int(1280 / IMAGE_SCALER), int(720 / IMAGE_SCALER)))
        # results = hands.process(scaled_image)

        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        copied_image = image.copy()

        landmarks = results.multi_hand_landmarks
        if not landmarks:
            landmarks = []
        for i, landmark in enumerate(landmarks):
            mp_draw.draw_landmarks(image, landmark)
            this_landmark = landmark.landmark

            thumb_angle = finger_angle(this_landmark, 4, 2, 0)
            # print("thumb", np.degrees(thumb_angle))
            is_thumb_open = np.degrees(thumb_angle) > 140

            index_angle = finger_angle(this_landmark, 8, 5, 0)
            # print("index", np.degrees(index_angle) > 90)
            is_index_open = np.degrees(index_angle) > 100 

            middle_angle = finger_angle(this_landmark, 12, 9, 0)
            # print("middle", np.degrees(middle_angle) > 90)
            is_middle_open = np.degrees(middle_angle) > 100

            ring_angle = finger_angle(this_landmark, 16, 13, 0)
            # print("ring", np.degrees(ring_angle) > 90)
            is_ring_open = np.degrees(ring_angle) > 100

            pinky_angle = finger_angle(this_landmark, 20, 17, 0)
            # print("pinky", np.degrees(pinky_angle) > 90)
            is_pinky_open = np.degrees(pinky_angle ) > 100

            is_hand_open = is_thumb_open and is_index_open and is_middle_open and is_ring_open and is_pinky_open

            print(f"{i} hand", is_hand_open)
            if time.time() - last_capture_time < 1:
                continue
            if is_hand_open:
                capture(copied_image)
                last_capture_time = time.time()
                print("captured")


        this_time = time.time()
        dt = this_time - last_time
        # print(1 / dt)
        last_time = this_time

        key = cv.pollKey()
        if key == 27:
            break
        if key == 113:
            capture(image)

        if not ret:
            continue

        cv.imshow("Copied Image", copied_image)
        cv.imshow("Image", image)
        ret, buffer = cv.imencode('.jpg', copied_image)
        frame_data = base64.b64encode(buffer).decode('utf-8')
        r.set("latest_frame", frame_data)

        frame_data = r.get("latest_frame")
        decoded_data = base64.b64decode(frame_data)

        image_array = np.frombuffer(decoded_data, dtype=np.uint8)

        decoded_image = cv.imdecode(image_array, cv.IMREAD_COLOR)
        cv.imshow("buffer", decoded_image)

        time.sleep(0.03)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
