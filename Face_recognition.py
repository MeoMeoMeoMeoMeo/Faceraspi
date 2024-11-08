import cv2
import numpy as np
import os
import pickle

# Thử import RPi.GPIO; nếu không thành công, bỏ qua
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Không thể import RPi.GPIO. Đang chạy trên môi trường không phải Raspberry Pi.")
    GPIO_AVAILABLE = False

# Cấu hình GPIO nếu có sẵn
if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)

baseDir = os.path.dirname(os.path.abspath(__file__))

# Tải các bộ phân loại Haar cascade
faceCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_frontalface_default.xml"))
profileFaceCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_profileface.xml"))
eyeCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_eye.xml"))
smileCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_smile.xml"))

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(baseDir, "trainer.yml"))

labels = {}
with open(os.path.join(baseDir, "labels"), "rb") as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

camera = cv2.VideoCapture(0)
camera.set(3, 640)  # Chiều rộng khung hình
camera.set(4, 480)  # Chiều cao khung hình

minW = 0.1 * camera.get(3)
minH = 0.1 * camera.get(4)

while True:
    ret, im = camera.read()
    if not ret:
        print("Không thể truy cập vào camera.")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(minW), int(minH)))

    # Phát hiện mặt chính diện
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence < 60:  # Chỉ in tên nếu độ tin cậy < 60%
            name = labels.get(id_, "Không xác định")
            confidence_text = f"{round(100 - confidence)}%"
            if GPIO_AVAILABLE:
                GPIO.output(18, GPIO.HIGH)  # Kích hoạt GPIO khi nhận diện thành công
        else:
            name = "Không xác định"
            confidence_text = f"{round(100 - confidence)}%"
            if GPIO_AVAILABLE:
                GPIO.output(18, GPIO.LOW)  # Không kích hoạt GPIO khi nhận diện không thành công

        cv2.putText(im, str(name), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(im, str(confidence_text), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Phát hiện mặt nghiêng
    profile_faces = profileFaceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Khung màu xanh dương

    # Phát hiện mắt
    eyes = eyeCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(im, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)  # Khung màu vàng

    # Phát hiện cười
    smiles = smileCascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(im, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)  # Khung màu xanh lá cây

    cv2.imshow('Camera', im)
    key = cv2.waitKey(10)
    if key == 27:  # Nhấn ESC để thoát
        break

camera.release()
cv2.destroyAllWindows()

# Dọn dẹp GPIO nếu có sử dụng
if GPIO_AVAILABLE:
    GPIO.cleanup()
