# Gerekli paketi içe aktarıyoruz.
import cv2
import numpy as np 
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
detector = cv2.CascadeClassifier(os.path.join(path, 'Classifiers', 'face.xml'))

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

minW = 1 * camera.get(3)
minH = 1 * camera.get(4)

# Lấy đường dẫn thư mục hiện tại (dự án)
baseDir = os.path.dirname(os.path.abspath(__file__))
# Tạo đường dẫn tới tệp haarcascade
faceCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_frontalface_default.xml"))
faceCascadeProfile = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_profileface.xml"))
eyeCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_eye.xml"))
eyeCascadeWithGlasses = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_eye_tree_eyeglasses.xml"))
smileCascade = cv2.CascadeClassifier(os.path.join(baseDir, "haarcascade_smile.xml"))

# Lấy tên từ người dùng trước khi sử dụng
name = input("name")

# Tạo đường dẫn tới thư mục images trong cùng thư mục dự án
dirName = os.path.join(baseDir, "images", name)

# Kiểm tra nếu thư mục chưa tồn tại, tạo thư mục
if not os.path.exists(dirName):
    os.makedirs(dirName)

print(f"Thư mục sẽ được lưu tại: {os.path.abspath(dirName)}")

count = len(os.listdir(dirName)) + 1  # Bắt đầu đếm từ số ảnh hiện có trong thư mục
frame_count = 0  # Thêm bộ đếm khung hình

while True:
    ret, im = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Phát hiện khuôn mặt chính diện
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Phát hiện nụ cười bên trong khuôn mặt
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smileCascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(im, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)

        # Phát hiện mắt bên trong khuôn mặt
        eyes = eyeCascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(im, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 255), 2)

    # Phát hiện khuôn mặt nghiêng
    profile_faces = faceCascadeProfile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Phát hiện mắt đeo kính
    eyes_with_glasses = eyeCascadeWithGlasses.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes_with_glasses:
        cv2.rectangle(im, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)

    for (x, y, w, h) in faces:
        frame_count += 1  # Tăng bộ đếm khung hình mỗi lần có khung hình mới

        if frame_count % 20 == 0:  # Chỉ lưu ảnh khi bộ đếm là bội số của 20
            roiGray = gray[y:y+h, x:x+w]
            fileName = os.path.join(dirName, f"{name}{count}.jpg")
            cv2.imwrite(fileName, roiGray)
            cv2.imshow("face", roiGray)
            count += 1

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('im', im)
    key = cv2.waitKey(10)
    if key == 27:  # Nhấn ESC để thoát
        break

camera.release()
cv2.destroyAllWindows()
