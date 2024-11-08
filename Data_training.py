import os
import numpy as np 
from PIL import Image 
import cv2
import pickle

# Lấy đường dẫn thư mục hiện tại của tệp mã
baseDir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến các tệp Haar cascade
faceCascadePath = os.path.join(baseDir, "haarcascade_frontalface_default.xml")
profileFaceCascadePath = os.path.join(baseDir, "haarcascade_profileface.xml")
eyeCascadePath = os.path.join(baseDir, "haarcascade_eye.xml")
smileCascadePath = os.path.join(baseDir, "haarcascade_smile.xml")

faceCascade = cv2.CascadeClassifier(faceCascadePath)
profileFaceCascade = cv2.CascadeClassifier(profileFaceCascadePath)
eyeCascade = cv2.CascadeClassifier(eyeCascadePath)
smileCascade = cv2.CascadeClassifier(smileCascadePath)

# Kiểm tra nếu Haar cascades được tải thành công
if faceCascade.empty() or profileFaceCascade.empty() or eyeCascade.empty() or smileCascade.empty():
    print("Lỗi khi tải các tệp Haar cascade.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

# Đường dẫn đến thư mục images
imageDir = os.path.join(baseDir, "images")

# Kiểm tra nếu thư mục images tồn tại
if not os.path.exists(imageDir):
    print(f"Thư mục không tồn tại: {imageDir}")
    exit()
else:
    print(f"Thư mục tồn tại: {imageDir}")

currentId = 1
labelIds = {}
yLabels = []
xTrain = []

# Duyệt qua tất cả các thư mục con và tệp trong thư mục images
for root, dirs, files in os.walk(imageDir):
    for file in files:
        if file.lower().endswith(("png", "jpg", "jpeg")):
            path = os.path.join(root, file)
            print(f"Đang xử lý tệp: {path}")
            label = os.path.basename(root).replace(" ", "_").lower()

            if label not in labelIds:
                labelIds[label] = currentId
                currentId += 1

            id_ = labelIds[label]
            try:
                pilImage = Image.open(path).convert("L")
            except Exception as e:
                print(f"Lỗi khi mở hình ảnh {path}: {e}")
                continue

            imageArray = np.array(pilImage, "uint8")
            
            # Phát hiện khuôn mặt chính diện
            faces = faceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)
            # Phát hiện mặt nghiêng
            profile_faces = profileFaceCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)
            # Phát hiện mắt
            eyes = eyeCascade.detectMultiScale(imageArray, scaleFactor=1.1, minNeighbors=5)
            # Phát hiện miệng
            smiles = smileCascade.detectMultiScale(imageArray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))

            if len(faces) == 0 and len(profile_faces) == 0 and len(eyes) == 0 and len(smiles) == 0:
                print(f"Không phát hiện đặc điểm trong ảnh: {path}")
                continue  # Bỏ qua hình ảnh này nếu không có đặc điểm nào

            for (x, y, w, h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)

            for (x, y, w, h) in profile_faces:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)

            for (x, y, w, h) in eyes:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)

            for (x, y, w, h) in smiles:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)

# Kiểm tra nếu có dữ liệu để huấn luyện
if len(xTrain) == 0 or len(yLabels) == 0:
    print("Không có dữ liệu huấn luyện. Vui lòng kiểm tra lại hình ảnh trong thư mục 'images'.")
    exit()

# Lưu nhãn (labels) vào file
with open(os.path.join(baseDir, "labels"), "wb") as f:
    pickle.dump(labelIds, f)

# Huấn luyện mô hình và lưu kết quả
trainerPath = os.path.join(baseDir, "trainer.yml")
recognizer.train(xTrain, np.array(yLabels))
recognizer.save(trainerPath)
print("Đã hoàn thành huấn luyện.")
print(labelIds)
