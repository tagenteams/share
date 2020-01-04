import cv2
import numpy as np

# Webカメラから入力
cap = cv2.VideoCapture(0)

# カスケードファイルを指定して、検出器を作成
face_cascade_file = "haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_file)


# モザイクを作る関数
def mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]

    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)

    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2


while True:
    # 画像を取得
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 3)

    for (x, y, w, h) in faces:
        img = mosaic(img, (x, y, x + w, y + h), 10)

    cv2.imshow('img', img)

    # ESCかEnterキーが押されたら終了
    k = cv2.waitKey(1)
    if k == 13:
        break

cap.release()
cv2.destroyAllWindows()
