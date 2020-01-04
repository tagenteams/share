#!/usr/bin/python2
# coding: utf-8

import cv2
import numpy

cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')  # 顔認識用の特徴量

cam = cv2.VideoCapture(0)  # カメラを開く

laugh = cv2.imread('laughingman.png', -1)  # 笑い男画像の読み込み。-1はαチャンネル付きということのようだ？
mask = cv2.cvtColor(laugh[:, :, 3], cv2.COLOR_GRAY2BGR) / 255.0  # 笑い男からαチャンネルだけを抜き出して0から1までの値にする。あと3チャンネルにしておく。
laugh = laugh[:, :, :3]  # αチャンネルはもういらないので消してしまう。

while True:
    ret, img = cam.read()  # カメラから画像を読み込む。

    if not ret:
        print('error?')
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 画像認識を高速に行うためにグレースケール化。
    gray = cv2.resize(gray, (int(img.shape[1]/4.0) , int(img.shape[0]/4.0 )))  # そのままだと遅かったので画像を4分の1にしてさらに高速化。

    faces = cascade.detectMultiScale(gray)  # 顔を探す。

    if len(faces) > 0:
        for rect in faces:
            rect *= 4  # 認識を4分の1のサイズの画像で行ったので、結果は4倍しないといけない。

            # そのままだと笑い男が小さくって見栄えがしないので、少し大きくしてみる。
            #  単純に大きくするとキャプチャした画像のサイズを越えてしまうので少し面倒な処理をしている。
            rect[0] -= min(25, rect[0])
            rect[1] -= min(25, rect[1])
            rect[2] += min(50, img.shape[1] - (rect[0] + rect[2]))
            rect[3] += min(50, img.shape[0] - (rect[1] + rect[3]))

            # 笑い男とマスクを認識した顔と同じサイズにリサイズする。
            laugh2 = cv2.resize(laugh, tuple(rect[2:]))
            mask2 = cv2.resize(mask, tuple(rect[2:]))

            # 笑い男の合成。
            img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = laugh2[:,:] * mask2 + \
                                                                        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] * (1.0 - mask2)

    cv2.imshow('laughing man', img)

    if cv2.waitKey(10) > 0:
        break

cam.release()
cv2.destroyAllWindows()
