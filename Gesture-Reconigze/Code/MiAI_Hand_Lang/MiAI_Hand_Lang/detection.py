#! /usr/bin/env python3

import copy
import cv2
import numpy as np
from keras.models import load_model
import time
import pickle
# Cac khai bao bien
prediction = ''
score = 0
bgModel = None
alpha=1# thay đổi độ sáng
beta=-20# thay đổi độ tương phản 
gesture_names = {0: 'E',
                 1: 'L',
                 2: 'F',
                 3: 'V',
                 4: 'B'}

# Load model tu file da train
model = load_model("C:/Users/thi/Documents/thi/mymodel.h5")

# Ham de predict xem la ky tu gi
def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255 #image=image/255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


# Ham xoa nen khoi anh
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate) #global background do ng dùng đặt
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def get_hand_hist():
    with open("hand_histogram/hist", "rb") as f:
        hist = pickle.load(f)
    return hist


# Khai bao kich thuoc vung detection region
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Cac thong so lay threshold
threshold = 60 #giá trị ngưỡng để chuyển sang ảnh đen trắng, nếu lơn hơn threshold thì là trắng, bé hơn thì là đen
blurValue = 41
bgSubThreshold = 50#50    #change
learningRate = 0

# Nguong du doan ky tu
predThreshold= 95

isBgCaptured = 0  # Bien luu tru da capture background chua

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

hist = get_hand_hist()

while camera.isOpened():
    # Doc  tu webcam
    ret, frame = camera.read()


    frame=cv2.addWeighted(frame,alpha,np.zeros(frame.shape, frame.dtype),0,beta)

    # kích thước ảnh đọc được là 480,640,3
    # Lam min anh
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Lat ngang anh
    frame = cv2.flip(frame, 1)

    # Ve khung hinh chu nhat vung detection region
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    # Neu ca capture dc nen
    if isBgCaptured == 1:
        # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        # frame = clahe.apply(frame)
        # Tach nen
        img = remove_background(frame)
        # Lay vung detection
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI


        # Chuyen ve den trang
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#chuyen về ảnh xám
        # dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)#hand_histogram
        # disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        #cv2.filter2D(dst,-1,disc,dst)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#chuyen về ảnh xám
        # blur = cv2.GaussianBlur(dst, (blurValue, blurValue), 0)# bị đổi background
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0) # làm mịn ảnh, hay gọi là làm mờ ,blurValue kích thước GaussianBlur, càng lớn càng mờ 

        cv2.imshow('black_white', cv2.resize(blur, dsize=None, fx=0.5, fy=0.5)) # thay đổi kích thước hình ảnh

        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #anh trăng đen

        cv2.imshow('thresh', cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5))
        # lây FPS
        # fps = camera.get(cv2.CAP_PROP_FPS)
        # print(fps)

        if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.2):
            # Neu nhu ve duoc hinh ban tay
            if (thresh is not None):
                # Dua vao mang de predict
                target = np.stack((thresh,) * 3, axis=-1)
                target = cv2.resize(target, (224, 224))
                target = target.reshape(1, 224, 224, 3)
                prediction, score = predict_rgb_image_vgg(target)

                # Neu probality > nguong du doan thi hien thi
                print(score,prediction)
                if (score>=predThreshold):
                    cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                (0, 0, 255), 10, lineType=cv2.LINE_AA)
    thresh = None

    # Xu ly phim bam
    k = cv2.waitKey(10)
    if k == ord('q'):  # Bam q de thoat
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
        print('Background captured')

    elif k == ord('a'):
        alpha+=1
        print('increase alpha')

    elif k == ord('s'):
        alpha-=1
        print('reduce alpha')

    elif k == ord('z'):
        beta+=1
        print('increase beta')

    elif k == ord('x'):
        beta-=1
        print('reduce beta')

    elif k == ord('r'):

        bgModel = None
        isBgCaptured = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255),10,lineType=cv2.LINE_AA)
        print('Background reset')
        time.sleep(1)


    cv2.imshow('begin', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))


cv2.destroyAllWindows()
camera.release()
