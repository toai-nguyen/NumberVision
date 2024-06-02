import cv2
from imutils.contours import sort_contours
import imutils
import numpy as np
import os
# get contour position of image

def paddingImage(thresh):
  (tH, tW) = thresh.shape
  dX = int(max(0, 36 - tW) / 2.0)
  dY = int(max(0, 36 - tH) / 2.0)
  kernel = np.ones((5,5), np.uint8)
  padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
    value=(0, 0, 0))
  padded = cv2.resize(padded, (28, 28))
  return padded


def getContoursPosition(img):
  pos = []

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  imgBlur = cv2.GaussianBlur(gray, (7,7), 0)
  ret, thresh = cv2.threshold(imgBlur,150, 255, cv2.THRESH_BINARY_INV)

  edges = cv2.Canny(thresh, 100, 200)
  contours, hierachy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contours = sort_contours(contours, method="left-to-right")[0]

  for cnt in contours:
    if(cv2.contourArea(cnt) < 10):
      continue
    (x,y,w,h) = cv2.boundingRect(cnt)
    pos.append((x,y,w,h))
  return pos


def getROI(img, pos):
  train = []
  kernel = np.ones((5,5), np.uint8)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  for (x,y,w,h) in pos:
    roi = gray[y: y + h, x: x + w]
    ret, thresh = cv2.threshold(roi,150, 255, cv2.THRESH_BINARY_INV)
    (tH, tW) = thresh.shape
    if tW > tH:
      thresh = imutils.resize(thresh, width = 28)
    else:
      thresh = imutils.resize(thresh, height = 28)
    thresh = paddingImage(thresh)
    train.append(thresh)
  return train


def getPredict(train, model):
  y_pred = []
  for item in train:
    img_scale = np.array([item/255.0])
    y_proba = model.predict(img_scale)
    y_pred.append(np.argmax(y_proba))
  return y_pred

def findPredictOfImage(img, pos, model):
  train = getROI(img, pos)
  y_pred = getPredict(train, model)

  return y_pred
def drawPredict(img , y_pred, pos):
  i = 0
  label = "0123456789"
  for (x,y,w,h) in pos:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 5)
    cv2.putText(img, label[y_pred[i]], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    i += 1
  return img

def getPredictOfImage(file, model):
  img = cv2.imread(file)
  pos = getContoursPosition(img)
  y_pred = findPredictOfImage(img, pos, model)
  img = drawPredict(img, y_pred, pos)
  val = " ".join(str(num) for num in y_pred)
  
  save_directory = 'static/answer/'
  filename = 'saved_image.jpg'
  file_path = os.path.join(save_directory, filename)

  os.makedirs(save_directory, exist_ok=True)

  cv2.imwrite(file_path, img)

  predictArr = [file_path, val]
  return predictArr