import cv2 as cv

img = cv.imread("C:/Users/Praneeth Madusanka/Desktop/hand/New folder/CNN/custom_model_data/0/IMG_1124.JPG")
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('hsv', img)
img = cv.calcHist([img], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv.imshow('calhist', img)
cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
cv.imshow('nor', img)
cv.waitKey(0)
cv.destroyAllWindows()
