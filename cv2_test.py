import cv2
import numpy as np
import matplotlib.pyplot as plt
from upscailing import up

# 이미지를 로드합니다

img_path = './scan/datas/FIT/'
img_name = 'FIT_page_1.jpg'

# up(img_path, img_name)
# image = cv2.imread(f"{img_path}up_{img_name}")
image = cv2.imread(f"{img_path}{img_name}")

# image = cv2.imread('./scan/datas/pic.jpg')

# 이미지를 HSV 색상 공간으로 변환합니다
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색의 HSV 범위를 정의합니다 (두 범위를 사용하여 빨간색을 포함)
lower_red1 = np.array([0, 170, 0])
upper_red1 = np.array([20, 255, 255])
lower_red2 = np.array([160, 170, 100])
upper_red2 = np.array([180, 255, 255])

# 빨간색 범위에 해당하는 마스크를 생성합니다
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
# mask = cv2.add(mask1, mask2)
mask = mask1 + mask2

# 마스크를 사용하여 원본 이미지에서 빨간색 부분을 추출합니다
red_only = cv2.bitwise_and(image, image, mask=mask)

# 그레이스케일로 변환합니다
gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)

# 가우시안 블러를 적용합니다
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# 허프 원 변환을 사용하여 원을 검출합니다
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
                           param1=1, param2=1, minRadius=0, maxRadius=10)

# 원을 그립니다
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

# 결과를 보여줍니다
# cv2.imshow("Detected Red Circles", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('./scan/datas/detected_circles.jpg', image)
