import cv2


# read image
image = cv2.imread('./images/flower.jpg')

b = image.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

# RGB - image with Blue channel
cv2.imshow('B-RGB', b)

cv2.imwrite('./results/blue.jpg', b)

cv2.waitKey(0)