import cv2

# read image
image = cv2.imread('./images/flower.jpg')

# convert image to grayscale
img_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show grayscale image
cv2.imshow('Gray-scale', img_gs)

# save result to "results" directory
cv2.imwrite('./results/gs.jpg', img_gs)

cv2.waitKey(0)