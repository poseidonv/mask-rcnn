import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

img = cv2.imread('/home/poseidon/Downloads/airplane.jpg')
final_img = np.ones((800,1200,3), dtype = np.uint8) * 255
coordinates = [(179,134),(421,134),(141,199),(458,201),(301,137),(301,206),(301,160)]
coordinates_left = [(240,369),(220,402),(301,370),(301,380),(301,406),(361,368),(380,402)]
coordinates_right = [(840,168),(822,201),(900,170),(900,182),(900,205),(961,168),(981,202)]
# for coordinate in coordinates:
#     left = tuple([coordinate[1]+150, coordinate[0]+300])
#     right = tuple([coordinate[1]+750, coordinate[0]+100])
#     coordinates_left.append(left)
#     coordinates_right.append(right)
print(coordinates_left)
print(coordinates_right)

for coordinate in coordinates:
    cv2.circle(img, coordinate, 5, (0,0,255), -1)

img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
row, col = 200, 300
roi = final_img[300:300+row, 450:450+col]

img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 250, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
# print(mask.shape)
# print(roi.shape)

final_img_bg = cv2.bitwise_and(roi, roi, mask = mask)
img_fg = cv2.bitwise_and(img, img, mask = mask_inv)

dst = cv2.add(final_img_bg, img_fg)
final_img[300:300+row, 150:150+col] = dst
final_img[100:100+row, 750:750+col] = dst
cv2.line(final_img, (600,0),(600,800), (0,0,0), 10)
# for left, right in zip(coordinates_left, coordinates_right):
#     cv2.line(final_img, left,right, (0,255,0), 2, lineType=5)
cv2.imwrite('dst.jpg', dst)

cv2.imwrite('final_img_bg.jpg', final_img_bg)
cv2.imwrite('img_fg.jpg', img_fg)

cv2.imwrite('mask.jpg', mask)

cv2.imwrite('final_img.jpg', final_img)
cv2.imwrite('img.jpg',img)
