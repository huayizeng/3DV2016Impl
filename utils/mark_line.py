import cv2
import pickle
from IPython import embed
import numpy as np
import random

<<<<<<< HEAD
# line_sets = []
# x1 = -1
# y1 = -1
# x2 = -1
# y2 = -1
# drawing = False
# line_color = (0, 200, 0)
#
#
# def set_position(event, x, y, flags, param):
#     global x1, y1, x2, y2, drawing, line_color
#
#     if event == cv2.EVENT_RBUTTONDOWN:
#         line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         print(line_color)
#
#     elif event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         x1 = x
#         y1 = y
#         print(x, y)
#     #
#     # elif event == cv2.EVENT_MOUSEMOVE and drawing is True:
#     #     cv2.line(img, (x1, y1), (x, y), line_color, 5)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing = False
#         cv2.line(img, (x1, y1), (x, y), line_color, 5)
#         x2 = x
#         y2 = y
#         line = np.array([[x1, y1], [x2, y2]])
#         line_sets.append(line)
#         print("line saved!")
#         print(line)
#
#
# img = cv2.imread('../data/test.png')
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', set_position)
#
# print(img.shape)
#
# while 1:
#     cv2.imshow('image', img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cv2.destroyAllWindows()
# cv2.imwrite('../data/test_marked.png', img)
#
# with open('../data/data.pkl','wb') as f:
#     pickle.dump(line_sets, f)
#
# # [[500 234]
# #  [599 340]]
# # line saved!
# # [[600 195]
# #  [694 290]]
#

with open('../data/lines_set.pkl', 'rb') as f:
    data = pickle.load(f)

cluster = np.array([[0, 1], [2, 3, 4, 5], [6, 7, 8,9], [10, 11, 12, 13, 14], [15, 16]])


def get_line(list):
    k_list = []
    b_list = []
    for i in range(list.shape[0]):
        x1 = data[i][0][0]
        y1 = data[i][0][1]
        x2 = data[i][1][0]
        y2 = data[i][1][1]

        k = (y2 - y1)/ (x2 - x1)
        b = y1 - k * x1

        k_list.append(k)
        b_list.append(b)

    return np.array(k_list), np.array(b_list)


def vp(cluster):
    for i in range(cluster.shape[0]):
        k, b = get_line(cluster[i])

=======
line_sets = []
x1 = -1
y1 = -1
x2 = -1
y2 = -1
drawing = False
line_color = (0, 200, 0)


def set_position(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, line_color

    if event == cv2.EVENT_RBUTTONDOWN:
        line_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        print(line_color)

    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1 = x
        y1 = y
    #
    # elif event == cv2.EVENT_MOUSEMOVE and drawing is True:
    #     cv2.line(img, (x1, y1), (x, y), line_color, 5)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (x1, y1), (x, y), line_color, 5)
        x2 = x
        y2 = y
        line = np.array([[x1, y1], [x2, y2]])
        line_sets.append(line)
        print("line saved!")
        print(line)


img = cv2.imread('../data/test.png')
cv2.namedWindow('image')
cv2.setMouseCallback('image', set_position)

print(img.shape)

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cv2.imwrite('../data/test_marked.png', img)

with open('../data/data.pkl','wb') as f:
    pickle.dump(line_sets, f)

# [[500 234]
#  [599 340]]
# line saved!
# [[600 195]
#  [694 290]]
>>>>>>> 96fea252d01b862358ce790e7d4d5720116da6ef
