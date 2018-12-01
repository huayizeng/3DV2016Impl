import cv2
import pickle
from IPython import embed
import numpy as np
import random


############### dist of the projection #####################

dist = 'white house'
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
        print(x, y)
    #
    # elif event == cv2.EVENT_MOUSEMOVE and drawing is True:
    #     cv2.line(img, (x1, y1), (x, y), line_color, 5)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (x1, y1), (x, y), line_color, 2)
        x2 = x
        y2 = y
        line = np.array([[x1, y1], [x2, y2]])
        line_sets.append(line)
        print("line saved!")
        print(line)


img = cv2.imread('../data/' + dist + '/projection.png')
cv2.namedWindow('image')
cv2.setMouseCallback('image', set_position)

# print(img.shape)

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
#
cv2.destroyAllWindows()
cv2.imwrite('../data/'+ dist + '/test_marked.png', img)

with open('../data/' + dist + '/lines_set.pkl','wb') as f:
    pickle.dump(line_sets, f)