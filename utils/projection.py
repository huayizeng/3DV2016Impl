"""
This file aims at test if the camera model is correct. 
We render the silhouette as a binary image, and compare it to 'test.png'
"""
import numpy as np
import math
import os
import cv2  
from IPython import embed
from mesh import read_obj

def get_cam_matrix():
    f = 450.0
    K = np.asarray([[f, 0.0, 480.0],
        [0.0, f, 270.0],
        [0.0, 0.0, 1.0]])
    # RT = np.asarray([[0.6859206557273865, 0.7276763319969177, -4.011331711240018e-09, -0.3960070312023163],
    #     [0.32401347160339355, -0.3054208755493164, -0.8953956365585327, 0.3731381893157959],
    #     [-0.6515582203865051, 0.6141703724861145, -0.44527140259742737, 200]])
    RT = np.asarray([[-0.8475, -0.425, 0.6375, 75],
                     [-0.2125, 0.7225, -0.11125, 0],
                     [0.425, -0.5, -0.7225, -200]])
    test = np.array([[np.cos(3.14/4), 0, -np.sin(3.14/4)],
    				 [0, 1, 0],
    				 [np.sin(3.14/4), 0, np.cos(3.14/4)]
    				 ])
    RT[:, :3] = np.dot(test, RT[:, :3])
    return K, RT


if __name__ == '__main__':
    K, RT = get_cam_matrix()
    R = RT[:, :3]
    T = RT[:, 3]
    img_test = cv2.imread("../data/test.png")
    print("img_test.shape: ", img_test.shape[:2])
    k_intensity_background = 50
    im_reprojection = np.ones(img_test.shape[:2]) * k_intensity_background

    # raise NotImplementedError
    # TODO: Read .obj
    # TODO: In the following code, pts is sampled points from .obj; However, it's not correct. 
    # TODO: Instead, we should read .obj files, render each line.     
    path_pts = "../data/3dv2016.pts"
    point_set = np.loadtxt(path_pts).astype(np.float32)
    point_set, face = read_obj("../data/3dv2016.obj")
    # embed()
    # homog_point = np.append(point_set, np.ones([point_set.shape[0], 1]), 1)
    # projected_point = np.dot(cam, homog_point.transpose()).transpose()

    for p in point_set:
        p = np.dot(R, p) + T
        p = np.dot(K, p)
        p0_x = int(p[0] / p[2])
        p0_y = int(p[1] / p[2])
        if p0_x < 0 or p0_x >= im_reprojection.shape[1] or p0_y <0 or p0_y >= im_reprojection.shape[0]:
            continue
        im_reprojection[p0_y, p0_x] = 255
    path_image = "im_reprojection.jpg"
    cv2.imwrite(path_image, im_reprojection)
