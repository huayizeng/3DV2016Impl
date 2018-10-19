import numpy as np
from numpy import *
import pickle
from IPython import embed

with open("../data/lines_set.pkl", "rb") as f:
    lines = pickle.load(f)

with open("../data/cluster.pkl", "rb") as f:
    cluster = pickle.load(f)

with open("../data/faces.pkl", "rb") as f:
    faces = pickle.load(f)

# information of the projection
K = np.array([[1050.0, 0.0, 480.0], [0.0, 1050.0, 270.0], [0.0, 0.0, 1.0]])
Rt = np.array([[0.6859206557273865, 0.7276763319969177, -4.011331711240018e-09, -0.3960070312023163],
               [0.32401347160339355, -0.3054208755493164, -0.8953956365585327, 0.3731381893157959],
               [-0.6515582203865051, 0.6141703724861145, -0.44527140259742737, 11.250574111938477]])
F = 1050
H = 540
W = 960

# vanishing point
vp = {}
normal = {}
vp_threshold = 50


def plate_pi_normal(point_set):
    # The input are two points which are the points for the line we want to calculate. Each point is form like (y, x)
    point_set[:, 0] -= H/2
    point_set[:, 1] -= W/2
    point_set[:, [0, 1]] = point_set[:, [1, 0]]
    point_set = np.concatenate((point_set, np.array([[F, 1], [F, 1]])))
    point_set = np.array(mat(np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)).I) * point_set.T

    # calculate the 3D points of the projection in the co-ordinate of 3D model
    p_cam = np.append(Rt[:, 3], 1)

    # calculate the direction and do regulation
    d1 = (point_set[0, :3] - p_cam[:3]) / np.sum(point_set[0, :3] - p_cam[:3])
    d2 = (point_set[1, :3] - p_cam[:3]) / np.sum(point_set[1, :3] - p_cam[:3])

    normal_pi = np.cross(d1, d2)

    return normal_pi


def plate_normal(d1, d2):
    # calculate the normal for the plate contoured by two lines with direction
    return np.cross(d1, d2)


def vp_cal(points):
    points = np.array(points)
    cluster_lines = []
    list_vp = []

    for i in points:
        k = (lines[i][0][0] - lines[i][1][0]) / (lines[i][0][1] - lines[i][1][1])
        b = lines[i][0][0] - k * lines[i][0][1]
        cluster_lines.append(np.array((k, b)))

    # use two lines to calculate vp, and return the average result
    for i in range(len(cluster_lines) - 1):

        a = np.array([[cluster_lines[0][0], -1], [cluster_lines[i + 1][0], -1]])
        b = np.array([-cluster_lines[0][1], -cluster_lines[i + 1][1]])
        x = np.linalg.solve(a, b)
        list_vp.append(x)

    # a = np.array([[cluster_lines[0][0], -1], [cluster_lines[2][0], -1]])
    # b = np.array([-cluster_lines[0][1], -cluster_lines[2][1]])
    # x2 = np.linalg.solve(a, b)

    list_vp = np.array(list_vp)

    return np.mean(list_vp, axis=1)


if __name__ == "__main__":
    # owing that the lines marked has been clustered. Thus we create a new cluster for the cluster generation
    origin_cluster = cluster[1:4]

    # calculate known vp
    for i in range(origin_cluster.shape[0]):
        vp[i] = vp_cal(origin_cluster[i])


