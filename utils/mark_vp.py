import numpy as np
from numpy import *
import pickle
from IPython import embed

with open("../data/lines_set.pkl", "rb") as f:
    lines = pickle.load(f)
lines_no_cluster = np.array(range(lines.shape[0]))
lines_check = np.zeros(lines.shape[0])
lines_normal = [0 for _ in range(lines.shape[0])]

with open("../data/cluster.pkl", "rb") as f:
    cluster = pickle.load(f)
    # owing that the lines marked has been clustered. Thus we create a new cluster for the cluster generation
origin_cluster = cluster[1:4]
cluster_normal = [0 for _ in range(origin_cluster.shape[0])]

with open("../data/faces.pkl", "rb") as f:
    faces = pickle.load(f)
faces_check = np.zeros(faces.shape[0])
faces_normal = [0 for _ in range(faces.shape[0])]

# information of the projection
K = np.array([[1050.0, 0.0, 480.0], [0.0, 1050.0, 270.0], [0.0, 0.0, 1.0]])
Rt = np.array([[0.6859206557273865, 0.7276763319969177, -4.011331711240018e-09, -0.3960070312023163],
               [0.32401347160339355, -0.3054208755493164, -0.8953956365585327, 0.3731381893157959],
               [-0.6515582203865051, 0.6141703724861145, -0.44527140259742737, 11.250574111938477]])
KR = np.dot(K, Rt[:, :3])

F = 1050
H = 540
W = 960

# vanishing point
vp = []
vp_threshold = 50


def plate_pi_normal(point_set):
    # The input are two points which are the points for the line we want to calculate. Each point is form like (y, x)
    # embed()
    point_set[:, 0] -= int(H/2)
    point_set[:, 1] -= int(W/2)
    point_set[:, [0, 1]] = point_set[:, [1, 0]]
    # embed()
    point_set = np.concatenate((point_set, np.array([[F, 1], [F, 1]])), axis=1)
    embed()
    point_set = np.dot(np.array(mat(np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)).I), point_set.T)

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
    global lines_no_cluster

    for i in points:
        # 做实验，看一下哪种表达是正确的
        # k = (lines[i][0][0] - lines[i][1][0]) / (lines[i][0][1] - lines[i][1][1])
        # b = lines[i][0][0] - k * lines[i][0][1]
        # cluster_lines.append(np.array((k, b)))
        k = (lines[i][0][1] - lines[i][1][1]) / (lines[i][0][0] - lines[i][1][0])
        b = lines[i][0][1] - k * lines[i][0][0]
        cluster_lines.append(np.array((k, b)))

        # pop the clustered lines
        index = np.where(lines_no_cluster == i)
        lines_no_cluster = np.delete(lines_no_cluster, index)

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
    print("Average vp: {}".format(np.mean(list_vp, axis=0)))
    print("current no clustered lines: {}".format(lines_no_cluster))
    return np.mean(list_vp, axis=0)


def find_cluster(dir, clustered_line):
    global lines_no_cluster, lines_normal, lines_check
    new_vp = np.dot(KR, dir)
    new_vp /= new_vp[2]
    new_cluster = [clustered_line]
    for new_line in range(lines_no_cluster.shape[0]):
        i = lines_no_cluster[new_line]
        # 跟上面的内容一并修改，这里先写一个猜测的脚本，这里第二位1是y，0是x
        k = (lines[i][0][1] - lines[i][1][1]) / (lines[i][0][0] - lines[i][1][0])
        b = lines[i][0][1] - k * lines[i][0][0]
        y = k * new_vp[0] + b

        if abs(y - new_vp[1]) <= vp_threshold:
            new_cluster.append(i)
            lines_normal[i] = dir
            lines_check[i] = 1
            lines_no_cluster = np.delete(lines_no_cluster, new_line)

    return new_cluster


if __name__ == "__main__":

    # calculate known vp
    for i in range(origin_cluster.shape[0]):
        vp_res = vp_cal(origin_cluster[i])

        # 这里要不要先平移t的位置再做点乘？？
        normal = np.dot(np.dot(mat(Rt[:, :3]).I, mat(K).I), np.append(vp_res, np.array([1])))
        cluster_normal[i] = normal
        for j in origin_cluster[i]:
            lines_normal[j] = normal
            lines_check[j] = 1
        vp.append(vp_res)
    # embed()

    while np.where(lines_check == 0)[0].shape[0] != 0 or np.where(faces_check == 0)[0].shape[0] != 0:
        for face in range(faces.shape[0]):
            # find outlines for face
            if faces_check[face] == 0:
                cluster_list = []
                for line in faces[face]:
                    for cluster_iter in range(origin_cluster.shape[0]):
                        if line in origin_cluster[cluster_iter] and cluster_iter not in cluster_list:
                            cluster_list.append(cluster_iter)

                if len(cluster_list) >= 2:
                    faces_normal[face] = plate_normal(cluster_normal[cluster_list[0]], cluster_normal[cluster_list[1]])
                    faces_check[face] = 1

        for line in range(lines.shape[0]):
            if lines_check[line] == 0:
                for face in range(faces.shape[0]):
                    if line in faces[face] and faces_check[face] != 0:
                        lines_normal[line] = np.dot(faces_normal[face], plate_pi_normal(lines[line]))
                        lines_check[line] = 1
                        np.append(origin_cluster, find_cluster(lines_normal[line], line))
                        cluster_normal.append(lines_normal[line])
