import numpy as np
from numpy import *
import pickle
import 
from IPython import embed

###################### data import ##############################################

with open("../data/lines_set.pkl", "rb") as f:
    lines = pickle.load(f)
lines = np.array(lines)
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


#################### information of the projection ################################
F = 450
H = 540
W = 960

K = np.asarray([[F, 0.0, 480.0],
        [0.0, F, 270.0],
        [0.0, 0.0, 1.0]])
Rt = np.asarray([[-0.8475, -0.425, 0.6375, 75],
                     [-0.2125, 0.7225, -0.11125, 0],
                     [0.425, -0.5, -0.7225, -200]])
test = np.array([[np.cos(3.14/4), 0, -np.sin(3.14/4)],
    				 [0, 1, 0],
    				 [np.sin(3.14/4), 0, np.cos(3.14/4)]
    				 ])
Rt[:, :3] = np.dot(test, Rt[:, :3])
KR = np.dot(K, Rt[:, :3])

# vanishing point
vp = []
vp_threshold = 150


def plane_pi_normal(input_point_set):
    # The input are two points which are the points for the line we want to calculate. Each point is form like (x, y)
    
    point_set = input_point_set.copy()
    point_set[:, 0] -= int(W/2)
    point_set[:, 1] -= int(H/2)
    # embed()
    point_set = np.concatenate((point_set, np.array([[F, 1], [F, 1]])), axis=1)
    point_set = np.dot(np.array(mat(np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)).I), point_set.T).T

    # calculate the 3D points of the projection in the co-ordinate of 3D model
    p_cam = np.append(Rt[:, 3], 1)

    # calculate the direction and do regulation
    d1 = (point_set[0, :3] - p_cam[:3]) / np.sqrt(np.sum(np.square(point_set[0, :3] - p_cam[:3])))
    d2 = (point_set[1, :3] - p_cam[:3]) / np.sqrt(np.sum(np.square(point_set[1, :3] - p_cam[:3])))

    normal_pi = np.cross(d1, d2)
    normal_pi /= normal_pi[2]
    return normal_pi


def plane_normal(d1, d2):
    # calculate the normal for the plane contoured by two lines with direction
    normal = np.cross(d1, d2)
    normal /= np.sqrt(np.sum(np.square(normal[0, :])))
    return normal


def vp_cal(nodes):
	#### lines in the nodes should be in the same cluster
    nodes = np.array(nodes)
    cluster_lines = []
    list_vp = []
    global lines_no_cluster, lines_check

    

    for i in nodes:
        ########## do some experiment to check which expressions are correct

        # k = (lines[i][0][0] - lines[i][1][0]) / (lines[i][0][1] - lines[i][1][1])
        # b = lines[i][0][0] - k * lines[i][0][1]
        # cluster_lines.append(np.array((k, b)))
        k = (lines[i][0][1] - lines[i][1][1]) / (lines[i][0][0] - lines[i][1][0])
        b = lines[i][0][1] - k * lines[i][0][0]
        cluster_lines.append(np.array((k, b)))

        # pop the clustered lines
        index = np.where(lines_no_cluster == i)
        lines_no_cluster = np.delete(lines_no_cluster, index)
        lines_check[i] = 1

    # print("cluster_k,b : {}".format(cluster_lines))

    # use two lines to calculate vp, and return the average result
    for i in range(len(cluster_lines) - 1):

        a = np.array([[cluster_lines[0][0], -1], [cluster_lines[i + 1][0], -1]])
        b = np.array([-cluster_lines[0][1], -cluster_lines[i + 1][1]])
        x = np.linalg.solve(a, b)
        list_vp.append(x)

    # print("list_vp: {}".format(list_vp))
	
    # embed()
    list_vp = np.array(list_vp)
    print("Average vp: {}".format(np.mean(list_vp, axis=0)))
    # print("current no clustered lines: {}".format(lines_no_cluster))

    return np.mean(list_vp, axis=0)


def find_cluster(dir, clustered_line):
    global lines_no_cluster, lines_normal, lines_check
    new_vp = np.dot(KR, dir.T).T
    new_vp /= new_vp[0][2]
    new_cluster = []

    count = 0
    for new_line in range(lines_no_cluster.shape[0]):
    	new_line -= count
    	i = lines_no_cluster[new_line]
    	k = (lines[i][0][1] - lines[i][1][1]) / (lines[i][0][0] - lines[i][1][0])
    	b = lines[i][0][1] - k * lines[i][0][0]
    	y = k * new_vp[0][0] + b

    	# print(abs(y - new_vp[0][1]))
    	
    	if abs(y - new_vp[0][1]) <= vp_threshold:
    		new_cluster.append(i)
    		lines_normal[i] = dir
    		lines_check[i] = 1
    		lines_no_cluster = np.delete(lines_no_cluster, new_line)
    		count += 1

    return new_cluster


if __name__ == "__main__":

    # calculate known vp
    for i in range(origin_cluster.shape[0]):
        vp_res = vp_cal(origin_cluster[i])
        vp.append(vp_res)

        ##### algrothim #####
        ## p = [K,0][[R, t],[0, 1]][d, 0].T, where K.shape = (3,3), R.shape = (3, 3), t.shape = (3, 1)
        ## Considering 0 in [d, 0].T, we find that t in rotation matrix make no sense.
        ## So the equation for direction vector d = R.T dot K.I dot p still works.


        # cal the direction vector and do the normalization 
        normal = np.dot(np.dot(mat(Rt[:, :3]).I, mat(K).I), np.append(vp_res, np.array([1])))
        normal /= np.sqrt(np.sum(np.square(normal)))

        cluster_normal[i] = normal
        for j in origin_cluster[i]:
            lines_normal[j] = normal
            lines_check[j] = 1

    #### now the direction vectors for all the clustered lines have been calculated

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

                    # embed()
                    faces_normal[face] = plane_normal(cluster_normal[cluster_list[0]], cluster_normal[cluster_list[1]])
                    faces_check[face] = 1



        for line in range(lines.shape[0]):
            if lines_check[line] == 0:
                for face in range(faces.shape[0]):
                    if line in faces[face] and faces_check[face] != 0 and line not in origin_cluster[origin_cluster.shape[0] - 1]:
                        cal_line_normal = np.cross(faces_normal[face], plane_pi_normal(lines[line]))
                        cal_line_normal /= np.sqrt(np.sum(np.square(cal_line_normal[0,:])))
                        lines_normal[line] = cal_line_normal
                        lines_check[line] = 1

                        l = find_cluster(lines_normal[line], line)
                        # embed()
                        origin_cluster = list(origin_cluster)
                        origin_cluster.append(l)
                        origin_cluster = np.array(origin_cluster)
                        # origin_cluster = np.concatenate((origin_cluster, np.array(find_cluster(lines_normal[line], line))))
                        cluster_normal.append(lines_normal[line])
                        
        print(lines_check)
        print(faces_check)
        

    with open("../data/lines_dir.pkl", "wb") as f:
    	pickle.dump(lines_normal, f)
    	# print("lines_normal: {}".format(lines_normal))

    with open("../data/faces_normal.pkl", "wb") as f:
    	pickle.dump(faces_normal, f)
    	# print("faces_normal: {}".format(faces_normal))
    print(origin_cluster)


