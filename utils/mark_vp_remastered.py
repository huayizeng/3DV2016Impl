import numpy as np
import cv2
from numpy import *
import pickle
from IPython import embed
import vp_utils as util

###################### data import ##############################################

dist = "white house"
with open("../data/" + dist + "/lines_set.pkl", "rb") as f:
    lines = pickle.load(f)

lines = np.array(lines)
lines_check = np.zeros(lines.shape[0])
lines_normal = np.zeros((lines.shape[0], 3))

with open("../data/" + dist + "/cluster.pkl", "rb") as f:
    cluster = pickle.load(f)

with open("../data/" + dist + "/faces.pkl", "rb") as f:
    faces = pickle.load(f)
faces = np.array(faces)
faces_check = np.zeros(faces.shape[0])
faces_normal = np.zeros((faces.shape[0], 3))

###################### config initial ####################################
img = cv2.imread('../data/' + dist + '/projection.png')

F = 0
W = img.shape[1]
H = img.shape[0]
K = np.zeros((3, 3))
KI = np.zeros((3, 3))
R = np.zeros((3, 3))
RI = np.zeros((3, 3))
KR = np.zeros((3, 3))
vp_threshold = 200
vp = []

def vp_generation(cluster):
	#### lines in the cluster should be in the same cluster
    cluster = np.array(cluster)
    edgelets = util.lines_to_edgelet(lines[cluster])
    vp = util.ransac_vanishing_point(edgelets, 10, 300)
    return (vp/vp[2])[:2]

def camera_generation(vp):

	global F, K, R, KI, RI, KR

	x1 = vp[0]
	x2 = vp[1]
	x3 = vp[2]

	par = np.array([[x2[0] - x3[0], x2[1] - x3[1]], [x1[0] - x3[0], x1[1] - x3[1]]])
	res = np.array([x1[0] * (x2[0] - x3[0]) + x1[1] * (x2[1] - x3[1]), 
		x2[0] * (x1[0] - x3[0]) + x2[1] * (x1[1] - x3[1])])
	# embed()
	x0 = np.linalg.solve(par, res)

	# embed()

	F = np.sqrt(-np.dot((x1 - x0),(x2 - x0)))
	lambda1 = np.sqrt(np.cross(x2 - x3, x0 - x3) / np.cross(x2 - x3, x1 - x3))
	lambda2 = np.sqrt(np.cross(x3 - x1, x0 - x1) / np.cross(x2 - x3, x1 - x3))
	lambda3 = np.sqrt(np.cross(x1 - x2, x0 - x2) / np.cross(x2 - x3, x1 - x3))

	K = np.array([
		[F, 0, x0[0]],
		[0, F, x0[1]],
		[0, 0, 1]])

	R = np.array([
		[lambda1 * (x1[0] - x0[0]) / F, lambda2 * (x2[0] - x0[0]) / F, lambda3 * (x3[0] - x0[0]) / F],
		[lambda1 * (x1[1] - x0[1]) / F, lambda2 * (x2[1] - x0[1]) / F, lambda3 * (x3[1] - x0[1]) / F],
		[lambda1, lambda2, lambda3]])

	KI = np.array(mat(K).I)
	RI = np.array(mat(R).I)
	KR = np.dot(K, R)

	return

def plane_pi_normal(input_point_set):
    # The input are two points which are the points for the line we want to calculate. Each point is form like (x, y)
    
    # point_set = np.zeros_like(input_point_set).astype(float)

    # point_set[:, 0] = (input_point_set[:, 0] - K[0, 2]/2) / F
    # point_set[:, 1] = (input_point_set[:, 1] - K[1, 2]/2) / F

    point_set = np.concatenate((input_point_set, np.array([[F], [F]])), axis=1)
    point_set = np.dot(KI, point_set.T)

    direct = np.dot(RI, point_set).T
    direct = direct / np.sqrt(np.sum(np.square(direct), axis=1))[:, np.newaxis]
    pi_normal = np.cross(direct[0], direct[1])

    return pi_normal / np.sqrt(np.sum(np.square(pi_normal)))

def find_cluster(dir, clustered_line):

    new_vp = np.dot(KR, dir.T).T
    new_vp /= new_vp[2]
    new_cluster = [clustered_line]

    for new_line in range(lines.shape[0]):
    	if lines_check[new_line] == 1:
    		continue

    	x1 = lines[new_line, 0, 0]
    	x2 = lines[new_line, 1, 0]
    	y1 = lines[new_line, 0, 1]
    	y2 = lines[new_line, 1, 1]

    	dist = np.abs(((y2 - y1) * new_vp[0] + (x1 - x2) * new_vp[1] + (x2 * y1 - x1 * y2))/np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

    	# print(abs(y - new_vp[0][1]))
    	if dist <= vp_threshold:
    		new_cluster.append(new_line)
    		lines_normal[new_line] = dir
    		lines_check[new_line] = 1

    return new_cluster

def if_same_cluster(l1, l2):
	ll1 = 0
	ll2 = 0

	for i in range(cluster.shape[0]):
		if l1 in cluster[i]:
			ll1 = i
		if l2 in cluster[i]:
			ll2 = i

	return ll1 == ll2


############### calculate vp for orthogonal directions ##############

for i in range(cluster.shape[0]):
    vp_res = vp_generation(cluster[i])
    vp.append(vp_res)


vp = np.array(vp)
camera_generation(vp)

for i in range(vp.shape[0]):
	line_normal = np.dot(np.dot(RI, KI), np.append(vp[i], np.array([1])))
	line_normal = line_normal / np.sqrt(np.sum(np.square(line_normal)))
	lines_normal[cluster[i], :] = line_normal
	lines_check[cluster[i]] = 1

while not all(lines_check) or not all(faces_check):
	for face in range(faces.shape[0]):
		if faces_check[face] == 0:
			clustered_lines = []
			for j in faces[face]:
				if j in np.argwhere(lines_check == 1).flatten():
					clustered_lines.append(j)

			if len(clustered_lines) >= 2:
				for k in range(len(clustered_lines)):
					if not if_same_cluster(clustered_lines[0], clustered_lines[k]):
						## need to ensure whether this line really belongs to the face
						faces_normal[face] = np.cross(lines_normal[clustered_lines[0]], lines_normal[clustered_lines[k]])
						faces_check[face] = 1
						break

	for line in range(lines.shape[0]):
		if lines_check[line] == 0:
			for face in range(faces.shape[0]):
				if line in faces[face] and faces_check[face] == 1:
					pi_normal = plane_pi_normal(lines[line])
					line_normal = np.cross(pi_normal, faces_normal[face])
					line_normal = line_normal / np.sqrt(np.sum(np.square(line_normal)))

					lines_normal[line] = line_normal
					lines_check[line] = 1

					new_cluster = find_cluster(line_normal, line)
					new_cluster = np.array(new_cluster)
					
					tmp = list(cluster)
					tmp.append(new_cluster)
					cluster = np.array(tmp)
					

	print(lines_check)
	print(faces_check)

with open("../data/" + dist + "/lines_dir.pkl", "wb") as f:
    pickle.dump(lines_normal, f)
    # print("lines_normal: {}".format(lines_normal))

with open("../data/" + dist + "/faces_normal.pkl", "wb") as f:
    pickle.dump(faces_normal, f)
    # print("faces_normal: {}".format(faces_normal))

with open("../data/" + dist + "/parameter.pkl", "wb") as f:
    pickle.dump([K, R, F], f)

