import numpy as np
from numpy import *
import pickle
from IPython import embed
import sys

sys.setrecursionlimit(1000000)
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

KI = np.mat(K).I
RI = np.mat(Rt[:3, :3]).I
t = np.array(Rt[:3, 3]).flatten()

#################### can write this part into config.py later ###########

with open("../data/faces_normal.pkl", "rb") as f:
	normal = pickle.load(f)

with open("../data/faces.pkl", "rb") as f:
    face = pickle.load(f)

with open("../data/lines_set.pkl", "rb") as f:
	points = pickle.load(f)

with open("../data/face_points.pkl", "rb") as f:
	fp = pickle.load(f)


#################### clustering the pixels in faces and lines #################
##### ATTENTION: the strcture for fp is different with others, where x in 1st column and y in 2nd column
fc = np.zeros((H, W)) - 1
lc = np.zeros((H, W)) - 1
label = np.argwhere(fc == -1)

threshold = 10 ## judge whether the point is in the line
points_set = np.zeros((H, W, 3))
points_mask = np.zeros((H, W))
y1 = 0
y2 = 0
x1 = 0
x2 = 0


####### points in faces ############################
####### ATTENTION: all the points are in the sequence of inverse time circle
for i in range(fp.shape[0]):
	mask = np.zeros(label.shape[0])
	for j in range(fp[i].shape[0]):
		
		x1 = fp[i][j - 1][0]
		y1 = fp[i][j - 1][1]
		x2 = fp[i][j][0]
		y2 = fp[i][j][1]

		tmp = (x2 - x1) * (label[:, 0] - y1) - (y2 - y1) * (label[:, 1] - x1)
		mask[np.where(tmp.flatten() < 0)] += 1
	    ## less than 0 owing to the reverse of the co-ordinates
	index = np.argwhere(mask == fp[i].shape[0])
	fc[tuple(label[index[:, 0]].T)] = i

######### points in lines ############################

for i in range(points.shape[0]):
	x1 = points[i][0][0]
	y1 = points[i][0][1]
	x2 = points[i][1][0]
	y2 = points[i][1][1]

	tmp = (x2 - x1) * (label[:, 0] - y1) - (y2 - y1) * (label[:, 1] - x1)
	index = np.argwhere(np.abs(tmp) < threshold)
	tmp_label = label[index[:, 0]].copy()

	check = 1
	while check == 1:
		check = 0
		for j in range(tmp_label.shape[0]):
			if not ((tmp_label[j][0] > min(points[i][0][1], points[i][1][1]) and tmp_label[j][0] < max(points[i][0][1], points[i][1][1]))):
				check = 1
				tmp_label = np.delete(tmp_label, (j * 2, j * 2 + 1)).reshape(-1, 2)
				break
	
	lc[tuple(tmp_label.T)] = i


###########  reconstruction setting  ###################
z_ori = -200 # original z for projection
x_pro = 312
y_pro = 269

ori_point = z_ori * np.dot(RI[:3, :3], (np.array([(x_pro - W/2)/F, (y_pro - H/2)/F, 1]))) - np.dot(RI, t)
points_set[y_pro, x_pro, :] = ori_point[0]
print(points_set[y_pro, x_pro, :])
points_mask[y_pro, x_pro] = 1
normal_mask = np.ones(face.shape[0])

########### reconstruction ##############################

def reconstruct(y_pro, x_pro, h, w, nor):
	# embed()
	y_pro = int(y_pro)
	x_pro = int(x_pro)
	h = int(h)
	w = int(w)
	nor = int(nor)

	vec = points_set[y_pro, x_pro, :]

	y = y_pro + h
	x = x_pro + w

	if points_mask[y, x] != 0:
		return (-1, -1, -1, -1)

	cur_vec = np.array(np.dot(RI[:3, :3], np.array([(x - W/2)/F, (y - H/2)/F, 1]))).flatten()


	if fc[y, x] == -1 and lc[y, x] == -1:
		return (-1, -1, -1, -1)

	#### face to face
	elif fc[y, x] == nor:
		cur_normal = np.array(normal)[np.uint(fc[y, x])].flatten()
		depth = (np.dot(vec, cur_normal) + np.dot(np.dot(RI, t), cur_normal)) / np.dot(cur_vec, cur_normal)
		# embed()
		cur_vec = cur_vec * np.array(depth).flatten() - np.array(np.dot(RI, t)).flatten()

		points_set[y, x, :] = cur_vec
		# if abs(points_set[y, x, 2]) > 800:
			# embed()
		points_mask[y, x] = 1
		print(("x: {} y:{} z:{}, nor:{}").format(points_set[y, x, 0], points_set[y, x, 1], points_set[y, x, 2], np.uint(fc[y, x])))
		return (y, x, 1, -1)

    #### face to line
	elif lc[y, x] != -1:
		
		depth = (points_set[y_pro, x_pro, 2] + np.array(np.dot(RI, t)).flatten()[2]) / cur_vec[2]
		points_set[y, x, :] = cur_vec * depth - np.array(np.dot(RI, t)).flatten()
		print("lc")
		print(("x: {} y:{} z:{}").format(points_set[y, x, 0], points_set[y, x, 1], points_set[y, x, 2]))
		points_mask[y, x] = 1

		return (y, x, 1, -1)

    #### line to face
	elif fc[y, x] != -1 and normal_mask[np.uint(fc[y, x])] == 1:
				
		depth = (points_set[y_pro, x_pro, 2] + np.array(np.dot(RI, t)).flatten()[2]) / cur_vec[2]
		points_set[y, x, :] = cur_vec * depth - np.array(np.dot(RI, t)).flatten()
		points_mask[y, x] = 1
		print("fc")
		print(("x: {} y:{} z:{}").format(points_set[y, x, 0], points_set[y, x, 1], points_set[y, x, 2]))
		next = np.uint(fc[y, x])
		normal_mask[np.uint(fc[y, x])] = 0
		return (y, x, 1, next)

	else:
		return (-1, -1, -1, -1)


############  reconstructing  ##########################
cur_nor = np.array([[y_pro, x_pro, fc[y_pro, x_pro]]])

queue = np.array([[y_pro, x_pro, -1, 0, cur_nor[0, 2]]])
queue = np.concatenate((queue, np.array([[y_pro, x_pro, 1, 0, cur_nor[0, 2]]])), axis=0)
queue = np.concatenate((queue, np.array([[y_pro, x_pro, 0, -1, cur_nor[0, 2]]])), axis=0)
queue = np.concatenate((queue, np.array([[y_pro, x_pro, 0, 1, cur_nor[0, 2]]])), axis=0)

normal_mask[np.uint(fc[y_pro, x_pro])] = 0

loop_iter = 0
while cur_nor.shape[0] != 0:
	while queue.shape[0] != 0:
		(y, x, check, line_index) = reconstruct(queue[0, 0], queue[0, 1], queue[0, 2], queue[0, 3], queue[0, 4])
		# embed()
		if line_index != -1:
			cur_nor = np.concatenate((cur_nor, np.array([[y, x, line_index]])), axis=0)

		if check == 1:
			queue = np.concatenate((queue, np.array([[y, x, -1, 0, cur_nor[0, 2]]])), axis=0)
			queue = np.concatenate((queue, np.array([[y, x, 1, 0, cur_nor[0, 2]]])), axis=0)
			queue = np.concatenate((queue, np.array([[y, x, 0, -1, cur_nor[0, 2]]])), axis=0)
			queue = np.concatenate((queue, np.array([[y, x, 0, 1, cur_nor[0, 2]]])), axis=0)

		loop_iter += 1
		queue = np.delete(queue, 0, 0)
		# print(loop_iter)

	cur_nor = np.delete(cur_nor, 0, 0)
	# embed()
	if cur_nor.shape[0] != 0:
		queue = np.array([[cur_nor[0, 0], cur_nor[0, 1], -1, 0, cur_nor[0, 2]]])
		queue = np.concatenate((queue, np.array([[cur_nor[0, 0], cur_nor[0, 1], 1, 0, cur_nor[0, 2]]])), axis=0)
		queue = np.concatenate((queue, np.array([[cur_nor[0, 0], cur_nor[0, 1], 0, -1, cur_nor[0, 2]]])), axis=0)
		queue = np.concatenate((queue, np.array([[cur_nor[0, 0], cur_nor[0, 1], 0, 1, cur_nor[0, 2]]])), axis=0)


with open("../data/reconstruction.pkl", "wb") as f:
	pickle.dump(points_set, f)

for i in range(points_set.shape[0]):
		for j in range(points_set.shape[1]):
			if points_set[i, j, 0] < 0 and points_set[i, j, 1] < 0 and points_set[i, j, 2] > 2:
				points_set[i, j] = -points_set[i, j]

with open("../data/reconstruction.obj","w") as f:
	iter = 0
	for i in range(points_set.shape[0]):
		for j in range(points_set.shape[1]):
			if points_set[i, j, 0] != 0 and points_set[i, j, 1] != 0:
				if abs(points_set[i, j, 0]) < 600 and abs(points_set[i, j, 1]) < 600 and abs(points_set[i, j, 2] < 600):
					iter += 1
					print(iter)
					f.write('v ' + str(points_set[i, j, 0]) + ' ' + str(points_set[i, j, 1]) + ' ' + str(points_set[i, j, 2]) + '\n')