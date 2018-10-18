import numpy as np
import pickle
from IPython import embed

with open("../data/lines_set.pkl", "rb") as f:
    lines = pickle.load(f)

with open("../data/cluster.pkl", "rb") as f:
    cluster = pickle.load(f)

with open("../data/faces.pkl", "rb") as f:
    faces = pickle.load(f)

# vanishing point
vp = {}
normal = {}
vp_threshold = 50


def vp_cal(points):
    points = np.array(points)
    cluster_lines = []
    for i in points:
        k = (lines[i][0][0] - lines[i][1][0]) / (lines[i][0][1] - lines[i][1][1])
        b = lines[i][0][0] - k * lines[i][0][1]
        cluster_lines.append(np.array((k, b)))

    # use two lines to calculate vp, and return the average result
    a = np.array([[cluster_lines[0][0], -1], [cluster_lines[1][0], -1]])
    b = np.array([-cluster_lines[0][1], -cluster_lines[1][1]])
    x1 = np.linalg.solve(a, b)

    a = np.array([[cluster_lines[0][0], -1], [cluster_lines[2][0], -1]])
    b = np.array([-cluster_lines[0][1], -cluster_lines[2][1]])
    x2 = np.linalg.solve(a, b)

    return x1/2 + x2/2


if __name__ == "__main__":
    # owing that the lines marked has been clustered. Thus we create a new cluster for the cluster generation
    origin_cluster = cluster[1:4]

    # calculate known vp
    for i in range(origin_cluster.shape[0]):
        vp[i] = vp_cal(origin_cluster[i])

