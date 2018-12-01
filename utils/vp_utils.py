"""
This file keeps Huayi's manually collection of code pieces from Qinjie and Adam's VP estimation code
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
from skimage import feature, color, transform, io
from IPython import embed

def compute_edgelets_from_anno(vl):
    vl = np.array(vl).astype(np.int32)
    lines = vl[:, :4].reshape(vl.shape[0], 2, 2)
    locations = 0.5 * (lines[:, 0, :] + lines[:, 1, :])
    directions = lines[:, 1, :] - lines[:, 0, :]
    strengths = np.linalg.norm(directions, axis=1)
    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    return (locations, directions, strengths)

def get_vanishing_points_and_inliers_lsd_anno_pa(edgelets, threshold_inlier):
    vp = ransac_vanishing_point(edgelets, threshold_inlier, 100, w_gt=True)
    return vp

def readline_lsd(path):
    lines = np.loadtxt(path).astype(np.int32)
    lines = lines[:, :4].reshape(lines.shape[0], 2, 2)
    return lines

def compute_edgelets_from_lsd(lsd_path):
    lines_all = readline_lsd(lsd_path)
    locations = 0.5 * (lines_all[:, 0, :] + lines_all[:, 1, :])
    directions = lines_all[:, 1, :] - lines_all[:, 0, :]
    strengths = np.linalg.norm(directions, axis=1)
    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)

def get_vanishing_points_and_inliers_lsd(edgelets, threshold_inlier, algorithm='independent', reestimate=True):
    if algorithm == 'independent':
        vp1 = ransac_vanishing_point(edgelets, threshold_inlier, 300)
        print(vp1)
        if reestimate:
            vp1 = reestimate_model(vp1, edgelets, threshold_inlier)
        embed()
        edgelets2, inliers1 = remove_inliers(vp1, edgelets, threshold_inlier)
        # Find second vanishing point
        vp2 = ransac_vanishing_point(edgelets2, threshold_inlier, 300)
        if reestimate:
            vp2 = reestimate_model(vp2, edgelets2, threshold_inlier)

        edgelets3, inliers2 = remove_inliers(vp2, edgelets2, threshold_inlier)
        # Find second vanishing point
        vp3 = ransac_vanishing_point(edgelets3, threshold_inlier, 300)
        if reestimate:
            vp3 = reestimate_model(vp3, edgelets3, threshold_inlier)

    return vp1, vp2, vp3, edgelets2, edgelets3

def lines_to_edgelet(lines_all):
    locations = 0.5 * (lines_all[:, 0, :] + lines_all[:, 1, :])
    directions = lines_all[:, 1, :] - lines_all[:, 0, :]
    strengths = np.linalg.norm(directions, axis=1)
    directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    return (locations, directions, strengths)

def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.

    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

def compute_votes(edgelets, model, threshold_inlier):
    """Compute votes for each of the edgelet against a given vanishing point.

    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.

    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.

    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.

    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths

def ransac_vanishing_point(edgelets, threshold_inlier, num_ransac_iter, w_gt=False):
    """Estimate vanishing point using Ransac. Modified from ransac_vanishing_point

    Parameters
    ----------
    lines_all:
        N x 2 x 2 array
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.

    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.

    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)
    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        # print(ransac_iter)
        if w_gt:
            ind1 = np.random.choice(range(num_pts))
            ind2 = np.random.choice(range(num_pts))
        else:
            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)
        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)
        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            # print("Current best model has {} votes at iteration {}".format(
            #     current_votes.sum(), ransac_iter))

    return best_model

def reestimate_model(model, edgelets, threshold_reestimate):
    """Reestimate vanishing point using inliers and least squares.

    All the edgelets which are within a threshold are used to reestimate model

    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.

    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
    """
    locations, directions, strengths = edgelets

    inliers = compute_votes(edgelets, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    best_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((best_model, [1.]))

def remove_inliers(model, edgelets, threshold_inlier):
    """Remove all inlier edglets of a given model.

    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    lines_all:
        N x 2 x 2 array

    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """

    inliers = compute_votes(edgelets, model, threshold_inlier) > 0
    locations, directions, strengths = edgelets
    inlier_edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    remaining_edgelets = (locations[~inliers], directions[~inliers], strengths[~inliers])
    return remaining_edgelets, inlier_edgelets

def vis_edgelets(plt, edgelets, show=True):
    """Helper function to visualize edgelets."""
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]
        plt.plot(xax, yax, 'm', linewidth=3)

    if show:
        plt.show()

def vis_model(plt, plot_thred, model, edgelets, threshold, show=True, linecolor='b-', plot_num=35, linewidth=2, extend=0.1):
    """Helper function to visualize computed model."""
    locations, directions, strengths = edgelets
    # vis_edgelets(plt, edgelets, False)
    inliers = compute_votes(edgelets, model, threshold) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vp = model / model[2]
    # plt.plot(vp[0], vp[1], 'bo')
            
    for i in range(np.min([locations.shape[0], plot_num])):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        length_sqr = (locations[i, 0] - vp[0])**2 + (locations[i, 1] - vp[1])**2
        if length_sqr > plot_thred :
            ratio = np.sqrt(float(plot_thred) / length_sqr)
            xax = [locations[i, 0], (vp[0] - locations[i, 0]) * ratio * extend + locations[i, 0]]
            yax = [locations[i, 1], (vp[1] - locations[i, 1]) * ratio * extend + locations[i, 1]]
        plt.plot(xax, yax, linecolor, linewidth)
    if show:
        plt.show()

def homogenous_to_cartesian(point, eps=1e-6):
    if isinstance(point, np.ndarray):
        return np.array([point[0] / point[2], point[1] / point[2]])

    return [point[0] / point[2], point[1] / point[2]]

def is_parallel(l1, l2, eps=1e-6):
    return (l1[0] / (l1[1] + eps)) == (l2[0] / (l2[1] + eps))

def normalize_homogenous_line(line):
    norm_factor = np.sqrt(np.square(line[0]) + np.square(line[1]))

    if norm_factor == 0.0:
        return np.array([0, 0, 1])

    return line / norm_factor


def calculate_angle_between_lines(line1, line2):
    line1 = normalize_homogenous_line(np.array(line1))
    line2 = normalize_homogenous_line(np.array(line2))
    angle_in_rads = np.arccos(np.clip((line1[0] * line2[0]) + (line1[1] * line2[1]), -1.0, 1.0))
    angle = np.rad2deg(angle_in_rads)
    return angle if angle <= 90.0 else 180 - angle

def cartesian_to_homogenous(point):
    if isinstance(point, np.ndarray):
        return np.array([point[0], point[1], 1])

    return point + [1]

if __name__ == '__main__':
    edgelets = (np.array([[0, 1], [2, 1]]), np.array([[1, 1], [0, 3]]), np.array([1, 3]))
    lines = edgelet_lines(edgelets)
    print(lines)