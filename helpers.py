import numpy as np


def distance(p1, p2):
    """
    Args:
        p1: p2: points in the image plane
    Returns:
        The distance between the points
    """
    dist = np.asarray(p1) - np.asarray(p2)
    return np.sqrt(np.dot(dist, dist))


def rad(center, p1, p2, p3, p4):
    """
    Gets a center of a circle and 4 points on the circle and returns the circle radius
    Args:
        p1:p2 - p3 - p4: points on the circle in the image plane
        center: Center point of the circle
    Returns:
        The distance between the points
    """
    return (distance(center, p1) + distance(center, p2) + distance(center, p3) + distance(center, p4)) / 4
