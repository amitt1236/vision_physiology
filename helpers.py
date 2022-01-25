import numpy as np


def distance(p1, p2):
    dist = np.asarray(p1) - np.asarray(p2)
    return np.sqrt(np.dot(dist, dist))


def rad(center, p1, p2, p3, p4):
    return (distance(center, p1) + distance(center, p2) + distance(center, p3) + distance(center, p4)) / 4
