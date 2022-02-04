from helpers import distance


def speaks(points):
    """
    input: the current frame and face landmarks
    output: The distance between the lips normalized by the horizontal length of the nose
    thus the distance of the subject from the camera doesnt interferes with the result
    """
    t1 = points[37]
    b1 = points[84]
    t2 = points[0]
    b2 = points[17]
    t3 = points[267]
    b3 = points[314]

    normal1 = points[4]
    normal2 = points[6]

    return (distance(t1, b1) + distance(t2, b2) + distance(t3, b3)) / distance(normal1, normal2)
