from helpers import distance

'''
input: the current frame and face landmarks
output: The distance between the lips normelized by the horizontal length of the nose
thus the distance of the subject from the camera doesnt intefires with the result
'''
def speaks(frame, points):
    t1 = points.landmark[37]
    b1 = points.landmark[84]
    t2 = points.landmark[0]
    b2 = points.landmark[17]
    t3 = points.landmark[267]
    b3 = points.landmark[314]

    normal1 = points.landmark[4]
    normal2 = points.landmark[6]

    return (distance(t1,b1,frame.shape) + distance(t2,b2,frame.shape) + distance(t3, b3, frame.shape)) / distance(normal1,normal2, frame.shape)


