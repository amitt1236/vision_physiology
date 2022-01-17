import numpy as np
import cv2
from helpers import distance, rad, relative

'''
This class implements eyes crop from raw frame, and other methods to get data about the condition
all this data will be used for gaze estimation
'''


class Pupil:
    IRIS_DIAMETER = 11.7  # iris diameter of the human eye remains roughly constant at 11.7±0.5 mm

    def __init__(self, frame, points):
        self.points = points
        self.frame = frame
        self.shape = frame.shape

    '''
    input: face landmarks and raw frame. 
    return:  an image of the left eye after procesed throu a CLAHE filter.
    '''

    def left_eye(self):
        # Left eye
        l_p = self.points.landmark[473]  # pupil
        l1 = self.points.landmark[474]
        l2 = self.points.landmark[475]
        l3 = self.points.landmark[476]
        l4 = self.points.landmark[477]

        left_radius = rad(l_p, l1, l2, l3, l4, self.shape)  # raduis of the retina
        left_point = relative(l_p, self.shape)  # pupil location
        pup_left = self.frame[int(left_point[1] - left_radius): int(left_point[1] + left_radius),
                   int(left_point[0] - left_radius): int(left_point[0] + left_radius)]

        return CLAHE(pup_left)

    '''
    input: face landmarks and raw frame. 
    return:  an image of the right eye after procesed throu a CLAHE filter.
    '''

    def right_eye(self):
        # Right eye
        r_p = self.points.landmark[468]  # Pupil
        r1 = self.points.landmark[469]
        r2 = self.points.landmark[470]
        r3 = self.points.landmark[471]
        r4 = self.points.landmark[472]

        right_radius = rad(r_p, r1, r2, r3, r4, self.shape)  # raduis of the retina
        right_point = relative(r_p, self.shape)  # pupil location
        pup_right = self.frame[int(right_point[1] - right_radius): int(right_point[1] + right_radius),
                    int(right_point[0] - right_radius): int(right_point[0] + right_radius)]

        return CLAHE(pup_right)

    '''
    input: face landmarks and raw frame 
    output: mean between the two eyes of the following ratio: 1/2 as:
    1: vertical length (in pixels) of the distance between the eyes edges
    2: distance between the bottom side of the eyelid to the bottom part of the eye 
    '''

    def eyes_close(self):
        upper_left = self.points.landmark[386]
        lower_left = self.points.landmark[374]
        vertical_left1 = self.points.landmark[362]
        vertical_left2 = self.points.landmark[263]

        upper_right = self.points.landmark[159]
        lower_right = self.points.landmark[145]
        vertical_right1 = self.points.landmark[33]
        vertical_right2 = self.points.landmark[133]

        # Ratio average, every eye distance was multiplied by 100 for easier represntation 
        return (distance(upper_left, lower_left, self.shape) / distance(vertical_left1, vertical_left2,
                                                                        self.shape) * 100 \
                + distance(upper_right, lower_right, self.shape) / distance(vertical_right1, vertical_right2,
                                                                            self.shape) * 100) / 2

    '''
    input: face landmarks and raw frame 
    output: the location of the retina in the eye socket. (left-right,up-down) values are normalized [-1,1]
    note: currently horizontal detection not working well, because the eyelid interferes with the camera view.
    consider different camera location.
    '''

    def location(self):
        vert_right1 = self.points.landmark[33]
        vert_right2 = self.points.landmark[133]
        vert_left1 = self.points.landmark[362]
        vert_left2 = self.points.landmark[263]
        vert_left = distance(vert_left1, vert_left2, self.shape)  # vertical length of the left eye socket
        vert_right = distance(vert_right1, vert_right2, self.shape)  # vertical length of the right eye socket

        horz_right1 = self.points.landmark[23]
        horz_right2 = self.points.landmark[27]
        horz_left1 = self.points.landmark[253]
        horz_left2 = self.points.landmark[257]
        horz_left = distance(horz_left1, horz_left2, self.shape)  # horizontal length of the left eye socket
        horz_right = distance(horz_right1, horz_right2, self.shape)  # horizontal length of the right eye socket

        # center of the right eye socket
        center_right = ((relative(vert_right1, self.shape)[0] + relative(vert_right2, self.shape)[0]) // 2,
                        (relative(vert_right1, self.shape)[1] + relative(vert_right2, self.shape)[1]) // 2)

        # center of the left eye socket
        center_left = ((relative(vert_left1, self.shape)[0] + relative(vert_left2, self.shape)[0]) // 2,
                       (relative(vert_left1, self.shape)[1] + relative(vert_left2, self.shape)[1]) // 2)

        pup_right = relative(self.points.landmark[468], self.shape)  # left pupil location
        pup_left = relative(self.points.landmark[473], self.shape)  # right pupil location

        # position in the eye socket, normelized by the vertical and horizontal length divided by 3
        # divition by three is needed because the pupil cant reach to the edge of the eye socket
        left_horz_shift = (pup_left[1] - center_left[1]) / (horz_left / 3)
        right_horz_shift = (pup_right[1] - center_right[1]) / (horz_right / 3)
        left_vert_shift = (pup_left[0] - center_left[0]) / (vert_left / 3)
        right_vert_shift = (pup_right[0] - center_right[0]) / (vert_right / 3)

        return ((left_vert_shift + right_vert_shift) / 2,
                (left_horz_shift + right_horz_shift) / 2)  # (right to left. up to down) (-1, 1)


'''
CLAHE filter, Contrast Limited Adaptive Histogram Equalization
used to clean reflection from the frame of the eyes 
params:
clipLimit: This parameter sets the threshold for contrast limiting. The default value is 40. 
tileGridSize: This sets the number of tiles in the row and column. 
It is used while the image is divided into tiles for applying CLAHE. 
'''


def CLAHE(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE()
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
