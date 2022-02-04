import numpy as np
import cv2
from helpers import distance, rad


class Pupil:
    """
    This class implements eyes crop from raw frame, and other methods to get data about the condition
    all this data will be used for gaze estimation
    """
    IRIS_DIAMETER = 11.7  # iris diameter of the human eye remains roughly constant at 11.7±0.5 mm

    def __init__(self, frame, points):
        self.points = points
        self.frame = frame

    def left_eye(self):
        """
        input: face landmarks and raw frame.
        return:  an image of the left eye after procesed throu a CLAHE filter.
        """
        l_p = self.points[473]  # pupil
        l1 = self.points[474]
        l2 = self.points[475]
        l3 = self.points[476]
        l4 = self.points[477]

        left_radius = rad(l_p, l1, l2, l3, l4)  # raduis of the retina

        pup_left = self.frame[int(l_p[1] - left_radius): int(l_p[1] + left_radius),
                   int(l_p[0] - left_radius): int(l_p[0] + left_radius)]

        return CLAHE(pup_left)

    def right_eye(self):
        """
        input: face landmarks and raw frame.
        return:  an image of the right eye after procesed throu a CLAHE filter.
        """
        r_p = self.points[468]  # Pupil
        r1 = self.points[469]
        r2 = self.points[470]
        r3 = self.points[471]
        r4 = self.points[472]

        right_radius = rad(r_p, r1, r2, r3, r4)  # raduis of the retina

        pup_right = self.frame[int(r_p[1] - right_radius): int(r_p[1] + right_radius),
                    int(r_p[0] - right_radius): int(r_p[0] + right_radius)]

        return CLAHE(pup_right)

    def eyes_close(self):
        """
        input: face landmarks and raw frame
        output: mean between the two eyes of the following ratio: 1/2 as:
        1: vertical length (in pixels) of the distance between the eyes edges
        2: distance between the bottom side of the eyelid to the bottom part of the eye
        """
        upper_left = self.points[386]
        lower_left = self.points[374]
        vertical_left1 = self.points[362]
        vertical_left2 = self.points[263]

        upper_right = self.points[159]
        lower_right = self.points[145]
        vertical_right1 = self.points[33]
        vertical_right2 = self.points[133]

        # Ratio average, every eye distance was multiplied by 100 for easier represntation 
        return (distance(upper_left, lower_left) / distance(vertical_left1, vertical_left2) * 100 \
                + distance(upper_right, lower_right) / distance(vertical_right1, vertical_right2, ) * 100) / 2

    def location(self):
        """
        input: face landmarks and raw frame
        output: the location of the retina in the eye socket. (left-right,up-down) values are normalized [-1,1]
        note: currently horizontal detection not working well, because the eyelid interferes with the camera view.
        consider different camera location.
        """
        vert_right1 = self.points[33]
        vert_right2 = self.points[133]
        vert_left1 = self.points[362]
        vert_left2 = self.points[263]
        vert_left = distance(vert_left1, vert_left2)  # vertical length of the left eye socket
        vert_right = distance(vert_right1, vert_right2)  # vertical length of the right eye socket

        horz_right1 = self.points[23]
        horz_right2 = self.points[27]
        horz_left1 = self.points[253]
        horz_left2 = self.points[257]
        horz_left = distance(horz_left1, horz_left2)  # horizontal length of the left eye socket
        horz_right = distance(horz_right1, horz_right2)  # horizontal length of the right eye socket

        # center of the right eye socket
        center_right = ((vert_right1[0] + vert_right2[0]) // 2,
                        (vert_right1[1] + vert_right2[1]) // 2)

        # center of the left eye socket
        center_left = ((vert_left1[0] + vert_left2[0]) // 2,
                       (vert_left1[1] + vert_left2[1]) // 2)

        pup_right = self.points[468]  # left pupil location
        pup_left = self.points[473]  # right pupil location

        # position in the eye socket, normalized by the vertical and horizontal length divided by 3
        # division by three is needed because the pupil can't reach to the edge of the eye socket
        left_horz_shift = (pup_left[1] - center_left[1]) / (horz_left / 3)
        right_horz_shift = (pup_right[1] - center_right[1]) / (horz_right / 3)
        left_vert_shift = (pup_left[0] - center_left[0]) / (vert_left / 3)
        right_vert_shift = (pup_right[0] - center_right[0]) / (vert_right / 3)

        return ((left_vert_shift + right_vert_shift) / 2,
                (left_horz_shift + right_horz_shift) / 2)  # (right to left. up to down) (-1, 1)


def CLAHE(frame):
    """
    CLAHE filter, Contrast Limited Adaptive Histogram Equalization
    used to clean reflection from the frame of the eyes
    params:
    clipLimit: This parameter sets the threshold for contrast limiting. The default value is 40.
    tileGridSize: This sets the number of tiles in the row and column.
    It is used while the image is divided into tiles for applying CLAHE.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE()
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    v = clahe.apply(v)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
