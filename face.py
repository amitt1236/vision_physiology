import numpy as np
import cv2

'''
Creating a frame that containing only the skin portion of the head 
This frame is passed to the POS algorithm to calculate the heartrate from the image data
'''
# points for polygons for eyes socket, eyebrows, and face
eye_num_l = [257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258]
eye_num_r = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
eyebrow_l = [124, 46, 53, 52, 65, 55, 107, 66, 105, 63, 70, 156]
eyebrow_r = [285, 336, 296, 334, 293, 300, 383, 353, 276, 283, 282, 295]

face = [97, 206, 207, 192, 215, 177, 137, 227, 34, 139, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 301, 368, 264,
        447, 366, 401, 433, 416, 434, 436, 426, 327, 326, 2]


# function to create a clean frame that containing only the skin portion of the head
def clean_face(frame, points):
    eye_cnt_l = np.array([[points[i]] for i in eye_num_l], dtype=np.int32)
    eye_cnt_r = np.array([[points[i]] for i in eye_num_r], dtype=np.int32)
    eyebrow_cnt_l = np.array([[points[i]] for i in eyebrow_l], dtype=np.int32)
    eyebrow_cnt_r = np.array([[points[i]] for i in eyebrow_r], dtype=np.int32)
    face_cnt = np.array([[points[i]] for i in face], dtype=np.int32)

    zeros = np.zeros(frame.shape).astype(frame.dtype)
    out = frame.copy()
    cv2.fillPoly(zeros, [face_cnt], (255, 255, 255))
    out = cv2.bitwise_and(out, zeros)
    cv2.fillPoly(out, [eye_cnt_l], (0, 0, 0))
    cv2.fillPoly(out, [eye_cnt_r], (0, 0, 0))
    cv2.fillPoly(out, [eyebrow_cnt_l], (0, 0, 0))
    cv2.fillPoly(out, [eyebrow_cnt_r], (0, 0, 0))
    return out
