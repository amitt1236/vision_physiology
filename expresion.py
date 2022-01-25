
import math
from helpers import relative
from typing import List, Mapping, Optional, Tuple, Union

def centered_frame(frame, points):
    nose_tip = relative(points.landmark[4], frame.shape)
    p1 = relative(points.landmark[234], frame.shape) # left 
    p2 = relative(points.landmark[454], frame.shape) # right 
    p3 = relative(points.landmark[10], frame.shape) # top
    p4 = relative(points.landmark[152], frame.shape) # bootom
    
    max_vert = math.max(p1[0], p2[0], p3[0], p4[0])
    min_vert = math.min(p1[0], p2[0], p3[0], p4[0])
    max_horz = math.max(p1[1], p2[1], p3[1], p4[1])
    min_horz = math.min(p1[1], p2[1], p3[1], p4[1])

    frame_vert = max(abs(nose_tip[0] - min_vert), abs(nose_tip[0] - max_vert))
    frame_horz = max(abs(nose_tip[1] - min_horz), abs(nose_tip[1] - max_horz))

    frame = frame[nose_tip[0] - frame_vert: nose_tip[0] + frame_vert ,
    nose_tip[1] - frame_horz: nose_tip[1] + frame_horz]
