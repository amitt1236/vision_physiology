import cv2
import math
import numpy as np
import mediapipe as mp
import scipy
from pulse import BPfilter, cpu_POS, rgb_mean
import pupil
import face
from pulse import *
import speaker
import gaze
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
'''
Stream
'''
fps = 29.79           # camera frames per second
frame_count = 0       # current frame counter
# FOCAL_LENGTH = 3.7  # logitec stream cam focal length is 3.7 mm

'''
Blink 
'''
blink_flag = True               # blinking flag, true if eyes are closed
blink_counter = 0               # counts the number of blinks from the beginning of the stream
BLINK_THRESHOLD = 5000          # threshold to consider a blink
'''
pulse
'''
signal = []                     # contains signal frames for pulse calculation
RGB_LOW_TH = np.int32(55)       # RGB values lower bound for the POS algorithm to consider
RGB_HIGH_TH = np.int32(200)     # RGB lower bound
WIN_SIZE = 6                    # (window size * fps) = number of frames for each processing phase
SIG_STRIDE = 1                  # how many frames to skip between each calculation
sig_buff_counter = 0            # counter for current stride phase
BPM_obj = None                  # initialize an object for POS algorithm
bpm_buffer = []                 # stores the latest bpm data in order to calculate the mean and reduce noise
bpm_mean_window = 10            # number of data point to calculate the mean

'''
speaker
'''

SPEAKER_SENSITIVITY = 30              # higher values equals to less sensitive
win_len = 60                          # length of the window to calculate the variance
sum_distance = np.zeros((win_len,1))  # holds the distance between the lips for each frame
speaker_min = math.inf                # minimum distance between the lips

'''
points
'''
points_arr = np.zeros((478,2))

'''
plot
'''
ploted = False
# to run GUI event loop
plt.ion()
# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
figure2, ax2 = plt.subplots(figsize=(10, 8))

 
# camera stream:
cap = cv2.VideoCapture(0)
# fps = cap.get(cv2.CAP_PROP_FPS)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,                            # number of faces to track in each frame
        refine_landmarks=True,                      # includes iris landmark in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:                            # no frame input
            print("Ignoring empty camera frame.")
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        shape = np.array([[image.shape[1], 0],[0, image.shape[0]]])

        if results.multi_face_landmarks:
            frame_count = frame_count + 1

            for i in range(478):
                points_arr[i][0] = results.multi_face_landmarks[0].landmark[i].x
                points_arr[i][1] = results.multi_face_landmarks[0].landmark[i].y
            points_relative = points_arr @ shape

            # eyes detection
            pup = pupil.Pupil(image, points_relative)
            pup_left = pup.left_eye()    # frame that contains the left eye
            pup_right = pup.right_eye()  # frame that contains the right eye

            # blink detection
            if not (pup.eyes_close() < BLINK_THRESHOLD):
                blink_flag = True
            if (pup.eyes_close() < BLINK_THRESHOLD) and blink_flag:
                blink_flag = False
                blink_counter = blink_counter + 1

            # returns the skin portion of the face, without eyes, mouth, etc.
            clean = face.clean_face(image, points_relative)

            # head pose estimation
            gaze.gaze(image, points_relative)

            '''
            speaker detection:
            determining if the recognized person is speaking.
            calculating the variance of the distance between the lips over predefined window of frames
            '''
            sum_distance[frame_count % win_len] = speaker.speaks(points_relative)
            var_cur = np.var(sum_distance,0)
            speaker_min = min(speaker_min, var_cur)
            if var_cur >= SPEAKER_SENSITIVITY * speaker_min:
                isSpeaking_signal = True
            else:
                isSpeaking_signal = False

            '''
            pulse detection:
            determining the recognized person pulse from image data using POS algorithm
            '''
            tmp = rgb_mean(clean, RGB_LOW_TH, RGB_HIGH_TH)
            signal.append(tmp)
            if frame_count > int(fps * WIN_SIZE):  # initial window size
                signal = signal[1:]
                if sig_buff_counter == 0:
                    sig_buff_counter = SIG_STRIDE
                    copy_sig = np.array(signal, dtype=np.float32)
                    copy_sig = np.swapaxes(copy_sig, 0, 1)
                    copy_sig = np.swapaxes(copy_sig, 1, 2)

                    # pre filter#
                    copy_sig = BPfilter(copy_sig, fps)

                    # BVP#
                    bvp = cpu_POS(copy_sig, fps)

                    # ploting BVP
                    x = [i for i in range(len(bvp[0]))]
                    if not ploted:
                        line1, = ax.plot([0, 175], [40000, -40000])
                        line2, = ax2.plot([0, 220], [5000000, -10])
                        ploted = True
                    
                    line1.set_xdata(x)
                    line1.set_ydata(bvp[0])
                    figure.canvas.draw()
                    figure.canvas.flush_events()

                    # Post filter#
                    bvp = np.expand_dims(bvp, axis=1)
                    bvp = BPfilter(bvp, fps)
                    bvp = np.squeeze(bvp, axis=1)

                    freq, power  = Welch(bvp, fps)

                    line2.set_xdata(freq)
                    line2.set_ydata(power)
                    figure2.canvas.draw()
                    figure2.canvas.flush_events()
                    
                    # BPM#
                    if BPM_obj is None:
                        BPM_obj = BPM(bvp, fps, minHz=0.7, maxHz=3.0)
                    else:
                        BPM_obj.bvp_sig = bvp
                    bpm = BPM_obj.BVP_to_BPM()
                    print(bpm)
                else:
                    sig_buff_counter = sig_buff_counter - 1

        # Flip the image horizontally for a selfie-view display.
        try:
            cv2.imshow('clean', clean)
            cv2.imshow('Left eye', cv2.resize(pup_left, (600, 600)))
            cv2.imshow('Right eye', cv2.resize(pup_right, (600, 600)))
        except:
            pass

        cv2.imshow('MediaPipe Face Mesh', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
            
cap.release()
