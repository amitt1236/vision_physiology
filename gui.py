import math
from PyQt5 import QtGui
from PyQt5.QtWidgets import QGridLayout, QPushButton, QWidget, QApplication, QLabel
from PyQt5.QtGui import QPixmap, QFont
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import mediapipe as mp
import face
import pupil
from pulse import *
import record
import speaker
import gaze

cam = 0

# OpenCv thread
# noinspection PyUnresolvedReferences
class VideoThread(QThread):
    raw_frame_signal = pyqtSignal(np.ndarray)
    clean_face_signal = pyqtSignal(np.ndarray)
    left_pupil_signal = pyqtSignal(np.ndarray)
    right_pupil_signal = pyqtSignal(np.ndarray)
    count_blink_signal = pyqtSignal(int)
    bpm_signal = pyqtSignal(int)
    isSpeaking_signal = pyqtSignal(bool)

    '''
    Stream
    '''
    _run_flag = True      # flag for stream loop control
    fps = 29.79           # camera frames per second
    frame_count = 0       # current frame counter
    # FOCAL_LENGTH = 3.7  # logitec stream cam focal length is 3.7 mm

    '''
    Blink 
    '''
    blink_flag = True        # blinking flag, true if eyes are closed
    blink_counter = 0        # counts the number of blinks from the beginning of the stream
    BLINK_THRESHOLD = 10     # threshold to consider a blink
    '''
    pulse
    '''
    signal = []                    # contains signal frames for pulse calculation
    RGB_LOW_TH = np.int32(55)      # RGB values lower bound for the POS algorithm to consider
    RGB_HIGH_TH = np.int32(200)    # RGB lower bound
    WIN_SIZE = 6                   # (window size * fps) = number of frames for each processing phase
    SIG_STRIDE = 1                 # how many frames to skip between each calculation
    sig_buff_counter = 0           # counter for current stride phase
    BPM_obj = None                 # initialize an object for POS algorithm
    bpm_buffer = []                # stores the latest bpm data in order to calculate the mean and reduce noise
    bpm_mean_window = 10           # number of data point to calculate the mean

    '''
    speaker
    '''
    SPEAKER_SENSITIVITY = 100                # higher values equals to less sensitive
    win_len = 30                            # length of the window to calculate the variance
    sum_distance = np.zeros((win_len, 1))   # holds the distance between the lips for each frame
    speaker_min = math.inf                  # minimum distance between the lips

    '''
    points
    '''
    points_arr = np.zeros((478,2))


    def run(self):
        mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
        # camera stream:
        cap = cv2.VideoCapture(cam)
        # fps = cap.get(cv2.CAP_PROP_FPS) 
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,  # number of faces to track in each frame
                refine_landmarks=True,  # includes iris landmark in the face mesh model
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            while self._run_flag:
                success, image = cap.read()
                if not success:  # no frame input
                    print("Ignoring empty camera frame.")
                    continue
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
                results = face_mesh.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                shape = np.array([[image.shape[1], 0],[0, image.shape[0]]])

                if results.multi_face_landmarks:
                    self.frame_count = self.frame_count + 1

                    for i in range(478):
                        self.points_arr[i][0] = results.multi_face_landmarks[0].landmark[i].x
                        self.points_arr[i][1] = results.multi_face_landmarks[0].landmark[i].y
                        points_relative = self.points_arr @ shape

                    # eyes detection
                    pup = pupil.Pupil(image, points_relative)
                    pup_left = pup.left_eye()
                    pup_right = pup.right_eye()
                    self.left_pupil_signal.emit(pup_left)  # pass left eye frame to gui
                    self.right_pupil_signal.emit(pup_right)  # pass right eye frame to gui

                    # blink detection
                    if not (pup.eyes_close() < self.BLINK_THRESHOLD):
                        self.blink_flag = True
                    if (pup.eyes_close() < self.BLINK_THRESHOLD) and self.blink_flag:
                        self.blink_flag = False
                        self.blink_counter = self.blink_counter + 1
                    self.count_blink_signal.emit(self.blink_counter)  # pass blink counter

                    # returns the skin portion of the face, without eyes, mouth, etc.
                    clean = face.clean_face(image, points_relative)
                    self.clean_face_signal.emit(clean)

                    # gaze pose estimation    
                    gaze.gaze(image, points_relative)
                    self.raw_frame_signal.emit(image)  # pass main frame with head pose estimation to gui

                    '''
                    speaker detection:
                    determining if the recognized person is speaking.
                    calculating the variance of the distance between the lips over predefined window of frames
                    '''
                    self.sum_distance[self.frame_count % self.win_len] = speaker.speaks(points_relative)
                    var_cur = np.var(self.sum_distance, 0)
                    self.speaker_min = min(self.speaker_min, var_cur)
                    
                    if var_cur >= self.SPEAKER_SENSITIVITY * self.speaker_min:
                        self.isSpeaking_signal.emit(True)
                    else:
                        self.isSpeaking_signal.emit(False)

                    '''
                    pulse detection:
                    determining the recognized person pulse from image data using POS algorithm
                    '''
                    tmp = rgb_mean(clean, self.RGB_HIGH_TH, self.RGB_HIGH_TH)
                    self.signal.append(tmp)
                    if self.frame_count > int(self.fps * self.WIN_SIZE):  # initial window size
                        self.signal = self.signal[1:]
                        if self.sig_buff_counter == 0:
                            self.sig_buff_counter = self.SIG_STRIDE
                            copy_sig = np.array(self.signal, dtype=np.float32)
                            copy_sig = np.swapaxes(copy_sig, 0, 1)
                            copy_sig = np.swapaxes(copy_sig, 1, 2)
                            # pre filter
                            copy_sig = BPfilter(copy_sig, self.fps)

                            # BVP
                            bvp = cpu_POS(copy_sig, self.fps)

                            # Post filter
                            bvp = np.expand_dims(bvp, axis=1)
                            bvp = BPfilter(bvp, self.fps)
                            bvp = np.squeeze(bvp, axis=1)

                            # BPM
                            if self.BPM_obj is None:
                                self.BPM_obj = BPM(bvp, self.fps, minHz=0.7, maxHz=3.0)
                            else:
                                self.BPM_obj.bvp_sig = bvp
                            bpm = self.BPM_obj.BVP_to_BPM()
                            # mean calculation
                            if len(self.bpm_buffer) >= self.bpm_mean_window:
                                self.bpm_buffer.pop(len(self.bpm_buffer) - 1)
                            self.bpm_buffer.append(bpm)
                            mean = int(sum(self.bpm_buffer) / len(self.bpm_buffer))

                            self.bpm_signal.emit(mean)
                        else:
                            self.sig_buff_counter = self.sig_buff_counter - 1

    # when window is close, close the stream
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


# noinspection PyUnresolvedReferences
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("face detection")
        self.display_width = 640
        self.display_height = 480
        self.display_eye_size = 310
        # create the label that holds the raw image
        self.raw_label = QLabel(self)
        self.raw_label.resize(self.display_width, self.display_height)

        # create the label that holds the clean image
        self.clean_label = QLabel(self)
        self.clean_label.resize(self.display_width, self.display_height)

        # create the label that holds the left pupil
        self.left_pupil_label = QLabel(self)
        self.left_pupil_label.resize(self.display_eye_size, self.display_eye_size)

        # create the label that holds the right pupil
        self.right_pupil_label = QLabel(self)
        self.right_pupil_label.resize(self.display_eye_size, self.display_eye_size)

        # create the label that holds the blink count
        self.count_blink = QLabel('Arial font', self)
        self.count_blink.setFont(QFont('Arial', 20))

        # create the label that holds the heartrate
        self.heart_rate = QLabel('Heart rate: 0', self)
        self.heart_rate.setFont(QFont('Arial', 20))

        # create the label that holds the recording flag
        self.recordFlag = QLabel('Recording: False', self)
        self.recordFlag.setFont(QFont('Arial', 20))

        # creates the label that holds the speaking flag
        self.speaksFlag = QLabel('Speaks: False', self)
        self.speaksFlag.setFont(QFont('Arial', 20))

        # create the recording buttons
        self.record_button = QPushButton("Start recording", self)
        self.record_button.setFixedSize(200, 150)
        self.record_button.clicked.connect(self.update_record)
        self.record_status = False
        self.rec = record.Recorder()

        # create a grid box layout
        # raw output
        grid = QGridLayout()
        grid.addWidget(self.raw_label, 0, 0, 1, 4)
        # clean face
        grid.addWidget(self.clean_label, 0, 4, 1, 4)
        # left pupil
        grid.addWidget(self.left_pupil_label, 1, 0, 4, 2, Qt.AlignLeft)
        # right pupil
        grid.addWidget(self.right_pupil_label, 1, 2, 4, 2, alignment=Qt.AlignRight)
        # blink count
        grid.addWidget(self.count_blink, 1, 4, Qt.AlignTop)
        # heart rate
        grid.addWidget(self.heart_rate, 1, 5, alignment=Qt.AlignmentFlag.AlignTop)
        # record flag
        grid.addWidget(self.recordFlag, 1, 6, alignment=Qt.AlignmentFlag.AlignTop)
        # speaking flag
        grid.addWidget(self.speaksFlag, 1, 7, alignment=Qt.AlignmentFlag.AlignTop)
        # recording buttons
        grid.addWidget(self.record_button, 2, 7, 2, 1, alignment=Qt.AlignmentFlag.AlignTop)

        self.setLayout(grid)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to update values
        self.thread.raw_frame_signal.connect(self.update_raw)
        self.thread.clean_face_signal.connect(self.update_clean)
        self.thread.left_pupil_signal.connect(self.update_left_pupil)
        self.thread.right_pupil_signal.connect(self.update_right_pupil)
        self.thread.count_blink_signal.connect(self.update_blink)
        self.thread.bpm_signal.connect(self.update_heart_rate)
        self.thread.isSpeaking_signal.connect(self.update_speaks)
        self.thread.start()

    def closeEvent(self, event):
        if self.record_status:
            self.rec.stop_recoeding()
        self.thread.stop()
        event.accept()

    # Updates label
    @pyqtSlot(bool)
    def update_speaks(self, flag):
        self.speaksFlag.setText("Speaks: " + str(flag))

    @pyqtSlot(bool)
    def update_record(self):
        if not self.record_status:  # if not recording
            self.record_status = True
            self.rec.start_recording()
            self.record_button.setText("Stop recording")
        else:
            self.record_status = False
            self.rec.stop_recoeding()
            self.record_button.setText("Start recording")

    @pyqtSlot(np.ndarray)
    def update_raw(self, cv_img):
        """Updates the main frame"""
        qt_img = self.convert_cv_qt(cv_img)
        self.raw_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_clean(self, cv_img):
        """Updates the clean face frame"""
        qt_img = self.convert_cv_qt(cv_img)
        self.clean_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_right_pupil(self, cv_img):
        """Updates the right pupil frame"""
        qt_img = self.convert_cv_qt2(cv_img)
        self.right_pupil_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_left_pupil(self, cv_img):
        """Updates the left pupil frame"""
        qt_img = self.convert_cv_qt2(cv_img)
        self.left_pupil_label.setPixmap(qt_img)

    @pyqtSlot(int)
    def update_blink(self, val):
        """Updates the eye blink counter"""
        self.count_blink.setText("Blink count: " + str(val))

    @pyqtSlot(int)
    def update_heart_rate(self, val):
        """Updates the heart rate measure"""
        self.heart_rate.setText("Heart rate: " + str(val))

    # processing full frame from opencv
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # processing eyes frame from opencv
    def convert_cv_qt2(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.display_eye_size, self.display_eye_size, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())