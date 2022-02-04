
import subprocess
from datetime import datetime

'''
##To get device list##
ffmpeg -f avfoundation -list_devices true -i ""

##Recording input##
ffmpeg -f avfoundation -i "<screen device index>:<audio device index>" output.mkv
ffmpeg -f avfoundation -framerate 30 -i "1:0" output.mkv
'''


class Recorder:
    """
    Records camera and microphone input using ffmpeg, creating a new subprocess for the task
    """
    camera_index = 0
    mic_index = 0

    def start_recording(self):
        self.R = subprocess.Popen(
            ["ffmpeg", "-f", "avfoundation", "-framerate", "25", "-i", str(self.camera_index) + ":" + str(self.mic_index) 
            , str(datetime.now()) + ".mkv"])
    
    def stop_recoeding(self):
        self.R.terminate()
        self.R.kill()
