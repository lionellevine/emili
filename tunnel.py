# real-time emotion visualizer using FER labels sourced from on-device camera

from PyQt5.QtWidgets import QApplication # GUI uses PyQt
from PyQt5.QtCore import QThread # videoplayer lives in a QThread
from gui import Visualizer, VideoPlayerWorker
#from emili_core import * # core threading logic
from paz import processors as pr
from paz.pipelines import DetectMiniXceptionFER # facial emotion recognition pipeline

import numpy as np
import sys
import argparse
from paz.backend.camera import Camera
import threading
import queue
import time
from datetime import datetime

class EmoTunnel(DetectMiniXceptionFER): # video pipeline for real-time FER visualizer
    def __init__(self, start_time, dims, offsets):
        super().__init__(offsets)
        self.start_time = start_time
        self.current_frame = None # other threads have read access
        self.frame_lock = threading.Lock()  # Protects access to current_frame
        self.display_width = dims[0]
        self.display_height = dims[1]
        self.time_series = [] # list of [time, scores] pairs
        #self.draw = pr.TunnelBoxes(self.time_series, self.colors, True) # override the default draw method

    def get_current_frame(self):
        with self.frame_lock:  # Ensure exclusive access to current_frame
            return self.current_frame

    def call(self, image, time_series=[]):
        results = super().call(image) # classify faces in the image, draw boxes and labels
        image, faces = results['image'], results['boxes2D']
        emotion_data = self.report_emotion(faces)
        if(emotion_data is not None):
            self.time_series.append([emotion_data['time'],emotion_data['scores']])
        frame_to_display = self.construct_frame(time_series)
        with self.frame_lock:  
            self.current_frame = frame_to_display # update the current frame
        return results
    
    def construct_frame(self, time_series): # todo: write!
        frame = np.zeros((self.display_width, self.display_height, 3), dtype=np.uint8)
        return frame

    def report_emotion(self, faces): # add to emotion_queue to make available to other threads
        current_time = time_since(self.start_time) # milliseconds since start of session
        num_faces = len(faces)
        if(num_faces>0):
            max_height = 0
            for k,box in enumerate(faces): # find the largest face 
                if(box.height > max_height):
                    max_height = box.height
                    argmax = k
            if(max_height>150): # don't log small faces (helps remove false positives)
                face_id = f"{argmax+1} of {num_faces}"
                box = faces[argmax] # log emotions for the largest face only. works well in a single-user setting. todo: improve for social situations! 
                emotion_data = {
                    "time": current_time,
                    "face": face_id,
                    "class": box.class_name,
                    "size": box.height,
                    "scores": (box.scores.tolist())[0]  # 7-vector of emotion scores, converted from np.array to list
                }
                gui_app.signal.fresh_scores.emit([current_time, emotion_data['scores']]) # send scores to be displayed in the GUI
                emotion_queue.put(emotion_data)
                return emotion_data
        return None # no large faces found
                #new_data_event.set()  # Tell the other threads that new data is available
                
 #   def __del__(self): # no log file, not needed
 #       self.log_file.close()  # Close the file when the instance is deleted
 #       print("Log file closed.")
    
def time_since(start_time):
    return int((time.time() - start_time) * 1000) # milliseconds since start of session

if __name__ == "__main__":

    start_time = time.time() 
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_session_event = threading.Event() # triggered when the user enters 'q' to end the session

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1, help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()
    camera = Camera(args.camera_id)

    emotion_queue = queue.Queue() # real-time emotion logs updated continuously

    window_dims = [720, 720] # width, height

    EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]

    app = QApplication(sys.argv)
    gui_app = Visualizer(start_time, window_dims, np.array(EMOTION_COLORS), emotion_queue, end_session_event)

    pipeline = EmoTunnel(start_time, window_dims, [args.offset, args.offset]) # video processing pipeline

    print(f"Real-time emotion visualizer using FER labels sourced from on-device camera.")
    print(f"Type 'q' to end the session.")

    gui_app.show() # Start the GUI

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [1280, 720] # width, height (16:9 aspect ratio)
    video_thread = QThread() # video thread: OpenCV is safe in a QThread but not a regular thread
    video_worker = VideoPlayerWorker(
        start_time,
        video_dims,
        pipeline, # applied to each frame of video
        camera)
    video_worker.moveToThread(video_thread)

    video_thread.started.connect(video_worker.run) # connect signals and slots
    video_worker.finished.connect(video_thread.quit)
    video_worker.finished.connect(video_worker.deleteLater)
    video_thread.finished.connect(video_thread.deleteLater)
    video_worker.frameReady.connect(gui_app.display_frame) # update the FER tab with new video frame

    video_thread.start()
    print("Started video thread.")
    app.exec_() # start the GUI app. This should run in the main thread. Lines after this only execute if user closes the GUI.

    print("GUI app closed by user.")
    video_thread.quit()
    VideoPlayerWorker.stop_flag = True
    print("Quitting video thread...")
    #video_thread.wait()     # hangs. todo: fix
    print("Session ended.")
