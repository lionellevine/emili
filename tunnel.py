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
from copy import deepcopy
import cProfile
import pstats

class EmoTunnel(DetectMiniXceptionFER): # video pipeline for real-time FER visualizer
    def __init__(self, start_time, dims, offsets, speed=25):
        super().__init__(offsets)
        self.start_time = start_time
        self.current_frame = None # other threads have read access
        self.frame_lock = threading.Lock()  # Protects access to current_frame
        self.display_width = dims[0]
        self.display_height = dims[1]
        self.time_series = [] # list of [time, scores] pairs
        self.binned_time_series = [] # list of [time, mean_scores] pairs
        self.current_bin = [] # list of [time, scores] pairs in the current bin
        self.speed = speed # tunnel expansion rate in pixels per second, recommend 25-50
        self.interval = 1000//speed # ms per pixel
        self.bin_end_time = self.interval # start a new bin every interval ms
        self.no_data_indicator = np.full(7,1e5) # mean scores for an empty bin
        self.last_bin_mean = np.full(7,1e5) # mean scores for the most recent bin
        #self.signal = signal
        #self.draw = pr.TunnelBoxes(self.time_series, self.colors, True) # override the default draw method

    def get_current_frame(self):
        with self.frame_lock:  # Ensure exclusive access to current_frame
            return self.current_frame

    def call(self, image):

        # binning logic: every interval ms, record the mean scores of the current bin, signal GUI to update, start a new bin
        current_time = time_since(self.start_time)
        #print(f"(pipeline.call) current_time: {current_time}")
        #print(f"(pipeline.call) bin_end_time: {self.bin_end_time}")
        if self.bin_end_time < current_time: # done with current bin
            new_bin_data = []
            if(len(self.current_bin)>0):
                #print(f"(pipeline.call) done with bin")
                #print(f"(pipeline.call) current_bin: {self.current_bin}")
                self.last_bin_mean = np.mean(self.current_bin, axis=0)
                #print(f"(pipeline.call) bin_mean: {self.last_bin_mean}")
                new_bin_data.append([self.bin_end_time, deepcopy(self.last_bin_mean)])
                self.bin_end_time += self.interval
                self.current_bin = [] # start a new bin
            while(self.bin_end_time < current_time): # catch up to the current time
                #print("(pipeline.call) catching up, empty bin")
                self.last_bin_mean = 0.9*self.last_bin_mean + 0.1*self.no_data_indicator # no new data, discount to indicate staleness
                #print(f"(pipeline.call) bin_end_time: {self.bin_end_time}")
                #print(f"(pipeline.call) last_bin_mean: {self.last_bin_mean}")
                new_bin_data.append([self.bin_end_time, deepcopy(self.last_bin_mean)]) # empty bin
#                new_bin_data.append([self.bin_end_time, np.full(7,1e5)]) # empty bin
                self.bin_end_time += self.interval
            #print(f"(pipeline.call) new_bin_data: ")
            #for timestamp, scores in new_bin_data:
                #print(f"    (pipeline.call) timestamp, scores/1e6: {timestamp, scores/1e6}")
            self.binned_time_series.extend(new_bin_data)
            #print("(pipeline.call) binned_time_series:")
            #for timestamp, scores in reversed(self.binned_time_series):
            #    print(f"    (pipeline.call) timestamp, scores/1e6: {timestamp, scores/1e6}")
            #self.signal.emit() # signal GUI to update the visualizer tab

        # get emotion data from current frame
        results = super().call(image) # classify faces in the image, draw boxes and labels
        #image, faces = results['image'], results['boxes2D']
        faces = results['boxes2D']
        emotion_data = self.report_emotion(faces)
        if(emotion_data is not None):
            timestamp, scores = emotion_data['time'], emotion_data['scores']
            self.time_series.append([timestamp,scores])
            self.current_bin.append(scores)
    
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
                #emotion_queue.put(emotion_data)
                return emotion_data
        return None # no large faces found
                #new_data_event.set()  # Tell the other threads that new data is available
                
 #   def __del__(self): # no log file, not needed
 #       self.log_file.close()  # Close the file when the instance is deleted
 #       print("Log file closed.")
    
def time_since(start_time):
    return int((time.time() - start_time) * 1000) # milliseconds since start of session

if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time() 
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    end_session_event = threading.Event() # triggered when the user closes the GUI window

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1, help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()
    camera = Camera(args.camera_id)

    #emotion_queue = queue.Queue() # real-time emotion logs updated continuously

    window_dims = [800, 800] # width, height
    speed = 40 # tunnel speed in pixels per second
    pipeline = EmoTunnel(start_time, 
                         window_dims, 
                         [args.offset, args.offset], 
                         #gui_app.signal.fresh_scores, # signals GUI to update the visualizer tab
                         speed
                         ) # video processing pipeline

    EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]
    
    app = QApplication(sys.argv)
    gui_app = Visualizer(start_time, window_dims, np.array(EMOTION_COLORS), speed, pipeline, end_session_event)

    print(f"Real-time emotion visualizer using FER labels sourced from on-device camera.")

    gui_app.show() # Start the GUI

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [640, 360] # width, height (16:9 aspect ratio)
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
    video_worker.stop()  # Signal the worker to stop
    #video_thread.quit()  # redundant with above, the finished signal will do this
    print("Quitting video thread...")
    video_thread.wait()  # Wait for the thread to finish
    print("Session ended.")
    profiler.disable()

    # Print the statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(50)  # Adjust the number to show more or fewer lines
