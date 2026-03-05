# real-time emotion visualizer using FER labels sourced from on-device camera

from PyQt5.QtWidgets import QApplication # GUI uses PyQt
from PyQt5.QtCore import QThread # videoplayer lives in a QThread
from gui import Visualizer, VideoPlayerWorker
#from sonifier import Sonifier # optional audio thread
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
import sys
import argparse
from paz.backend.camera import Camera
import threading
import time
from datetime import datetime
from copy import deepcopy
import cProfile
import pstats

class EmoTunnel: # video pipeline for real-time FER visualizer using SigLIP2
    # SigLIP2 classes: Ahegao(0), Angry(1), Happy(2), Neutral(3), Sad(4), Surprise(5)
    # EMILI classes:   anger(0), disgust(1), fear(2), happiness(3), sadness(4), surprise(5), neutral(6)
    SIGLIP_TO_EMILI = [-1, 0, 3, 6, 4, 5]  # siglip index -> emili index (-1 = discard)
    EMILI_LABELS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    def __init__(self, start_time, dims, offsets=None, speed=25):
        self.start_time = start_time
        self.current_frame = None # other threads have read access
        self.frame_lock = threading.Lock()  # Protects access to current_frame
        self.display_width = dims[1]
        self.display_height = dims[0]
        self.time_series = [] # list of [time, scores] pairs
        self.binned_time_series = [] # list of [time, mean_scores] pairs
        self.current_bin = [] # list of scores in the current bin
        self.speed = speed # tunnel expansion rate in pixels per second, recommend 25-50
        self.interval = 1000//speed # ms per pixel
        self.bin_end_time = self.interval # start a new bin every interval ms
        self.no_data_indicator = np.full(7, 1e5) # mean scores for an empty bin
        self.last_bin_mean = np.full(7, 1e5) # mean scores for the most recent bin

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        model_id = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
        print(f"Loading {model_id} (downloading on first run, ~350 MB)...")
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SiglipForImageClassification.from_pretrained(model_id).to(self.device).eval()
        print(f"Emotion model loaded on {self.device}.")

    def __call__(self, image):
        return self.call(image)

    def call(self, image):
        # binning logic: every interval ms, record the mean scores of the current bin
        current_time = time_since(self.start_time)
        if self.bin_end_time < current_time:
            new_bin_data = []
            if len(self.current_bin) > 0:
                self.last_bin_mean = np.mean(self.current_bin, axis=0)
                new_bin_data.append([self.bin_end_time, deepcopy(self.last_bin_mean)])
                self.bin_end_time += self.interval
                self.current_bin = []
            while self.bin_end_time < current_time:
                self.last_bin_mean = 0.9*self.last_bin_mean + 0.1*self.no_data_indicator
                new_bin_data.append([self.bin_end_time, deepcopy(self.last_bin_mean)])
                self.bin_end_time += self.interval
            self.binned_time_series.extend(new_bin_data)

        annotated = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            areas = [w * h for (x, y, w, h) in faces]
            max_idx = int(np.argmax(areas))
            x, y, w, h = faces[max_idx]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

            if h > 150:
                crop = Image.fromarray(image[y1:y2, x1:x2])
                scores = self._classify(crop)
                dominant_idx = int(np.argmax(scores))
                dominant_label = self.EMILI_LABELS[dominant_idx]

                cv2.putText(annotated, dominant_label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                self._draw_scores(annotated, scores, x1, y2)

                self.time_series.append([current_time, scores])
                self.current_bin.append(scores)

        with self.frame_lock:
            self.current_frame = annotated
        return {'image': annotated}

    def _draw_scores(self, image, scores, x1, y2):
        labeled = sorted(
            [(self.EMILI_LABELS[i], int(scores[i] / 1e4)) for i in range(7)],
            key=lambda t: t[1], reverse=True
        )
        visible = [(label, pct) for label, pct in labeled if pct >= 3]
        if not visible:
            return

        line_h, pad = 24, 6
        bg_x1, bg_y1 = x1, y2 + pad
        bg_x2 = min(x1 + 155, image.shape[1])
        bg_y2 = min(bg_y1 + len(visible) * line_h + pad, image.shape[0])

        roi = image[bg_y1:bg_y2, bg_x1:bg_x2]
        if roi.size > 0:
            image[bg_y1:bg_y2, bg_x1:bg_x2] = (roi * 0.4).astype(np.uint8)

        y_text = bg_y1 + line_h - 4
        for label, pct in visible:
            cv2.putText(image, f"{label}: {pct}", (x1 + 5, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 0), 1, cv2.LINE_AA)
            y_text += line_h

    def _classify(self, pil_crop):
        inputs = self.processor(images=pil_crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        siglip_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        emili_scores = np.zeros(7, dtype=np.float64)
        for siglip_i, emili_i in enumerate(self.SIGLIP_TO_EMILI):
            if emili_i >= 0:
                emili_scores[emili_i] = siglip_probs[siglip_i]

        total = emili_scores.sum()
        if total > 0:
            emili_scores /= total

        return (emili_scores * 1e6).tolist()
    
def time_since(start_time):
    return int((time.time() - start_time) * 1000) # milliseconds since start of session

if __name__ == "__main__":

    profiler = cProfile.Profile() # for performance profiling
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

    window_dims = [720, 720] # width, height
    speed = 40 # tunnel speed in pixels per second
    pipeline = EmoTunnel(start_time, 
                         window_dims, 
                         [args.offset, args.offset], 
                         #gui_app.signal.fresh_scores, # signals GUI to update the visualizer tab
                         speed
                         ) # video processing pipeline

    EMOTION_COLORS = [[255, 0, 0], [45, 90, 45], [255, 0, 255], [255, 255, 0],
                  [0, 0, 255], [0, 255, 255], [0, 255, 0]]
    
    tonic = 110 # Hz

    app = QApplication(sys.argv)
    gui_app = Visualizer(start_time, window_dims, np.array(EMOTION_COLORS), speed, pipeline, end_session_event)

    print(f"Real-time emotion visualizer using FER labels sourced from on-device camera.")

    gui_app.show() # Start the GUI

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [800, 450] # width, height (16:9 aspect ratio)
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

    # audio_thread = QThread() # audio thread
    # audio_worker = Sonifier(start_time, speed, tonic, pipeline, end_session_event)
    # audio_worker.moveToThread(audio_thread)
    # audio_thread.start()
    # print("Started audio thread.")

    app.exec_() # start the GUI app. This should run in the main thread. Lines after this only execute if user closes the GUI.

    print("GUI app closed by user.")
    video_worker.stop()  # Signal the worker to stop
    #video_thread.quit()  # redundant with above, the finished signal will do this
    print("Quitting video thread...")
    video_thread.wait()  # Wait for the thread to finish
    print("Session ended.")
    profiler.disable()

    # Print profiling stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(20) # stats from the most expensive processes
