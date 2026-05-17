# video chat with OpenAI models (pipe real-time emotion logs along with user's chats)

from PyQt5.QtWidgets import QApplication  # GUI uses PyQt
from PyQt5.QtCore import QThread  # videoplayer lives in a QThread
from gui import ChatApp, VideoPlayerWorker
from emili_core import *  # core threading logic

import sys
import argparse
from paz.backend.camera import Camera
import threading
import time
from datetime import datetime
import os

if __name__ == "__main__":

    # Default to the latest mini model family for better cost/latency tradeoff.
    # gpt-5.4-mini supports text+image input in chat completions.
    model_name = "gpt-5.4-mini"
    vision_model_name = "gpt-5.4-mini"
    secondary_model_name = "gpt-5.4-nano"
    max_context_length = 16000
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()  # all threads can access this, no need to pass it!

    # full and condensed transcripts are written here at end of session
    transcript_path = "transcript"
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    # snapshots of camera frames sent to OpenAI are written here
    snapshot_path = "snapshot"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if (use_tts):
        tts_path = "tts_audio"  # temporary storage for text-to-speech audio files
        if not os.path.exists(tts_path):
            os.makedirs(tts_path)

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int,
                        default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    # Optional user ID to load per-user personalized FER artifacts.
    parser.add_argument('--user_id', type=str, default=None,
                        help='User id to load personalized FER model')
    parser.add_argument('--personalization_root', type=str, default='data/personalization',
                        help='Root folder containing personalized FER artifacts')
    args = parser.parse_args()
    camera = Camera(args.camera_id)

    chat_window_dims = [600, 600]  # width, height
    app = QApplication(sys.argv)
    gui_app = ChatApp(start_time, chat_window_dims, user_chat_name, assistant_chat_name,
                      chat_queue, chat_timestamps, new_chat_event, end_session_event)

    # Resolve optional personalization files for this user.
    personalized_checkpoint = None
    intensity_path = None
    if args.user_id is not None:
        model_dir = os.path.join(
            args.personalization_root, args.user_id, 'models')
        candidate_checkpoint = os.path.join(
            model_dir, 'personalized_siglip2.pt')
        candidate_intensity = os.path.join(
            model_dir, 'intensity_calibrator.json')
        # Load only if artifacts exist; otherwise fall back to base model behavior.
        if os.path.exists(candidate_checkpoint):
            personalized_checkpoint = candidate_checkpoint
        if os.path.exists(candidate_intensity):
            intensity_path = candidate_intensity

    # Main FER pipeline used by the video worker and chat threads.
    pipeline = Emolog(start_time,
                      [args.offset, args.offset],
                      personalized_checkpoint=personalized_checkpoint,
                      intensity_path=intensity_path)  # video processing pipeline

    # Start background threads for timing, EMA, and chat API calls.
    tick_thread = threading.Thread(target=tick)
    tick_thread.start()

    EMA_thread = threading.Thread(target=EMA_thread, args=(
        start_time, snapshot_path, pipeline), daemon=True)
    EMA_thread.start()

    sender_thread = threading.Thread(
        target=sender_thread,
        args=(model_name, vision_model_name, secondary_model_name,
              max_context_length, gui_app, transcript_path, start_time_str),
        daemon=True)
    sender_thread.start()

    assembler_thread = threading.Thread(target=assembler_thread, args=(
        start_time, snapshot_path, pipeline), daemon=True)
    assembler_thread.start()

    print(
        f"Video chat with {model_name} using emotion labels sourced from on-device camera.")
    print(f"Chat is optional, the assistant will respond to your emotions automatically!")
    print(f"Type 'q' to end the session.")

    gui_app.show()  # Start the GUI

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [800, 450]  # width, height (16:9 aspect ratio)
    # video thread: OpenCV is safe in a QThread but not a regular thread
    video_thread = QThread()
    video_worker = VideoPlayerWorker(
        start_time,
        video_dims,
        pipeline,  # applied to each frame of video
        camera)
    video_worker.moveToThread(video_thread)

    video_thread.started.connect(video_worker.run)  # connect signals and slots
    video_worker.finished.connect(video_thread.quit)
    video_worker.finished.connect(video_worker.deleteLater)
    video_thread.finished.connect(video_thread.deleteLater)
    video_worker.frameReady.connect(gui_app.display_frame)

    video_thread.start()
    print("Started video thread.")
    # start the GUI app. This should run in the main thread. Lines after this only execute if user closes the GUI.
    app.exec_()

    print("GUI app closed by user.")
    video_thread.quit()
 #   timer_thread.join()
 #   print("Timer thread joined.") # won't join while sleeping
    print("Video thread closed.")
    new_chat_event.set()  # signal assembler thread to stop waiting
    assembler_thread.join()
    print("Assembler thread joined.")
    new_message_event.set()  # signal sender thread to stop waiting
    sender_thread.join()
    print("Sender thread joined.")
    tick_event.set()  # signal tick and EMA threads to stop waiting
    EMA_thread.join()
    print("EMA thread joined.")
    tick_thread.join()
    print("Tick thread joined.")

    print("Session ended.")
