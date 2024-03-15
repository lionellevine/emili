# video chat with OpenAI models (pipe real-time emotion logs along with user's chats)

from PyQt5.QtWidgets import QApplication # GUI uses PyQt
from PyQt5.QtCore import QThread # videoplayer lives in a QThread
from gui import ChatApp, VideoPlayerWorker
from emili_core import * # core threading logic

import sys
import argparse
from paz.backend.camera import Camera
import threading
import time
from datetime import datetime
import os

from openai import OpenAI
client = OpenAI()

if __name__ == "__main__":

    # pricing as of March 2024 per 1M tokens read: gpt-3.5-turbo-0125 $0.50, gpt-4-0125-preview $10, gpt-4 $30
    model_name = "gpt-4-0125-preview" # start with a good model
    secondary_model_name = "gpt-3.5-turbo-0125" # switch to a cheaper model if the conversation gets too long
    max_context_length = 16000
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time() # all threads can access this, no need to pass it!
    tock_interval = None # default 60000 ms between OpenAI API calls, if no user text input. Set None to disable

    transcript_path = "transcript" # full and condensed transcripts are written here at end of session
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    if(use_tts):
        tts_path = "tts_audio" # temporary storage for text-to-speech audio files
        if not os.path.exists(tts_path):
            os.makedirs(tts_path)

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1, help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()
    camera = Camera(args.camera_id)

    chat_window_dims = [600, 600] # width, height
    app = QApplication(sys.argv)
    gui_app = ChatApp(start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue, chat_timestamps, new_chat_event, end_session_event)

    tick_thread = threading.Thread(target=tick)
    tick_thread.start()

    EMA_thread = threading.Thread(target=EMA_thread, args=(start_time,), daemon=True)
    EMA_thread.start()

    sender_thread = threading.Thread(
        target=sender_thread, 
        args=(model_name, secondary_model_name, max_context_length, gui_app, transcript_path, start_time_str), 
        daemon=True)
    sender_thread.start()

    assembler_thread = threading.Thread(target=assembler_thread, daemon=True)
    assembler_thread.start()

    if(tock_interval is not None):
        timer_thread = threading.Thread(target=timer_thread, args=(start_time,tock_interval), daemon=True)
        timer_thread.start()

    print(f"Video chat with {model_name} using emotion labels sourced from on-device camera.")
    print(f"Chat is optional, the assistant will respond to your emotions automatically!")
    print(f"Type 'q' to end the session.")

    gui_app.show() # Start the GUI

    print("Started GUI app.")
    print("gui_app.thread()", gui_app.thread())
    print("QThread.currentThread()", QThread.currentThread())

    video_dims = [800, 450] # width, height (16:9 aspect ratio)
    video_thread = QThread() # video thread: GPT-4 says OpenCV is safe in a QThread but not a regular thread?
    video_worker = VideoPlayerWorker(
        video_dims,
        Emolog(start_time, [args.offset, args.offset]), 
        camera)
    video_worker.moveToThread(video_thread)

    video_thread.started.connect(video_worker.run) # connect signals and slots
    video_worker.finished.connect(video_thread.quit)
    video_worker.finished.connect(video_worker.deleteLater)
    video_thread.finished.connect(video_thread.deleteLater)
    video_worker.frameReady.connect(gui_app.display_frame)

    video_thread.start()
    print("Started video thread.")
    app.exec_() # start the GUI app. This should run in the main thread. Lines after this only execute if user closes the GUI.

    print("GUI app closed by user.")
    video_thread.quit()
 #   timer_thread.join()
 #   print("Timer thread joined.") # won't join while sleeping
    print("Video thread closed.")
    new_chat_event.set() # signal assembler thread to stop waiting
    assembler_thread.join() 
    print("Assembler thread joined.")
    new_message_event.set() # signal sender thread to stop waiting
    sender_thread.join()
    print("Sender thread joined.")
    tick_event.set() # signal tick and EMA threads to stop waiting
    EMA_thread.join()
    print("EMA thread joined.")
    tick_thread.join()
    print("Tick thread joined.")
        
    print("Session ended.")
