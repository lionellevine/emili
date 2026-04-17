# video chat with OpenAI models (pipe real-time emotion logs along with user's chats)

from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog # GUI uses PyQt
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

TARGET_OPTIONS = ["calm", "neutral", "happy", "less anxious", "less sad", "more confident"]


def select_chat_mode():
    chooser = QMessageBox()
    chooser.setWindowTitle("Choose Chat Mode")
    chooser.setText("Select how you want to use EMILI right now.")
    chooser.setInformativeText("Talk Freely keeps current behavior. Target Emotion adds coaching toward a chosen emotional direction.")
    talk_button = chooser.addButton("Talk Freely", QMessageBox.AcceptRole)
    target_button = chooser.addButton("Target Emotion", QMessageBox.ActionRole)
    chooser.addButton(QMessageBox.Cancel)
    chooser.exec_()

    clicked = chooser.clickedButton()
    if clicked == target_button:
        target, ok = QInputDialog.getItem(
            None,
            "Target Emotion",
            "Choose a target emotional direction:",
            TARGET_OPTIONS,
            0,
            False
        )
        if ok and target:
            return "target", target
    if clicked == talk_button:
        return "default", None
    return "default", None


if __name__ == "__main__":

    # Default to the latest mini model family for better cost/latency tradeoff.
    # gpt-5.4-mini supports text+image input in chat completions.
    model_name = "gpt-5.4-mini"
    vision_model_name = "gpt-5.4-mini"
    secondary_model_name = "gpt-5.4-nano"
    max_context_length = 16000
    start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time() # all threads can access this, no need to pass it!

    transcript_path = "transcript" # full and condensed transcripts are written here at end of session
    if not os.path.exists(transcript_path):
        os.makedirs(transcript_path)
    snapshot_path = "snapshot" # snapshots of camera frames sent to OpenAI are written here
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if(use_tts):
        tts_path = "tts_audio" # temporary storage for text-to-speech audio files
        if not os.path.exists(tts_path):
            os.makedirs(tts_path)

    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1, help='Scaled offset to be added to bounding boxes')
    parser.add_argument('--user_id', type=str, default=None, help='User id to load personalized FER model')
    parser.add_argument('--personalization_root', type=str, default='data/personalization',
                        help='Root folder containing personalized FER artifacts')
    args = parser.parse_args()
    camera = Camera(args.camera_id)

    chat_window_dims = [600, 600] # width, height
    app = QApplication(sys.argv)
    gui_app = ChatApp(start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue, chat_timestamps, new_chat_event, end_session_event)
    selected_mode, selected_target = select_chat_mode()
    configure_chat_mode(selected_mode, selected_target)

    personalized_checkpoint = None
    intensity_path = None
    if args.user_id is not None:
        model_dir = os.path.join(args.personalization_root, args.user_id, 'models')
        candidate_checkpoint = os.path.join(model_dir, 'personalized_siglip2.pt')
        candidate_intensity = os.path.join(model_dir, 'intensity_calibrator.json')
        if os.path.exists(candidate_checkpoint):
            personalized_checkpoint = candidate_checkpoint
        if os.path.exists(candidate_intensity):
            intensity_path = candidate_intensity

    pipeline = Emolog(start_time,
                      [args.offset, args.offset],
                      personalized_checkpoint=personalized_checkpoint,
                      intensity_path=intensity_path) # video processing pipeline

    tick_thread = threading.Thread(target=tick)
    tick_thread.start()

    EMA_thread = threading.Thread(target=EMA_thread, args=(start_time,snapshot_path,pipeline), daemon=True)
    EMA_thread.start()

    sender_thread = threading.Thread(
        target=sender_thread, 
        args=(model_name, vision_model_name, secondary_model_name, max_context_length, gui_app, transcript_path, start_time_str), 
        daemon=True)
    sender_thread.start()

    assembler_thread = threading.Thread(target=assembler_thread, args=(start_time,snapshot_path,pipeline), daemon=True)
    assembler_thread.start()

    print(f"Video chat with {model_name} using emotion labels sourced from on-device camera.")
    if selected_mode == "target":
        print(f"Chat mode: target emotion ({selected_target})")
    else:
        print("Chat mode: talk freely")
    print(f"Chat is optional, the assistant will respond to your emotions automatically!")
    print(f"Type 'q' to end the session.")

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
