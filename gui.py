from PyQt5.QtWidgets import QMainWindow, QTabWidget, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QTransform

#from paz.backend.camera import VideoPlayer
#from paz.backend.camera import Camera
#from paz.pipelines import DetectMiniXceptionFER
from paz.backend.image import show_image, resize_image
from paz.backend.image.opencv_image import convert_color_space, BGR2RGB
import numpy as np
import json

from emili_core import time_since

class VideoPlayerWorker(QObject):
    finished = pyqtSignal()
    frameReady = pyqtSignal(np.ndarray)

    def __init__(self, image_size, pipeline, camera, topic='image'):
        super().__init__()
        self.image_size = image_size
        self.pipeline = pipeline
        self.camera = camera
        self.topic = topic
        self.stop_flag = False

    def step(self):
        if self.camera.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')

        frame = self.camera.read()
        if frame is None:
            print('Frame: None')
            return None
        frame = convert_color_space(frame, BGR2RGB)
        return self.pipeline(frame)

    def run(self):
        self.camera.start()
        while not self.stop_flag:
            output = self.step() # FER pipeline returns a dictionary with keys 'image' and 'boxes2D' (bounding boxes for faces)
            image = output[self.topic] # typically, self.topic = 'image'
            boxes = output['boxes2D']
            if image is None:
                continue
            image = resize_image(image, tuple(self.image_size)) # image is a numpy array of shape [width,height,3] and dtype uint8
            #print("emitting frameReady signal")
            self.frameReady.emit(image)
        self.camera.stop()

# Define a signal class to handle new chat messages
class ChatSignal(QObject):
    new_message = pyqtSignal(dict)  # Signal to display a new user message
    update_transcript = pyqtSignal(list)  # Signal to update the transcript display

class ChatApp(QMainWindow):
    def __init__(self, start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue, chat_timestamps, new_chat_event, end_session_event):
        super().__init__()
        self.start_time = start_time
        self.user_chat_name = user_chat_name
        self.assistant_chat_name = assistant_chat_name
        self.chat_queue = chat_queue
        self.chat_timestamps = chat_timestamps
        self.new_chat_event = new_chat_event
        self.end_session_event = end_session_event

        self.setWindowTitle("EMILI: Emotionally Intelligent Listener")
        self.resize(*chat_window_dims)  # unpack [width, height]
        self.move(100, 100)  # window position: (0,0) is top left

        # Main layout
        main_layout = QVBoxLayout()

        # Tab widget for different tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Shared input bar at the bottom
        self.chat_input = QLineEdit()
        self.chat_input.setFixedHeight(72)  # Set the height to accommodate three lines of text
        self.chat_input.setStyleSheet("QLineEdit { height: 80px; font-size: 24px; }")  # Adjust the height and font-size as needed
        self.chat_input.returnPressed.connect(self.act_on_user_input)  # function to call when user presses Enter
        main_layout.addWidget(self.chat_input)

        # Central widget setup
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.chat_signal = ChatSignal()
        self.init_chat_tab()
        self.init_FER_tab()
        self.init_transcript_tab()
        self.chat_signal.new_message.connect(self.display_new_message)
        self.chat_signal.update_transcript.connect(self.update_transcript_display)

    def closeEvent(self, event): # called when user closes the GUI window
        self.end_session_event.set()  # Signal other threads that the session should end
        event.accept()  # Continue the closing process

    def act_on_user_input(self):
        user_input = self.chat_input.text().rstrip('\n')  # remove trailing newline
        if user_input:
            self.chat_signal.new_message.emit({"role": "user", "content": user_input}) # Signal chat pane to display user message
            self.chat_input.clear()
            self.chat_timestamps.put(time_since(self.start_time))  # milliseconds since start of session
            self.chat_queue.put(user_input) # pass user message to the assembler thread
            self.new_chat_event.set()  # Signal new chat to the assembler thread

    def display_frame(self, image):
        # Convert the numpy array image to QPixmap and display it on a QLabel
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Create a QTransform for horizontal flipping. todo: flip elsewhere so the text doesn't reverse!
        #reflect = QTransform()
        #reflect.scale(-1, 1)  # Scale by -1 on the X axis for horizontal flip
        #reflected_pixmap = pixmap.transformed(reflect)

        #image_label will be displayed in the FER tab of the GUI
        self.image_label.setPixmap(pixmap)
        #self.image_label.setPixmap(reflected_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def init_FER_tab(self):
        self.FER_tab = QWidget()
        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.FER_tab.setLayout(layout)
        self.tab_widget.addTab(self.FER_tab, "FER")

    def init_transcript_tab(self):
        self.transcript_tab = QWidget()  # Create a new tab widget
        layout = QVBoxLayout()  # Use a vertical layout
        
        # Create a read-only QTextEdit widget to display the transcript
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        layout.addWidget(self.transcript_display)  # Add the QTextEdit to the layout
        
        self.transcript_tab.setLayout(layout)  # Set the layout for the transcript tab
        self.tab_widget.addTab(self.transcript_tab, "Transcript")  # Add the transcript tab to the main tab widget

    def init_chat_tab(self):
        self.chat_tab = QWidget()
        layout = QVBoxLayout()

        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setStyleSheet("QTextEdit { font-size: 18pt; }")
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # # User input area: moved to main window
        # self.chat_input = QLineEdit()
        # self.chat_input.returnPressed.connect(self.act_on_user_input) # function to call when user presses Enter
        # layout.addWidget(self.chat_input)

        self.chat_tab.setLayout(layout)
        self.tab_widget.addTab(self.chat_tab, "Chat")

    def display_new_message(self, message):  # Display new message in the chat tab
        sender = message["role"]
        content = message["content"]
        if sender == "user":
            sender = self.user_chat_name
            text = f"{sender}: {content}" # todo: color by user emotion
            text = f"<span style='font-size:18pt;'>{sender}: {content}</span><br>"
            self.chat_display.append(text)
        elif message["role"] == "assistant":
            sender = self.assistant_chat_name
            colorful_text = f"<span style='font-size:18pt;'>{sender}: <span style='color:green;'>{content}</span></span><br>"
            self.chat_display.append(colorful_text) # todo: check for verbose

    def update_transcript_display(self, transcripts):
        messages, full_transcript = transcripts
        # Convert the JSON data to a pretty-printed string
        transcript_json = json.dumps(full_transcript, indent=4, sort_keys=False) # newlines escape as '\\n'
        transcript_json = transcript_json.replace('\\n', '\n')  # Replace escaped newlines with actual newlines
        scroll_position = self.transcript_display.verticalScrollBar().value()  # Save the current scroll position
        self.transcript_display.setPlainText(transcript_json)  # renders as plain text, no HTML
        self.transcript_display.verticalScrollBar().setValue(scroll_position) # Restore the scroll position


        # transcript_html = transcript_json.replace('\\n', '<br>') # render line breaks
        # self.transcript_display.setHtml(transcript_html)
