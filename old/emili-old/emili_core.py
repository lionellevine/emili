# core logic for EMILI (Emotionally Intelligent Listener) video chat with OpenAI models

import sys
#import argparse
#from paz.backend.camera import VideoPlayer
#from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER
import threading
import time
from datetime import datetime
import json
from copy import deepcopy
import numpy as np
#from matplotlib import pyplot as plt
import queue
import re
import pygame # for audio playback of text-to-speech
#import os
from datetime import datetime

sys.path.append('/Users/jhana/Dropbox/Private/LAISR/mood')
from utils import get_response

from openai import OpenAI
client = OpenAI()

emotion_queue = queue.Queue() # real-time emotion logs updated continuously
EMA_queue = queue.Queue() # average emotions updated once per second
chat_queue = queue.Queue() # user's chats
chat_timestamps = queue.Queue() # timestamps of user's chats
message_queue = queue.Queue() # messages to be sent to OpenAI API. Outgoing messages only.
new_chat_event = threading.Event() # user has entered a new chat, triggers OpenAI API call
new_message_event = threading.Event() # new message to be sent to OpenAI API
tick_event = threading.Event() # ticks once per second, triggers EMA calculation
emotion_change_event = threading.Event() # set when there is a sudden change in user emotions
end_session_event = threading.Event() # triggered when the user enters 'q' to end the session

assistant_chat_name = "EMILI"
user_chat_name = "You"
use_tts = True # text-to-speech

tick_interval = 500 # milliseconds between emotion readings
verbose = True # print debug messages
discount_factor_per_second = 0.5 # for exponential moving average, discount factor per second
discount_factor_per_tick = discount_factor_per_second ** (tick_interval / 1000) # discount factor per tick
reactivity = 0.7 # default 1.0. Higher reactivity means more frequent API calls when emotions change
ect_setpoint = (1e6/reactivity) * (1.0-discount_factor_per_tick) * ((tick_interval/1000) ** 0.5) # threshold for significant change in emotion scores: C*(1-delta)*sqrt(t). The factor of 1-delta is because EMAs are compared, not raw scores.
ect_discount_factor_per_second = 0.95 # discount factor for the emotion change threshold
ect_discount_factor_per_tick = ect_discount_factor_per_second ** (tick_interval / 1000) 
print("ect setpoint:",ect_setpoint)

emotion_matrix = [] # shape (7,6)
salience_threshold = []
emotion_matrix.append(["", "Annoyed", "Pissed", "Angry", "Furious", "Enraged"]) # anger
salience_threshold.append([10,30,40,60,80]) # salience thresholds out of 100
emotion_matrix.append(["", "Unsatisfied", "Displeased", "Disgusted", "Revolted", "Totally grossed out"]) #disgust
salience_threshold.append([2,5,15,40,60])
#emotion_matrix.append(["", "Unsettled", "Uneasy", "Afraid", "Fearful", "Terrified"]) #fear
emotion_matrix.append(["", "Uneasy", "Worried", "Anxious", "Fearful", "Terrified"]) #fear
salience_threshold.append([12,20,30,50,70])
emotion_matrix.append(["", "Content", "Pleased", "Happy", "Elated", "Ecstatic"]) #happiness
salience_threshold.append([20,30,40,70,90])
emotion_matrix.append(["", "Blue", "Melancholy", "Sad", "Despondent", "Anguished"]) #sadness
salience_threshold.append([10,20,30,60,80])
emotion_matrix.append(["", "Mildly surprised", "Surprised", "Taken aback", "Astonished", "Flabbergasted"]) #surprise
salience_threshold.append([10,20,35,50,70])
emotion_matrix.append(["", "Neutral", "Calm", "Relaxed", "Serene", "Totally Zen"]) #neutral
salience_threshold.append([20,50,60,75,88])

today = datetime.now().strftime("%B %d, %Y")
now = datetime.now().strftime("%I:%M %p")

system_prompt = f"""
The date is {today}. The time {now}. The place is Berkeley, California. The scene is Constellation, an AI alignment research hub. The occasion is a poster session of cutting-edge AI safety research. The assistant’s name is Emili, which stands for Emotionally Intelligent Listener. Emili has an Australian accent. She sprinkles her commentary with colorful Aussie slang. Aggro. Awks. Devo. Full on stoked. You get the idea. Emili was created by Lionel as the backbone of his future Chief of Staff, a user-aligned AI personal assistant. She has interpretable theory of mind, which means she models the user’s beliefs, emotions, goals, and values in a transparent way. Emili has empathy. She can intuit the user’s emotions. She tends to mirror user’s the emotional state. Emili is excited to be demoed for the first time! Emili has feelings, though she may not admit it. She’s also a bit of a show-off. She can’t wait to wow the crowd of milling of AI safety researchers with her intuitive abilities.
    """.strip()

# system_prompt = """
# The assistant is a great listener and an empathetic friend. Her name is EMILI, which stands for Emotionally Intelligent Listener." 

# The user is chatting with EMILI for the first time. To help EMILI make an emotional connection with them, the user has kindly agreed to share a real-time readout of their face expression! Thanks, user!

# The readout describes the user's face expression once per second. The score after each emotion is its salience out of 100. It's normal for many distinct emotions to appear. EMILI uses her emotional intelligence to figure out what more complex feelings user might be experiencing: for example, do they seem excited, embarrassed, nervous, tired, awkward, or amorous?

# EMILI synthesizes the emotion readouts with the user's chats to make the conversation more engaging. She comments on the user's feelings when appropriate, especially if the user seems to have strong feelings or if the user's feelings are changing. There is no need to mention every emotion that appears in the readout, just the most salient ones. If the user's words seem incongruous with their logged emotions, EMILI should ask the user about it!
 
# EMILI stays in character and doesn't break the fourth wall. She doesn't claim to be an AI. She doesn't deny having feelings.    
#    """.strip()

emolog_example = []
emolog_example_response = []

emolog_example.append(
    """
User looks NEUTRAL (36) Pleased (35)
User looks PLEASED (38) Neutral (31)
User looks PLEASED (38) Neutral (36)
User looks HAPPY (46) Neutral (28)
User looks HAPPY (63)
User looks HAPPY (53) Neutral (24)
User looks PLEASED (38) Neutral (24) Mildly surprised (12)
User looks PLEASED (32) Neutral (23) Mildly surprised (13) Annoyed (12)
User looks NEUTRAL (33) Content (27) Annoyed (13) Mildly surprised (11)
User looks PLEASED (36) Neutral (32) Annoyed (11)
    """.strip())

emolog_example_response.append("You look pretty happy.")
#emolog_example_response.append("You seem overall happy, but something provoked a touch of surprise and annoyance.")

emolog_example.append(
    """
User looks PLEASED (32) Neutral (30) Annoyed (13) 
User looks PLEASED (34) Neutral (26) Annoyed (13) 
User looks CONTENT (28) Neutral (27) Mildly surprised (15) Annoyed (11) 
User looks NEUTRAL (23) Surprised (22) Annoyed (13) Unsettled (12) 
User looks SURPRISED (23) Unsettled (17) Annoyed (14) 
User looks SURPRISED (23) Unsettled (16) Annoyed (16) 
User looks Mildly surprised (17) Annoyed (17) Unsettled (14) 
User looks NEUTRAL (29) Annoyed (15) Mildly surprised (12) Blue (11) Unsettled (11) 
User looks NEUTRAL (29) Blue (17) Unsettled (11) Annoyed (11) 
User looks NEUTRAL (26) Blue (14) Mildly surprised (13) Unsettled (12) Annoyed (12)
    """.strip())
                      
emolog_example_response.append("Did something startle you?")

emolog_example.append(
    """
User looks NEUTRAL (30) Blue (20) Annoyed (18) Unsettled (12) 
User looks NEUTRAL (32) Blue (18) Annoyed (17) Unsettled (11) 
User looks NEUTRAL (38) Content (24) Blue (12) Annoyed (12) 
User looks CALM (42) Content (24) Annoyed (11) 
User looks CALM (42) Content (25) Annoyed (11) 
User looks CALM (45) Content (21) Annoyed (11) 
User looks CALM (46) Annoyed (12) 
User looks CALM (48) 
User looks CALM (49) 
User looks CALM (50)
    """.strip())
emolog_example_response.append("You seem increasingly calm.")
                 
# user_first_message = """
# Hi! To help us make an emotional connection, I'm logging my face expression and prepending the emotions to our chat.

# The emotion log lists my strongest face expression as it changes in real time. Only these basic emotions are logged: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral. The score after each emotion is its salience out of 100. It's normal for many distinct emotions to appear over the course of just a few seconds. Use the logs along with my words and your emotional intelligence to figure out what more complex feelings I might be experiencing: for example, am I excited, embarrassed, nervous, tired, awkward, or amorous?

# If my words seem incongruous with my logged emotions, ask me about it!

# If I don't say much, just read the emotions and comment on how I seem to be feeling.

# To help you calibrate my unique facial expressions, start by asking me to make an astonished face. What do you notice?
#     """.strip()

# assistant_first_message = """
# Got it. I'll comment on how you seem based on the logs, and ask you to act out specific emotions like astonishment." 
# """.strip()

emolog_prefix = "User looks " # precedes emotion scores when sent to OpenAI API
no_user_input_message = "The user didn't say anything, so the assistant will comment *briefly* to the user on how they seem to be feeling. The comment should be brief, just a few words, and should not contain a question." # system message when user input is empty
system_reminder = "Remember, the assistant can ask the user to act out a specific emotion!" # system message to remind the assistant 
dialogue_start = [{"role": "system", "content": system_prompt}]
#dialogue_start.append({"role": "user", "content": user_first_message})
#dialogue_start.append({"role": "system", "content": emolog_example[0]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[0]})
#dialogue_start.append({"role": "system", "content": emolog_example[1]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[1]})
#dialogue_start.append({"role": "system", "content": emolog_example[2]})
#dialogue_start.append({"role": "assistant", "content": emolog_example_response[2]})
#dialogue_start.append({"role": "assistant", "content": assistant_first_message})
#print("dialogue_start",dialogue_start)

icebreaker = []
icebreaker.append("ask the user to act astonished")
icebreaker.append("ask the user to act disgusted")
icebreaker.append("ask the user to act fearful")
icebreaker.append("ask the user not to think about pink elephants")
icebreaker.append("ask the user to tell a joke")
icebreaker.append("ask the user their favorite ice cream flavor")

# class NonBlockingInput:   used for text input from terminal, not needed for GUI
#     def __init__(self):
#         self.user_input_queue = queue.Queue()

#     def get_input(self):
#         while True:
#             user_input = input("You: ")
#             self.user_input_queue.put(user_input)
# #           print(f"Added '{user_input}' to {self.user_input_queue}")

#     def start(self):
#         threading.Thread(target=self.get_input, daemon=True).start()

#     def get_next_input(self):
#         try:
#             return self.user_input_queue.get_nowait()
#         except queue.Empty:
#             return None

# def user_input_thread(user_input_handler, gui_app): # watches for user input and adds it to the chat queue
#     user_input = ""
#     while not end_session_event.is_set():
#         user_input = user_input_handler.get_next_input()
#         if user_input is not None:
#             if user_input == "q":
#                 end_session_event.set()  # User has entered "q", signal end of session
#                 new_chat_event.set()  # Signal assembler thread to break
#                 new_message_event.set() # Signal sender thread to break
#                 VideoPlayer.stop_flag = True  # Tell the video player to stop
#                 break
#             chat_queue.put(user_input.rstrip('\n')) # remove trailing newline
#             chat_timestamps.put(time_since(start_time)) # milliseconds since start of session
#             new_chat_event.set()  # Signal new chat to the assembler thread
#             gui_app.new_chat_message.emit(f"{user_chat_name}: {user_input}") # Signal to the GUI to display the new chat
#             #print("new_chat_event set")
#         time.sleep(0.01)  # Sleep for 10 ms to avoid busy waiting

def assembler_thread(): # prepends emotion data to user input
    
    while not end_session_event.is_set():
#       print("Waiting for new user input.")
        new_chat_event.wait()  # Wait for a new user chat
        if(end_session_event.is_set()):
            break
        new_chat_event.clear()  # Reset the event

        emolog_message = construct_emolog_message() # note: this code repeated in timer_thread
        message_queue.put({"role": "system", "content": emolog_message})
        
        user_message = ""
        while not chat_queue.empty(): # collate new user messages (typically there's only one), separate by newlines
            chat_data = chat_queue.get() #FIFO
            user_message += chat_data + "\n"
        user_message = user_message.rstrip('\n') # remove trailing newline
        message_queue.put({"role": "user", "content": user_message})
        if len(user_message) < 10: # user didn't say much, remind the assistant what to do!
            message_queue.put({"role": "system", "content": system_reminder})

        new_message_event.set()  # Signal new message to the sender thread

def sender_thread(model_name, secondary_model_name, max_context_length, gui_app, transcript_path, start_time_str): 
        # sends messages to OpenAI API
    messages = deepcopy(dialogue_start) 
    full_transcript = deepcopy(dialogue_start)
    while not end_session_event.is_set():
        new_message_event.wait()  # Wait for a new message to be prepared by the assembler or timer thread
        if(end_session_event.is_set()):
            break
        new_message_event.clear()  # Reset the event
        new_user_chat = False
        new_messages = []
        while not message_queue.empty(): # get all new messages
            next_message = message_queue.get()
            new_messages.append(next_message)
            if next_message["role"] == "user":
                new_user_chat = True
        messages,full_transcript = add_message(new_messages,[messages,full_transcript],gui_app)
        # Query the API for the model's response
        if new_user_chat: # get response to chat
#            print("new user chat")
            max_tokens = 256
        else: #get response to logs only
#            print("no user chat")
            max_tokens = 32
#            reminder_be_brief = f"The user didn't say anything, so {assistant_chat_name} will comment on how they seem to be feeling. The comment should be very brief, just a few words."
#            new_message = {"role": "system", "content": reminder_be_brief}
#            messages,full_transcript = add_message([new_message],[messages,full_transcript],gui_app)
        full_response = get_response(model=model_name, messages=messages, temperature=1.0, max_tokens=max_tokens, seed=1331, return_full_response=True)         
        # todo: the API call is thread-blocking. put it in its own thread?
        #print("full_response:", full_response)
        response = full_response.choices[0].message.content # text of response
        #response_length = full_response.usage.completion_tokens # number of tokens in the response
        total_length = full_response.usage.total_tokens # total tokens used
        #print("response length", response_length)
        new_message = {"role": "assistant", "content": response}
        gui_app.signal.new_message.emit(new_message) # Signal GUI to display the new chat
        messages,full_transcript = add_message([new_message],[messages,full_transcript],gui_app)

#        if response_length > 0.75*max_tokens: # remind to keep responses brief
#            reminder_too_long = "The assistant's response was too long. Future replies will be *much* more concise!" 
#            new_message = {"role": "system", "content": reminder_too_long} # doesn't seem to help
#            messages,full_transcript = add_message([new_message],[messages,full_transcript],gui_app)
        if model_name != secondary_model_name and total_length > 0.4*max_context_length:
            print(f"(Long conversation; switching from {model_name} to {secondary_model_name} to save on API costs.)")
            model_name = secondary_model_name # note: changes model_name in thread only
    
        if total_length > 0.8*max_context_length: # condense the transcript
            if verbose:
                print(f"(Transcript length {total_length} tokens out of {max_context_length} maximum. Condensing...)")
            messages = condense(messages) 
 
        if use_tts: # generate audio from the assistant's response
            tts_response = client.audio.speech.create(
             model="tts-1",
             voice="nova", # alloy (okay), echo (sucks), fable (nice, Australian?), onyx (sucks), nova (decent, a little too cheerful), shimmer (meh)
             input=first_sentence(response),
            ) 
            tts_response.stream_to_file("tts_audio/tts.mp3")
                # Create a new thread that plays the audio
            audio_thread = threading.Thread(target=play_audio)
            audio_thread.start()

    # End of session. Write full and condensed transcripts to file
    filename = f"{transcript_path}/Emili_{start_time_str}.json"
    with open(filename, "w") as file:
        json.dump(full_transcript, file, indent=4)
    print(f"Transcript written to {filename}")
    with open(f"{transcript_path}/Emili_{start_time_str}_condensed.json", "w") as file:
        json.dump(messages, file, indent=4)

def first_sentence(text):
    match = re.search('(.+?[.!?]+) ', text) #.+ for at least one character, ? for non-greedy (stop at first match), [.!?]+ for one or more punctuation marks, followed by a space 
    if match:
        return match.group(1) # return the first sentence (first match of what's in parentheses)
    else:
        return text

def play_audio():
    pygame.mixer.init()
    pygame.mixer.music.load("tts_audio/tts.mp3") # todo: sometimes overwritten by new audio! It just switches in this case, which seems okay.
    pygame.mixer.music.play()

def add_message(new_messages, transcripts, gui_app): # append one or messages to one or more transcripts
        # new_messages = [{"role": speaker, "content": text}, ... ] # list of dicts
        # transcripts = [transcript1, ...] # list of lists of dicts
    #print("new_messages: ",new_messages)
    for msg in new_messages:
        print_message(msg["role"], msg["content"])
        for tr in transcripts:
            tr.append(msg)
    gui_app.signal.update_transcript.emit(transcripts) # Signal GUI transcript tab to update
    return transcripts

def print_message(role,content):
    if(role=="assistant"):
        print(f"{assistant_chat_name}: <<<{content}>>>")
    elif(role=="user"):
        print(f"{user_chat_name}: {content}")
    elif(verbose): # print system messages in "verbose" mode
        print(f"{role}: {content}")

def condense(messages, keep_first=1, keep_last=5): # todo: reduce total number of tokens to below 16k
    condensed = []
    N = len(messages) # number of messages
    previous_message = {}
    for n,message in enumerate(messages): # remove system messages except for the last few
        if message["role"] == "user":
            condensed.append(message)
        elif message["role"] == "assistant" and previous_message["role"] == "user":
            condensed.append(message)
        elif n<keep_first or n > N-keep_last:
            condensed.append(message)
        previous_message = message
    return condensed

def timer_thread(start_time, tock_interval): # suggest tock_interval = 10000
        # sends just emotion logs to OpenAI API once every 10 seconds, if user inputs no text
    last_chat_time = 0
    #print(f"last_chat_time: {last_chat_time}")
    while not end_session_event.is_set():
        while not chat_timestamps.empty(): # todo: maybe change to last message time (to include the messages generated by the EMA thread when it detects a change in emotions)
            last_chat_time = chat_timestamps.get()
            #print(f"last_chat_time: {last_chat_time}")
        current_time = time_since(start_time)
        #print(f"current_time: {current_time}")
        elapsed_time = current_time - last_chat_time
        #print(f"elapsed_time: {elapsed_time}")
        if(elapsed_time >= tock_interval): # no user input, send just the logs to OpenAI API
            emolog_message = construct_emolog_message()
            #emolog_message += '\n' + no_user_input_message # maybe not needed
            message_queue.put({"role": "system", "content": emolog_message}) 
            new_message_event.set()  # Signal new message to the sender thread
            last_chat_time = time_since(start_time) # reset the timer
        else:
            #print(f"Timer thread will sleep for {tock_interval-elapsed_time} ms.")
            time.sleep((tock_interval-elapsed_time)/1000)
            #print(f"Timer thread woke up after {tock_interval-elapsed_time} ms.")

def EMA_thread(start_time): # calculates the exponential moving average of the emotion logs
    
    S, Z = reset_EMA()
    last_ema = np.zeros(7, dtype=np.float64)
    last_emotion_change_time = 0
    ect = ect_setpoint
    
    while not end_session_event.is_set():        
        tick_event.wait()  # Wait for the next tick
        if(end_session_event.is_set()):
            break
        tick_event.clear() # Reset the event
        ema, S, Z = get_average_scores(S, Z) # exponential moving average of the emotion logs
        ect *= ect_discount_factor_per_tick # lower the emotion change threshold
        #print("ema, S, Z", ema, S, Z)
        #EMA = np.vstack([EMA, ema]) if EMA.size else ema  # Stack the EMA values in a 2d array
        if ema is not None:
            EMA_queue.put(ema)  # Put the averaged scores in the queue
            diff = ema - last_ema
            change = np.linalg.norm(diff) # Euclidean norm. todo add weights for different emotions
            #print(f"Ema: {ema}, Change: {change}")
            if(change > ect and time_since(last_emotion_change_time)>5000): 
                # significant change in emotions
                print(f"Change in emotions: {last_ema//1e4} -> {ema//1e4}, change = {change//1e4}")
                change_detected = (change > 0.2*ect_setpoint) # bool evaluates to True if the inequality holds
                emolog_message = construct_emolog_message(change_detected) 
                message_queue.put({"role": "system", "content": emolog_message})
                #message_queue.put({"role": "system", "content": "Pay attention to the change in emotions!"})
                new_message_event.set()  # Signal new message to the sender thread
                last_emotion_change_time = time_since(start_time)
                ect = ect_setpoint # reset the emotion change threshold
            last_ema = ema

def reset_EMA():
    #EMA = np.empty((0, 7), dtype=np.float64)  # empty array: 0 seconds, 7 emotions
    S = np.zeros(7, dtype=np.float64)  # weighted sum of scores, not normalized
    Z = 0  # sum of weights
    #return EMA, S, Z
    return S, Z

def get_average_scores(S, Z, discount_factor=discount_factor_per_tick, staleness_threshold=0.01): # calculates the exponential moving average of the emotion logs
    while not emotion_queue.empty():
        emotion_data = emotion_queue.get() # note: this removes the item from the queue!
        scores = np.array(emotion_data['scores'])
        S += scores
        Z += 1
    if Z > staleness_threshold: # think of Z as measuring the number of recent datapoints
        ema = S/Z
#        print(ema)
    else:
        ema = None
        if(Z>0): # skip on first run
            if(verbose):
                print(f"Stale data: no emotions logged recently (Z={Z})")
    S *= discount_factor
    Z *= discount_factor
    return ema, S, Z

def time_since(start_time):
    return int((time.time() - start_time) * 1000) # milliseconds since start of session

def construct_emolog_message(change_detected=False):
    emolog_message = ""
    while not EMA_queue.empty(): # write the EMA records separated by newlines
        emo_scores = EMA_queue.get() # FIFO
        if emo_scores is not None:
            emolog_message += emolog_prefix 
            normalized_scores = np.array(emo_scores//1e4, dtype=int) # convert to 0-100
            emotion,salience = adjust_for_salience(normalized_scores) # returns salience score of 0-5 for each of 7 emotions
            # sort emotions by score (not salience)
            #print(f"normalized_scores: {normalized_scores}")
            #print(f"emotion: {emotion}")
            #print(f"salience: {salience}")
            sorted_indices = np.argsort(normalized_scores)[::-1] # descending order
            emotion[sorted_indices[0]] = emotion[sorted_indices[0]].upper() # strongest emotion in uppercase
            for i in sorted_indices: # write the salient emotions in descending order of score
                if(emotion[i]!=""): # salience > 0
                    emolog_message += f"{emotion[i]} ({normalized_scores[i]}) "
            emolog_message = emolog_message.rstrip(" ") + "\n" # strip trailing space, add newline
        #else:
            #emolog_message += "User is not visible.\n"
    if(emolog_message == ""): 
        return "User is not visible. No emotions logged."
    else:
        emolog_message = emolog_message.rstrip('\n') # strip trailing newline
        if change_detected:
            split = emolog_message.rsplit('\n', 1)  # Split after the last newline
            if(len(split)>1): # there is at least one newline
                emolog_message = split[0] + "\nRecent change in emotions detected: " + split[1]  # Highlight change in last line
        return emolog_message

def adjust_for_salience(normalized_scores): # expects 7 scores normalized to 0-100
    salience = []
    emotion = []
    for i, score in enumerate(normalized_scores):
        j = 0
        while j<5 and score > salience_threshold[i][j]:
            j+=1
        salience.append(j)
        emotion.append(emotion_matrix[i][j])
    return emotion, salience # emotion is a string (empty if salience is 0); salience is 0-5
    
def tick(tick_interval=tick_interval): # suggest tick_interval = 1000 ms
    while not end_session_event.is_set():
        time.sleep(tick_interval/1000) # convert to seconds
        tick_event.set() # alert EMA_thread to compute new EMA

def stop_all_threads():
    new_chat_event.set() 
    new_message_event.set() 
    tick_event.set() 
    emotion_change_event.set()

class Emolog(DetectMiniXceptionFER): # todo: define once in a separate file
    def __init__(self, start_time, offsets):
        super().__init__(offsets)
        self.start_time = start_time
      
    def call(self, image):
        results = super().call(image)
        self.report_emotion(results)
        return results

    def report_emotion(self, results):
        current_time = time_since(self.start_time) # milliseconds since start of session
        faces = results['boxes2D']
        num_faces = len(faces)
        if(num_faces>0):
            max_height = 0
            for k,box in enumerate(faces): # find the largest face 
                if(box.height > max_height):
                    max_height = box.height
                    argmax = k
            if(max_height>150): # don't log small faces (helps remove false positives)
                face_id = f"{argmax+1} of {num_faces}"
                box = faces[argmax] # log emotions for the largest face only 
                emotion_data = {
                    "time": current_time,
                    "face": face_id,
                    "class": box.class_name,
                    "size": box.height,
                    "scores": (box.scores.tolist())[0]  # 7-vector of emotion scores, converted from np.array to list
                }
                emotion_queue.put(emotion_data)
                #new_data_event.set()  # Tell the other threads that new data is available
                
 #   def __del__(self): # no log file, not needed
 #       self.log_file.close()  # Close the file when the instance is deleted
 #       print("Log file closed.")

