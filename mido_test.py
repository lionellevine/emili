import mido
from time import sleep

# List available MIDI outputs
MIDI_outputs = mido.get_output_names()
print(MIDI_outputs)

# Select the first MIDI output
MIDI_output_name = MIDI_outputs[0]
outport = mido.open_output(MIDI_output_name)

outport.send(mido.Message('control_change', control=121, value=0, channel=0)) # Reset all controllers

# Set the program (instrument) to Cello on channel 0
program_change = mido.Message('program_change', program=42, channel=0)
outport.send(program_change)

# Play a C major chord (note number 60, 64, 67) on channel 0 with a velocity of 96
notes = [60, 64, 67]
for note in notes:
    note_on = mido.Message('note_on', note=note, velocity=96, channel=0)
    outport.send(note_on)

# Let the notes play for 1 second
sleep(1)

# Stop the notes on channel 0
for note in notes:
    note_off = mido.Message('note_off', note=note, velocity=96, channel=0)
    outport.send(note_off)
    sleep(1)

# Change the program (instrument) to Vibraphone on channel 0
program_change = mido.Message('program_change', program=12, channel=0)
outport.send(program_change)

# Play the same C major chord with the new instrument on channel 0
for note in notes:
    note_on = mido.Message('note_on', note=note, velocity=96, channel=0)
    outport.send(note_on)

# Let the notes play for 1 second
sleep(1)

# Stop the notes on channel 0
for note in notes:
    note_off = mido.Message('note_off', note=note, velocity=96, channel=0)
    outport.send(note_off)
    sleep(1)

# Close the MIDI output port
outport.close
