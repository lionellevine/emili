import pygame
import pygame.midi
from time import sleep

# Initialize pygame MIDI
pygame.init()
pygame.mixer.init()
pygame.midi.init()


# List all available MIDI devices
for i in range(pygame.midi.get_count()):
    device_info = pygame.midi.get_device_info(i)
    print(f"Device {i}: {device_info}")

output_port = 1

# Open the specified MIDI output port
midi_out = pygame.midi.Output(output_port)

# Set up an instrument (0 is usually a grand piano)
instrument = 1
midi_out.set_instrument(instrument)

# Play a middle C note (note number 60)
note = 70
velocity = 127  # Max volume
midi_out.note_on(note, velocity)
sleep(2)  # Duration of the note

# Stop the note
midi_out.note_off(note, velocity)

# Close the MIDI stream and quit
midi_out.close()
pygame.midi.quit()
pygame.quit()