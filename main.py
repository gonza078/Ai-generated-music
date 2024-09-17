import os
from music21 import converter, instrument, note, chord
import numpy as np
from keras.utils import to_categorical # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dropout, Dense, Activation # type: ignore

# Function to extract notes and chords from a MIDI file
def get_notes_from_midi(file_path):
    notes = []
    midi = converter.parse(file_path)
    parts = instrument.partitionByInstrument(midi)

    if parts:  # If the file contains instrument parts
        for part in parts.parts:
            elements_to_parse = part.recurse()
            for element in elements_to_parse:
                if isinstance(element, note.Note):  # Handling single notes
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):  # Handling chords
                    notes.append('.'.join(str(n) for n in element.normalOrder))  # Convert chord to a tuple of notes
    else:  # If there are no separate parts, parse the flat structure
        elements_to_parse = midi.flat.notes
        for element in elements_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

# Function to process all MIDI files in a directory
def process_midi_files(directory):
    all_notes = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".mid") or file_name.endswith(".midi"):  # Check for MIDI files
            file_path = os.path.join(directory, file_name)
            notes = get_notes_from_midi(file_path)  # Extract notes from each MIDI file
            all_notes.extend(notes)
            print(f"Processed {file_name}")
    
    return all_notes

# Directory containing MIDI files
midi_directory = "C:\\Users\\Usman\\Downloads\\maestro-v3.0.0-midi\\maestro-v3.0.0\\2015"
all_midi_notes = process_midi_files(midi_directory)

# Create unique notes and their mappings
unique_notes = sorted(list(set(all_midi_notes)))  # Remove duplicates and sort notes
note_to_int = {note: num for num, note in enumerate(unique_notes)}  # Map notes to integers
int_to_note = {num: note for num, note in enumerate(unique_notes)}  # Reverse mapping

# Convert all notes to numerical form
numerical_notes = [note_to_int[note] for note in all_midi_notes]

# Sequence length for training data
sequence_length = 100
input_sequences = []
output_notes = []

# Create input sequences and corresponding output notes for the model
for i in range(0, len(numerical_notes) - sequence_length):
    input_sequences.append(numerical_notes[i:i + sequence_length])  # Input sequence of 'sequence_length'
    output_notes.append(numerical_notes[i + sequence_length])  # The next note in sequence as output

# Reshape input for the LSTM model and normalize
X = np.reshape(input_sequences, (len(input_sequences), sequence_length, 1))  # Reshape to LSTM input format
X = X / float(len(unique_notes))  # Normalize input

# One-hot encode the output labels
y = np.array(output_notes)
y = to_categorical(y, num_classes=len(unique_notes))

# Save all necessary data for later use
np.save('X.npy', X)
np.save('y.npy', y)
np.save('note_to_int.npy', note_to_int)
np.save('int_to_note.npy', int_to_note)
np.save('input_sequences.npy', input_sequences)

print(f"Data processed and saved successfully.\nTotal unique notes: {len(unique_notes)}")
