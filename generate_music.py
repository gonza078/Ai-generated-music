import numpy as np
from keras.models import load_model
import random
from music21 import instrument, note, stream, chord
from keras.preprocessing.sequence import pad_sequences # type: ignore

# Load the model and mappings
model = load_model('MusicModdel.keras', compile=False)
note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
int_to_note = np.load('int_to_note.npy', allow_pickle=True).item()
input_sequences = np.load('input_sequences.npy')


n_vocab = len(note_to_int)
sequence_length = 100
temperature = 1.0 
num_notes_to_generate = 100


start_sequence_index = random.randint(0, len(input_sequences) - 1)
input_seq = input_sequences[start_sequence_index].tolist()

# Ensure input is correctly shaped and normalized
input_seq = pad_sequences([input_seq], maxlen=sequence_length, padding='pre')[0]
input_seq = input_seq / float(n_vocab)

# Prediction output
prediction_output = []

# Function for sampling based on temperature
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


for note_index in range(num_notes_to_generate):
    input_seq_reshaped = np.reshape(input_seq, (1, len(input_seq), 1))

    
    prediction = model.predict(input_seq_reshaped, verbose=0)[0]
    
    
    index = sample_with_temperature(prediction, temperature)
    
    result_note = int_to_note.get(index, 'C4')  
    prediction_output.append(result_note)
        
    input_seq = np.append(input_seq[1:], index / float(n_vocab))

def create_midi(prediction_output, output_file="output.mid", chosen_instrument=instrument.Piano(), min_offset=0.5, max_offset=0.8):
    offset = 0  
    output_notes = []

    for pattern in prediction_output:
        
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = chosen_instrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = chosen_instrument
            output_notes.append(new_note)
        
        # Increment offset for next note/chord
        offset += random.uniform(min_offset, max_offset)
    
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"MIDI file saved as {output_file}")
    
create_midi(prediction_output, output_file="generated_music.mid")
