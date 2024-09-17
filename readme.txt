### README for MIDI Music Generation Using LSTM and Music21

This project focuses on generating music using Long Short-Term Memory (LSTM) neural networks and MIDI files. It involves two primary components:

1. **MIDI Data Processing**: This script processes MIDI files, extracts musical notes and chords, converts them into numerical representations, and prepares data for model training.
2. **Music Generation**: This script loads a pre-trained LSTM model to generate new music based on previously learned patterns and outputs the result as a MIDI file.

---

### Prerequisites

To run these scripts, you'll need the following libraries installed:

- `music21`: A Python toolkit for computer-aided musicology
- `keras`: A deep learning framework for building neural networks
- `tensorflow`: Required for running Keras models
- `numpy`: Library for numerical operations
- `random`: For generating random sequences
- `os`: To handle file system operations

Install the necessary packages using `pip`:

```bash
pip install music21 keras numpy
```

---

### Part 1: MIDI Data Processing

This script processes MIDI files, extracts notes and chords, and prepares them for training a music generation model.

#### Script: `midi_data_processing.py`

#### Key Functions:
- **`get_notes_from_midi(file_path)`**: 
    - Extracts notes and chords from a single MIDI file.
- **`process_midi_files(directory)`**:
    - Processes all MIDI files from the specified directory, extracts the notes, and stores them in a list.
  
#### Steps:
1. **Directory Setup**: Update the path to your MIDI files:
   ```python
   midi_directory = "C:\\Users\\Usman\\Downloads\\maestro-v3.0.0-midi\\maestro-v3.0.0\\2015"
use your dir where your dataset is placed
   ```

2. **Running the Script**:
    The script extracts notes and converts them into numerical data:
    ```bash
    python midi_data_processing.py
    ```

3. **Output Files**:
    The script will save the following files for model training:
    - `X.npy`: Input sequences (numerical representation of notes)
    - `y.npy`: Output note labels
    - `note_to_int.npy`: Mapping of notes to integers
    - `int_to_note.npy`: Mapping of integers to notes
    - `input_sequences.npy`: Sequence of notes for training

---

### Part 2: Music Generation

This script uses a pre-trained LSTM model to generate new music and save it as a MIDI file.

#### Script: `music_generation.py`

#### Key Functions:
- **`sample_with_temperature(predictions, temperature=1.0)`**:
    - Applies a temperature-based sampling method to the model's predictions, adding randomness to the generated output.
- **`create_midi(prediction_output, output_file="output.mid")`**:
    - Converts the predicted note sequence into a MIDI file.

#### Steps:
1. **Loading the Model**:
    Ensure that the pre-trained model `MusicModdel.keras` is in the working directory.
    
    To run the script:
    ```bash
    python music_generation.py
    ```

2. **Generated MIDI**:
    The script will generate a new music piece based on the model’s predictions and save it as `generated_music.mid`.

---

### File Descriptions

- **`midi_data_processing.py`**: 
  Script that processes MIDI files, converts them into sequences of notes, and prepares the data for training.

- **`music_generation.py`**: 
  Script that loads a pre-trained model, generates a sequence of notes, and saves the output as a MIDI file.

- **Output Files**:
  - `X.npy`, `y.npy`: Processed input data and labels.
  - `note_to_int.npy`, `int_to_note.npy`: Mapping dictionaries.
  - `input_sequences.npy`: Stored note sequences.
  - `generated_music.mid`: Generated MIDI music file.

---

### Notes

- **MIDI Files**: Ensure you have MIDI files in the directory you provide for processing.
- **Model**: You must have a pre-trained Keras LSTM model (`MusicModdel.keras`) available for generating music.
- **Temperature**: Adjusting the `temperature` value in the `music_generation.py` script will change the randomness and creativity of the generated music. A lower value will generate more predictable music, while a higher value will produce more varied and creative results.
I have also added which data set i used and also model which i trained...
Feel free to use and modify the code as needed for your projects.

### Author

This project was created by Usman Tariq. For any inquiries or issues, feel free to reach out!