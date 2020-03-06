import numpy as np
import wave
from deepspeech import Model as DeepSpeechModel
from pathlib import Path

# Configurable parameters
CTC_BEAM_WIDTH = 500
LANGUAGE_MODEL_WEIGHT = 0.75
LANGUAGE_MODEL_WORD_INS_BONUS = 1.85

# Parameters that can only be changed by re-training the model
NUM_SAMPLES_PER_WINDOW = 9
NUM_MFCC_FEATURES = 26

# Folder paths to pre-trained model data files
CURRENT_FOLDER = Path(__file__).parent.absolute()
SPEECH_MODEL = "deepspeech-0.5.1-models/output_graph.pbmm"
LANGUAGE_MODEL = "deepspeech-0.5.1-models/lm.binary"
LANGUAGE_MODEL_TRIE = "deepspeech-0.5.1-models/trie"
ALPHABET_CONFIG = "deepspeech-0.5.1-models/alphabet.txt"

# The audio file we want to transcribe
AUDIO_FILE = "test.wav"

# Convert the model files to absolute paths (required by DeepSpeech)
speech_model_path = str(CURRENT_FOLDER / SPEECH_MODEL)
language_model_path = str(CURRENT_FOLDER / LANGUAGE_MODEL)
language_model_trie_path = str(CURRENT_FOLDER / LANGUAGE_MODEL_TRIE)
alphabet_config_path = str(CURRENT_FOLDER / ALPHABET_CONFIG)
audio_file_path = str(CURRENT_FOLDER / AUDIO_FILE)

# Load the pre-trained speech model
deepspeech_model = DeepSpeechModel(
    speech_model_path,
    NUM_MFCC_FEATURES,
    NUM_SAMPLES_PER_WINDOW,
    alphabet_config_path,
    CTC_BEAM_WIDTH
)

# Load the pre-trained language model
deepspeech_model.enableDecoderWithLM(
    alphabet_config_path,
    language_model_path,
    language_model_trie_path,
    LANGUAGE_MODEL_WEIGHT,
    LANGUAGE_MODEL_WORD_INS_BONUS,
)

# Load audio file using the wave library
with wave.open(audio_file_path, 'rb') as input_wave_file:
    # Get the sample rate of the audio file
    sample_rate = input_wave_file.getframerate()

    # Get the length of the audio file
    num_samples = input_wave_file.getnframes()

    # Grab the samples from the audio file
    audio_binary_data = input_wave_file.readframes(num_samples)

    # Convert the audio data into a numpy array
    audio = np.frombuffer(audio_binary_data, np.int16)

# Transcribe the audio with the model!
text_transcription = deepspeech_model.stt(audio, sample_rate)

print(text_transcription)