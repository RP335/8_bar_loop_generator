import os
from pathlib import Path

from miditok.constants import BEAT_RES

BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

# Paths
MIDI_DIR = BASE_DIR/"midi_dir"
OUTPUT_DIR = BASE_DIR/"output"
TOKENIZER_PATH = OUTPUT_DIR/"tokenizer.json"
MODEL_PATH = OUTPUT_DIR/"model"

BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}

# Tokenizer config
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (50, 200),  # (min_tempo, max_tempo)
}
# Model config
MODEL_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "sliding_window": 256,
    "max_position_embeddings": 8192,
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 48,
    "gradient_accumulation_steps": 3,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_steps": 20000,
    "logging_steps": 20,
    "save_steps": 1000,
    "eval_steps": 1000,
}

GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "num_beams": 1,
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 15,
    "top_p": 0.95,
}