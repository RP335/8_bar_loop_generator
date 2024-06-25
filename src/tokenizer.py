import os

from miditok import REMI, TokenizerConfig
from pathlib import Path
from config import MIDI_DIR, TOKENIZER_PATH, TOKENIZER_PARAMS
import logging


def create_and_train_tokenizer():
    config = TokenizerConfig(**TOKENIZER_PARAMS)
    tokenizer = REMI(config)
    # print(os.path)
    if not os.path.exists(MIDI_DIR):
        raise ValueError(f"MIDI directory does not exist: {MIDI_DIR}")

    files_paths = list(MIDI_DIR.glob("*.mid"))
    if not files_paths:
        raise ValueError(f"No MIDI files found in {MIDI_DIR}")

    logging.info(f"Training tokenizer on {len(files_paths)} MIDI files")

    try:
        # tokenizer.train(vocab_size=TOKENIZER_PARAMS["vocab_size"], files_paths=files_paths)
        midi_paths = list(MIDI_DIR.resolve().glob("**/*.mid")) + list(MIDI_DIR.resolve().glob("**/*.midi"))

        tokenizer.train(
            vocab_size=30000,
            files_paths=midi_paths,
        )
        tokenizer.save_params(TOKENIZER_PATH)
        logging.info(f"Tokenizer trained and saved to {TOKENIZER_PATH}")
        return tokenizer
    except Exception as e:
        logging.error(f"Error in create_and_train_tokenizer: {str(e)}")
        # Print the first few file paths for debugging
        logging.error(f"First few file paths: {files_paths[:5]}")
        raise


def load_tokenizer():
    return REMI.from_file(TOKENIZER_PATH)
