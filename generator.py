from miditok.pytorch_data import DatasetMIDI

from config import OUTPUT_DIR
from src import tokenizer
from src.generate import generate_midi
from src.tokenizer import create_and_train_tokenizer

dataset_chunks_dir = OUTPUT_DIR / "midi_chunks"
split_files_paths = list(dataset_chunks_dir.glob("**/*.mid"))
tokenizer = create_and_train_tokenizer()

kwargs_dataset = {"max_seq_len": 1024, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"],
                  "eos_token_id": tokenizer["EOS_None"]}
dataset_test = DatasetMIDI(split_files_paths, **kwargs_dataset)
generate_midi(dataset_test, tokenizer)