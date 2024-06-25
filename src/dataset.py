import os
import logging

from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from config import MIDI_DIR, OUTPUT_DIR
from src.split_files_util import split_files_for_training

logging.basicConfig(level=logging.INFO)


def prepare_dataset(tokenizer, max_seq_len=1024):
    dataset_chunks_dir = OUTPUT_DIR / "midi_chunks"

    # Create the midi_chunks directory if it doesn't exist
    os.makedirs(dataset_chunks_dir, exist_ok=True)

    files_paths = list(MIDI_DIR.glob("**/*.mid"))

    if not files_paths:
        raise ValueError(f"No MIDI files found in {MIDI_DIR}")

    logging.info(f"Found {len(files_paths)} MIDI files")

    try:
        # Use split_files_for_training
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=dataset_chunks_dir,
            max_seq_len=max_seq_len,
            num_overlap_bars=2  # You can adjust this value
        )
        # augment_dataset(
        #     dataset_chunks_dir,
        #     pitch_offsets=[-12, 12],
        #     velocity_offsets=[-4, 4],
        #     duration_offsets=[-0.5, 0.5],
        # )

        # Verify the chunks created
        split_files_paths = list(dataset_chunks_dir.glob("**/*.mid"))
        if not split_files_paths:
            raise ValueError(f"No split files were created in {dataset_chunks_dir}")
        logging.info(f"Successfully split files. Created {len(split_files_paths)} chunks.")

        # Debug: List the first few split files
        for fpath in split_files_paths[:5]:
            logging.debug(f"Split file created: {fpath}")

        kwargs_dataset = {"max_seq_len": 1024, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"],
                          "eos_token_id": tokenizer["EOS_None"]}
        dataset = DatasetMIDI(split_files_paths, **kwargs_dataset)
        print("dataset_MIDI", dataset)
        logging.info(f"Dataset initialized with {len(dataset)} files.")
        return dataset
    except Exception as e:
        logging.error(f"Error in prepare_dataset: {str(e)}")
        # Print the first few file paths for debugging
        logging.error(f"First few file paths: {files_paths[:5]}")
        raise


def get_dataloader(dataset, tokenizer, batch_size=64):
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
