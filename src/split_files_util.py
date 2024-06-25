import logging
from pathlib import Path
from collections.abc import Sequence
from miditok import MusicTokenizer
from miditok.constants import SUPPORTED_MUSIC_FILE_EXTENSIONS, MAX_NUM_FILES_NUM_TOKENS_PER_NOTE, \
    SCORE_LOADING_EXCEPTION, MIDI_FILES_EXTENSIONS, TIME_SIGNATURE
from miditok.pytorch_data import get_average_num_tokens_per_note, split_score_per_note_density
from miditok.utils.utils import get_deepest_common_subdir, split_score_per_tracks
from symusic import Score, TextMeta, TimeSignature
from symusic.core import TimeSignatureTickList
from tqdm import tqdm


def split_files_for_training(
        files_paths: Sequence[Path],
        tokenizer: MusicTokenizer,
        save_dir: Path,
        max_seq_len: int,
        average_num_tokens_per_note: float | None = None,
        num_overlap_bars: int = 1,
        min_seq_len: int | None = None,
) -> list[Path]:
    """
    Split a list of music files into smaller chunks to use for training.

    :param files_paths: paths to music files to split.
    :param tokenizer: tokenizer.
    :param save_dir: path to the directory to save the files splits.
    :param max_seq_len: maximum token sequence length that the model will be trained with.
    :param average_num_tokens_per_note: average number of tokens per note associated to
        this tokenizer. If given ``None``, this value will automatically be calculated
        from the first 200 files with the
        :py:func:`miditok.pytorch_data.get_average_num_tokens_per_note` method.
    :param num_overlap_bars: will create chunks with consecutive overlapping bars. For
        example, if this argument is given ``1``, two consecutive chunks might end at
        the bar *n* and start at the bar *n-1* respectively, thus they will encompass
        the same bar. This allows to create a causality chain between chunks. This value
        should be determined based on the ``average_num_tokens_per_note`` value of the
        tokenizer and the ``max_seq_len`` value, so that it is neither too high nor too
        low. (default: ``1``).
    :param min_seq_len: minimum sequence length, only used when splitting at the last
        bar of the file. (default: ``None``, see default value of
        :py:func:`miditok.pytorch_data.split_score_per_note_density`)
    :return: the paths to the files splits.
    """
    logging.info(f"Starting to split files for training, saving to {save_dir}")

    # Safety checks
    split_hidden_file_path = save_dir / f".{hash(tuple(files_paths))}"
    if split_hidden_file_path.is_file():
        warn(
            f"These files have already been split in the saving directory ({save_dir})."
            f" Skipping file splitting.",
            stacklevel=2,
        )
        return [
            path
            for path in save_dir.glob("**/*")
            if path.suffix in SUPPORTED_MUSIC_FILE_EXTENSIONS
        ]

    if not average_num_tokens_per_note:
        average_num_tokens_per_note = get_average_num_tokens_per_note(
            tokenizer, files_paths[:MAX_NUM_FILES_NUM_TOKENS_PER_NOTE]
        )
        logging.info(f"Calculated average number of tokens per note: {average_num_tokens_per_note}")

    # Determine the deepest common subdirectory to replicate file tree
    root_dir = get_deepest_common_subdir(files_paths)

    # Splitting files
    new_files_paths = []
    for file_path in tqdm(
            files_paths,
            desc=f"Splitting music files ({save_dir})",
            miniters=int(len(files_paths) / 20),
            maxinterval=480,
    ):
        logging.info(f"Processing file: {file_path}")
        try:
            scores = [Score(file_path)]
        except SCORE_LOADING_EXCEPTION:
            logging.warning(f"Skipping file due to loading exception: {file_path}")
            continue

        # First preprocess time signatures to avoid cases where they might cause errors
        _preprocess_time_signatures(scores[0], tokenizer)

        # Separate track if needed
        tracks_separated = False
        if not tokenizer.one_token_stream and len(scores[0].tracks) > 1:
            scores = split_score_per_tracks(scores[0])
            tracks_separated = True

        # Split per note density
        for ti, score_to_split in enumerate(scores):
            score_chunks = split_score_per_note_density(
                score_to_split,
                max_seq_len,
                average_num_tokens_per_note,
                num_overlap_bars,
                min_seq_len,
            )

            # Save them
            for _i, chunk_to_save in enumerate(score_chunks):
                # Skip it if there are no notes, this can happen with
                # portions of tracks with no notes but tempo/signature
                # changes happening later
                if len(chunk_to_save.tracks) == 0 or chunk_to_save.note_num() == 0:
                    continue

                # Add a marker to indicate chunk number
                chunk_to_save.markers.append(
                    TextMeta(0, f"miditok: chunk {_i}/{len(score_chunks) - 1}")
                )
                if tracks_separated:
                    file_name = f"{file_path.stem}_t{ti}_{_i}{file_path.suffix}"
                else:
                    file_name = f"{file_path.stem}_{_i}{file_path.suffix}"
                # use with_stem when dropping support for python 3.8
                saving_path = (
                        save_dir / file_path.relative_to(root_dir).parent / file_name
                )
                saving_path.parent.mkdir(parents=True, exist_ok=True)
                if file_path.suffix in MIDI_FILES_EXTENSIONS:
                    logging.info(f"Saving chunk to {saving_path}")
                    chunk_to_save.dump_midi(saving_path)
                else:
                    chunk_to_save.dump_abc(saving_path)
                new_files_paths.append(saving_path)

    # Save file in save_dir to indicate file split has been performed
    with split_hidden_file_path.open("w") as f:
        f.write(f"{len(files_paths)} files after file splits")

    logging.info(f"Finished splitting files, total chunks created: {len(new_files_paths)}")
    return new_files_paths


def _preprocess_time_signatures(score: Score, tokenizer: MusicTokenizer) -> None:
    """
    Make sure a Score contains time signature valid according to a tokenizer.

    :param score: ``symusic.Score`` to preprocess the time signature.
    :param tokenizer: :class:`miditok.MusicTokenizer`.
    """
    if tokenizer.config.use_time_signatures:
        tokenizer._filter_unsupported_time_signatures(score.time_signatures)
        if len(score.time_signatures) == 0 or score.time_signatures[0].time != 0:
            score.time_signatures.insert(0, TimeSignature(0, *TIME_SIGNATURE))
    else:
        score.time_signatures = TimeSignatureTickList(
            [TimeSignature(0, *TIME_SIGNATURE)]
        )
