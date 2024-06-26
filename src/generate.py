from copy import deepcopy
from pathlib import Path

from datasets import tqdm
from miditok.pytorch_data import DataCollator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GenerationConfig




# Here the sequences are padded to the left, so that the last token along the time dimension
# is always the last token of each seq, allowing to efficiently generate by batch
def generate_midi(dataset_test, tokenizer):
    (gen_results_path := Path('gen_res')).mkdir(parents=True, exist_ok=True)
    generation_config = GenerationConfig(
        max_new_tokens=200,  # extends samples by 200 tokens
        num_beams=1,  # no beam search
        do_sample=True,  # but sample instead
        temperature=0.7,
        top_k=15,
        top_p=0.95,
        epsilon_cutoff=3e-4,
        eta_cutoff=1e-3,
        pad_token_id=0,
    )
    collator = DataCollator(0, copy_inputs_as_labels=True)
    collator.pad_on_left = True
    collator.eos_token = None
    dataloader_test = DataLoader(dataset_test, batch_size=16, collate_fn=collator)

    # Load the trained model from the saved directory
    model_path = Path("runs")  # Ensure this is the correct path to your saved model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    count = 0
    for batch in tqdm(dataloader_test, desc='Testing model / Generating results'):  # (N,T)
        res = model.generate(
            inputs=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            generation_config=generation_config
        )  # (N,T)

        # Saves the generated music, as MIDI files and tokens (json)
        for prompt, continuation in zip(batch["input_ids"], res):
            generated = continuation[len(prompt):]
            midi = tokenizer.decode([deepcopy(generated.tolist())])
            tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
            tokens = [seq.tolist() for seq in tokens]
            for tok_seq in tokens[1:]:
                _midi = tokenizer.decode([deepcopy(tok_seq)])
                midi.tracks.append(_midi.tracks[0])
            midi.tracks[0].name = f'Continuation of original sample ({len(generated)} tokens)'
            midi.tracks[1].name = f'Original sample ({len(prompt)} tokens)'
            midi.tracks[2].name = f'Original sample and continuation'
            midi.dump_midi(gen_results_path / f'{count}.mid')
            tokenizer.save_tokens(tokens, gen_results_path / f'{count}.json')

            count += 1
