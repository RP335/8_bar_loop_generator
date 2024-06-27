from copy import deepcopy
from pathlib import Path
import torch
from datasets import tqdm
from miditok.pytorch_data import DataCollator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from config import OUTPUT_DIR
from src.tokenizer import create_and_train_tokenizer

(gen_results_path := Path('gen_res')).mkdir(parents=True, exist_ok=True)

gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

music_tokenizer = create_and_train_tokenizer()


def text_to_music_tokens(prompt, gpt2_tokenizer, music_tokenizer):
    # Encode the text prompt into tokens using GPT-2 tokenizer
    gpt2_tokens = gpt2_tokenizer.encode(prompt, return_tensors='pt')

    music_tokens = []
    for token in gpt2_tokens[0]:
        if token < len(music_tokenizer):
            music_tokens.append(token.item())
        else:
            music_tokens.append(music_tokenizer.pad_token_id)
    return torch.tensor([music_tokens], dtype=torch.long)


def generate_midi_from_text(prompt):
    input_tokens = text_to_music_tokens(prompt, gpt2_tokenizer, music_tokenizer)

    model_path = Path("runs")  # Ensure this is the correct path to your saved model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=200,  # extends samples by 200 tokens
        num_beams=1,  # no beam search
        do_sample=True,  # but sample instead
        temperature=0.9,
        top_k=15,
        top_p=0.95,
        epsilon_cutoff=3e-4,
        eta_cutoff=1e-3,
        pad_token_id=music_tokenizer.pad_token_id,
    )

    # Generate 
    res = model.generate(
        inputs=input_tokens.to(model.device),
        generation_config=generation_config
    )

    generated = res[0].cpu().tolist()
    midi = music_tokenizer.decode([deepcopy(generated)])

    midi_path = gen_results_path / 'generated.mid'
    midi.dump_midi(midi_path)

    tokens_path = gen_results_path / 'generated.json'
    music_tokenizer.save_tokens([generated], tokens_path)

    print(f"Generated MIDI saved to {midi_path}")
    print(f"Generated tokens saved to {tokens_path}")


generate_midi_from_text("Create a smooth jazz melody with a lively rhythm.")
