from transformers import GenerationConfig
from config import GENERATION_CONFIG, OUTPUT_DIR


def generate_midi(model, tokenizer, prompt=None):
    generation_config = GenerationConfig(**GENERATION_CONFIG)

    if prompt is None:
        prompt = tokenizer.bos_token_id

    generated = model.generate(
        input_ids=prompt,
        generation_config=generation_config,
    )

    midi = tokenizer.decode(generated[0].tolist())
    output_path = OUTPUT_DIR / "generated_midi.mid"
    midi.dump_midi(output_path)
    print(f"Generated MIDI saved to {output_path}")
