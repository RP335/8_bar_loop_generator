from transformers import AutoModelForCausalLM, MistralConfig
from config import MODEL_CONFIG


def create_model(tokenizer):
    model_config = MistralConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=2048,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        sliding_window=256,
        max_position_embeddings=8192,
        pad_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
    )
    return AutoModelForCausalLM.from_config(model_config)
