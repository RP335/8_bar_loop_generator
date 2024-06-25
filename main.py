from src.tokenizer import create_and_train_tokenizer, load_tokenizer
from src.dataset import prepare_dataset, get_dataloader
from src.model import create_model
from src.train import  train
from src.generate import generate_midi
from config import OUTPUT_DIR


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create and train tokenizer
    tokenizer = create_and_train_tokenizer()

    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    print("dataset_101", dataset)
    # train_dataloader = get_dataloader(dataset, tokenizer)

    # Create model
    model = create_model(tokenizer)

    # Train model
    trainer = train(model, dataset, dataset, tokenizer)  # Using same dataset for train and eval

    # Generate MIDI

    generate_midi(trainer.model, tokenizer)


if __name__ == "__main__":
    main()


