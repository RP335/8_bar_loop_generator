from miditok.pytorch_data import DataCollator
from transformers import Trainer, TrainingArguments
from evaluate import load as load_metric
from torch import Tensor, argmax
from config import TRAINING_CONFIG, OUTPUT_DIR
from torch.cuda import is_available as cuda_available, is_bf16_supported
from torch.backends.mps import is_available as mps_available

metrics = {metric: load_metric(metric) for metric in ["accuracy"]}


def compute_metrics(eval_pred):
    """
    Compute metrics for pretraining.

    Must use preprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != -100
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())


def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """
    Preprocess the logits before accumulating them during evaluation.

    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids


def train(model, traindataset, eval_dataset, tokenizer):
    # Create config for the Trainer
    USE_CUDA = cuda_available()
    if not cuda_available():
        FP16 = FP16_EVAL = BF16 = BF16_EVAL = False
    elif is_bf16_supported():
        BF16 = BF16_EVAL = True
        FP16 = FP16_EVAL = False,
    else:
        BF16 = BF16_EVAL = False
        FP16 = FP16_EVAL = True
    USE_MPS = not USE_CUDA and mps_available()
    training_config = TrainingArguments(
        "runs", False, True, True, False, "steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=3,
        eval_accumulation_steps=None,
        eval_steps=1000,
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=3.0,
        max_steps=20000,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.3,
        log_level="debug",
        logging_strategy="steps",
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=5,
        no_cuda=not USE_CUDA,
        seed=444,
        fp16=FP16,
        fp16_full_eval=FP16_EVAL,
        bf16=BF16,
        bf16_full_eval=BF16_EVAL,
        load_best_model_at_end=True,
        label_smoothing_factor=0.,
        optim="adamw_torch",
        report_to=["tensorboard"],
        gradient_checkpointing=True,
    )

    collator = DataCollator(tokenizer["PAD_None"], copy_inputs_as_labels=True)
    trainer = Trainer(
        model=model,
        args=training_config,
        data_collator=collator,
        train_dataset=traindataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=None,
        preprocess_logits_for_metrics=preprocess_logits,
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
