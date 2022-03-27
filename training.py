from typing import Any, Dict, List

import torch.multiprocessing as tmp
from datasets import load_dataset
from torch.multiprocessing import Queue
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState


class TuneReporterCallback(TrainerCallback):
    def __init__(self, result_queue: Queue):
        self.result_queue = result_queue

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.result_queue.put(state.log_history[-1])


def _launch_mp(
    idx: int, config: Dict[str, Any], num_cores: int = 1, result_queue: Queue = None
):
    tokenizer = AutoTokenizer.from_pretrained("sgugger/gpt2-like-tokenizer")
    block_size = tokenizer.model_max_length

    def tokenize_fn(data):
        return tokenizer(data["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        key, *_ = examples.keys()
        total = (len(concatenated[key]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    model_config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(model_config)

    lm_datasets = (
        load_dataset("wikitext", "wikitext-2-raw-v1")
        .map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])
        .map(group_texts, batched=True, batch_size=512, num_proc=4)
    )

    training_args = TrainingArguments(
        output_dir=f"gpt2-wikitext2",
        evaluation_strategy="epoch",
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_ratio=config.get("warmup_ratio", 0),
        weight_decay=config.get("weight_decay", 0),
        num_train_epochs=config.get("num_train_epochs", 2),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        eval_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        push_to_hub=False,
        save_strategy="no",
        tpu_num_cores=num_cores,
    )

    callbacks: List[TrainerCallback] = []
    if idx == 0 and result_queue is not None:
        callbacks.append(TuneReporterCallback(result_queue))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=callbacks,
    )
    trainer.train()
