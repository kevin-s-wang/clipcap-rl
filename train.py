from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from datetime import datetime
from dataclasses import dataclass
from data import ImageCaptionDataset
from configuration_clipcap_rl import ClipCapRLConfig
from modeling_clipcap_rl import ClipCapRLModel
from typing import Optional, List, Dict, Any
from transformers import HfArgumentParser


class ImageCaptionDataCollator(DefaultDataCollator):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        print(len(features))
        return features

@dataclass
class ScriptArguments:
    data_dir: Optional[str] = "data"

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=f"phi3.5-clipcap-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        output_dir="./",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=50,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        num_train_epochs=2,
        warmup_steps=5,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
    )

    conf = ClipCapRLConfig()
    model = ClipCapRLModel(conf)
    
    train_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="train",
        data_dir=args.data_dir)
    
    eval_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="val",
        data_dir=args.data_dir)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
        data_collator=ImageCaptionDataCollator(),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)