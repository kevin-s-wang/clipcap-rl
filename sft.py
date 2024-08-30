from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

from datetime import datetime
from data import ImageCaptionDataset
from model import ClipCapModel, ModelConfig


if __name__ == "__main__":
    conf = ModelConfig(data_dir="data")

    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=f"phi3.5-clipcap-QLoRA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        output_dir="./",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        num_train_epochs=2,
        save_steps=1,
        logging_steps=1,
        warmup_steps=5,
        ddp_find_unused_parameters=False,
    )

    model = ClipCapModel(conf)
    
    train_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="train",
        data_dir=conf.data_dir)
    
    eval_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="val",
        data_dir=conf.data_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="",  # need a dummy field
        tokenizer=model.tokenizer,
        data_collator=DefaultDataCollator(),
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)