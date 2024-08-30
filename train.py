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
import torch.nn.functional as F
from typing import Tuple
import torch

class ImageCaptionDataCollator(DefaultDataCollator):
    def __call__(self, features: List[Tuple]) -> Dict[str, Any]:
        _features = {
            'image_embeddings': torch.stack([f[0] for f in features]),
            # 'caption_embeddings': [],
            'tokens': torch.stack([f[2] for f in features]),
            'mask': torch.stack([f[3] for f in features]),
        }

        return _features

@dataclass
class ScriptArguments:
    data_dir: Optional[str] = "data"

class ClipCapRLTrainer(Trainer):
    def __init__(self, prefix_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_length = prefix_length
        # self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        # print('inputs: ', inputs)
        # x, _, y, mask = inputs
        outputs = model(**inputs)
        logits = outputs.logits[:, self.prefix_length-1: -1]

        loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.shape[-1]), 
                inputs['tokens'].flatten(),
                ignore_index=self.tokenizer.pad_token_id)
        return (loss, outputs) if return_outputs else loss

# def compute_metrics(preds, labels):
#     return {}

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=f"gpt2-clipcap-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        output_dir="./",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        save_total_limit=3,
        # eval_strategy="steps",
        # eval_steps=50,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        remove_unused_columns=False,
        optim="adamw_torch",
        bf16=True,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        num_train_epochs=2,
        warmup_steps=5,
        # load_best_model_at_end=True,
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


    def compute_loss(model, inputs, return_outputs=False):
        x, _, y, mask = inputs
        outputs = model(x, y, mask)
        logits = outputs.logits[:, conf.prefix_length-1: -1]

        loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.shape[-1]), 
                y.flatten(),
                ignore_index=model.tokenizer.pad_token_id)
        return (loss, outputs) if return_outputs else loss

    trainer = ClipCapRLTrainer(
        prefix_length=conf.prefix_length,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
        data_collator=ImageCaptionDataCollator(),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)