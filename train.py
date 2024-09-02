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
import evaluate

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def get_compute_metrics_fn(tokenizer, prefix_length):
    def compute_metrics(pred):

        label_ids = torch.from_numpy(pred.label_ids).squeeze(1)
        predictions = pred.predictions[:, prefix_length-1: -1]
        pred_ids = torch.argmax(torch.from_numpy(predictions), dim=-1)

        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        metrics = {}
        metrics.update(bleu.compute(predictions= pred_str, references=label_str))
        metrics.update(rouge.compute(predictions=pred_str, references=label_str))
        metrics.update(meteor.compute(predictions=pred_str, references=label_str))
        return metrics
    return compute_metrics

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
        report_to="tensorboard",
        run_name=f"gpt2-clipcap-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
        output_dir="./clipcap-gpt2-medium",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=4,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=4,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        logging_steps=10,
        remove_unused_columns=False,
        optim="adamw_torch",
        bf16=True,
        learning_rate=2e-5,
        label_names=["tokens"],
        lr_scheduler_type="linear",
        save_safetensors=False,
        num_train_epochs=10,
        warmup_steps=5,
        load_best_model_at_end=True,
    )

    conf = ClipCapRLConfig(
                use_lora=True, 
                prefix_length=20,
                max_length=50,
            )
    model = ClipCapRLModel(conf)
    
    sample_frac = 0.01

    train_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="train",
        sample_frac=0.1,
        data_dir=args.data_dir)
    
    eval_dataset = ImageCaptionDataset(
        max_length=conf.max_length,
        prefix_length=conf.prefix_length,
        tokenizer=model.tokenizer,
        split="val",
        sample_frac=sample_frac,
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


    model.print_trainable_parameters()

    trainer = ClipCapRLTrainer(
        prefix_length=conf.prefix_length,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer,
        
        compute_metrics=get_compute_metrics_fn(model.tokenizer, conf.prefix_length),
        data_collator=ImageCaptionDataCollator(),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)