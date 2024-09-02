import shutil
from accelerate import PartialState
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from data import ImageCaptionDataset
from configuration_clipcap_rl import ClipCapRLConfig
from modeling_clipcap_rl import ClipCapRLModel
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScriptArguments:
    data_dir: Optional[str] = "data"

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
    # remove output_dir if exists

    config = ClipCapRLConfig.from_pretrained('out', local_files_only=True) 
    model = ClipCapRLModel.from_pretrained('out', local_files_only=True, config=config)

    value_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )

    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )

    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )

    train_dataset = ImageCaptionDataset(
        max_length=config.max_length,
        prefix_length=config.prefix_length,
        tokenizer=model.tokenizer,
        split="train",
        data_dir=args.data_dir)
    
    eval_dataset = ImageCaptionDataset(
        max_length=config.max_length,
        prefix_length=config.prefix_length,
        tokenizer=model.tokenizer,
        split="val",
        data_dir=args.data_dir)

    trainer = PPOv2Trainer(
        config=config,
        tokenizer=model.tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)