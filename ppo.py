from transformers import (
    HfArgumentParser,
    DefaultDataCollator,
)
from trl.trainer import PPOTrainer, PPOConfig
from data import ImageCaptionDataset
from configuration_clipcap_rl import ClipCapRLConfig
from modeling_clipcap_rl import ClipCapRLModel
from dataclasses import dataclass
from typing import Optional
import open_clip
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Tuple, Dict, Any


clip_model_name = 'ViT-B-32'
clip_pretrained = 'laion2b_s34b_b79k'

class ImageCaptionDataCollator(DefaultDataCollator):
    def __call__(self, features: List[Tuple]) -> Dict[str, Any]:
        _features = {
            'image_embeddings': torch.stack([f[0] for f in features]),
            # 'caption_embeddings': [],
            'tokens': torch.stack([f[2] for f in features]),
            'mask': torch.stack([f[3] for f in features]),
        }

        return _features

def reward(model, image_features: torch.FloatTensor, caption):
    with torch.no_grad():
        text_features = model.encode_text(caption)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = F.cosine_similarity(image_features, text_features)
        return similarity

def reward1(model, image_features: torch.FloatTensor, referece, candidate) -> torch.IntTensor:
    with torch.no_grad():
        referece_features = model.encode_text(referece)
        referece_features /= referece_features.norm(dim=-1, keepdim=True)
        
        candidate_features = model.encode_text(candidate)
        candidate_features /= candidate_features.norm(dim=-1, keepdim=True)

        reference_similarity = F.cosine_similarity(image_features, referece_features)
        candidate_similarity = F.cosine_similarity(image_features, candidate_features)
        return torch.sign(candidate_similarity - reference_similarity).int()

def reward2(model, image_features: torch.FloatTensor, referece, candidate) -> torch.FloatTensor:
    with torch.no_grad():
        referece_features = model.encode_text(referece)
        referece_features /= referece_features.norm(dim=-1, keepdim=True)
        
        candidate_features = model.encode_text(candidate)
        candidate_features /= candidate_features.norm(dim=-1, keepdim=True)

        reference_similarity = F.cosine_similarity(image_features, referece_features)
        candidate_similarity = F.cosine_similarity(image_features, candidate_features)

        return candidate_similarity - reference_similarity


@dataclass
class ScriptArguments:
    data_dir: Optional[str] = "data"

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
   
    config = ClipCapRLConfig.from_pretrained('out', local_files_only=True) 
    model = ClipCapRLModel.from_pretrained('out', local_files_only=True, config=config)

    train_dataset = ImageCaptionDataset(
        max_length=config.max_length,
        prefix_length=config.prefix_length,
        tokenizer=model.tokenizer,
        split="train",
        sample_frac=0.01,
        data_dir=args.data_dir)
    
    ppo_config = PPOConfig()

    trainer = PPOTrainer(
        config=ppo_config,
        tokenizer=model.tokenizer, 
        train_dataset=train_dataset,
        data_collator=ImageCaptionDataCollator(),
    )
    

    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained, device=trainer.device)
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    from tqdm import tqdm


    epochs = 10
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(trainer.dataloader):

            # CLIP embeddings of images
            image_embeddings = batch["image_embeddings"]
            
            # Reference captions
            references = batch["tokens"]
        
            # Generate candidate captions
            candidates = trainer.model.generate(image_embeddings=image_embeddings)
            candidate_texts = [model.tokenizer.decode(r.squeeze()) for r in candidates]
        
            #### Compute reward scores
            rewards = [reward2(model, image_embeddings[i], references[i], candidates[i]) for i in range(len(references))]
        
            #### Run PPO step
            stats = trainer.step(query_tensors, candidates, rewards)
            trainer.log_stats(stats, batch, rewards)

    #### Save model
    trainer.save_pretrained("clipcap-rl")


    device = 'mps'

    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained, device=device)
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    import pandas as pd
    sample = pd.read_parquet('.cache/prepare/data', engine='pyarrow').iloc[0]
    
    image_embeddings = torch.from_numpy(sample['image_embeddings']).to(device)
    print('sample: ', sample['caption'])
    candidate = clip_tokenizer([sample['caption']]).to(device)
    reference = clip_tokenizer(['man with red helmet on a dirt road.']).to(device)
    candidate = reference
    print('Reward: ', reward1(clip_model, image_embeddings, reference, candidate))