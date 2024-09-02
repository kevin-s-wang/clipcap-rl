import torch
from torch.utils.data import Dataset
from typing import Optional, List, Any
from transformers import AutoTokenizer
import pandas as pd

def check_splits(split: str) -> None:
    allowed_splits = ('train', 'val', 'test')
    if split not in allowed_splits:
        raise ValueError(f"Allowed splits are: {','.join(allowed_splits)}")
    
class ImageCaptionDataset(Dataset):
    def __init__(self, 
        max_length: int, 
        prefix_length: int, 
        tokenizer: AutoTokenizer, 
        split: str = 'train', 
        data_dir: Optional[str] = None,
        sample_frac: Optional[float] = None) -> None:

        super().__init__()
        
        check_splits(split)

        self.split = split
        self.prefix_length = prefix_length
        self.max_length = max_length
        self.data_dir = data_dir
        self.image_embeddings: Optional[torch.Tensor] = None
        self.caption_embeddings: Optional[torch.Tensor] = None
        self.metadata: Optional[List] = None
        self.tokenizer = tokenizer
        self.sample_frac = sample_frac

        self._load_data()

    def is_train(self) -> bool:
        return self.split == 'train'
    
    def is_validation(self) -> bool:
        return self.split == 'val'
    
    def is_test(self) -> bool:
        return self.split == 'test'

    def _load_data(self) -> None:
        self.data = pd.read_parquet(self.data_dir, engine='pyarrow', filters=[('split', '=', self.split)])

        # Sampling for faster training, especially for debugging and evaluation
        if self.sample_frac is not None:
            self.data = self.data.sample(frac=self.sample_frac)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Any:
        row = self.data.iloc[index]
        
        caption_tokens = self.tokenizer(row['caption'], 
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.max_length, 
                                        return_tensors='pt')
        
        
        image_embeddings = torch.tensor(row['image_embeddings'])
        caption_embeddings = torch.tensor(row['caption_embeddings'])
        
        input_ids = caption_tokens['input_ids']
        mask = caption_tokens['attention_mask']

        mask = mask.float().squeeze(0)
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        
        return  image_embeddings, caption_embeddings, input_ids, mask