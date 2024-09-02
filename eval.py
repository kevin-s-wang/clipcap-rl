from modeling_clipcap_rl import ClipCapRLModel
from configuration_clipcap_rl import ClipCapRLConfig

import pandas as pd
import torch


if __name__ == '__main__':
    config = ClipCapRLConfig.from_pretrained('out', local_files_only=True) 
    model = ClipCapRLModel.from_pretrained('out', local_files_only=True, config=config)
    
    model.eval()

    sample = pd.read_parquet('.cache/prepare/data', 
                    engine='pyarrow') \
                .sample(1).iloc[0]


    print('sample: ', sample['caption'])
    print('image: ', sample['filepath'])
    tokens = model.generate(torch.from_numpy(sample['image_embeddings']), temperature=0.5)
    print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))
