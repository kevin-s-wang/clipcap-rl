from modeling_clipcap_rl import ClipCapRLModel
from configuration_clipcap_rl import ClipCapRLConfig

import pandas as pd
import torch

@dataclass
class ScriptArguments:
    # Model name or path
    model_id: str
    data_dir: Optional[str] = "data"
    num_generations: Optional[int] = 1
    temperature: Optional[float] = 1.0

if __name__ == '__main__':

    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    config = ClipCapRLConfig.from_pretrained(args.model_id, local_files_only=True) 
    model = ClipCapRLModel.from_pretrained(args.model_id, local_files_only=True, config=config)
    
    model.eval()

    sample = pd.read_parquet(args.data_dir', 
                    engine='pyarrow', filters=[('split', '=', 'test')]) \
                .sample(1).iloc[0]
    
    print('Sample caption: ', sample['caption'])
    print('Image path: ', sample['filepath'])
    
    print('Generated captions:\n')
    image_embeddings = torch.from_numpy(sample['image_embeddings'])
    for seq in range(num_generations):
        tokens = model.generate(image_embeddings, temperature=temperature)
        print(f'[{seq}]', model.tokenizer.decode(tokens[0], skip_special_tokens=True))
