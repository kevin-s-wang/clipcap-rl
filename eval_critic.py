from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
from modeling_clipcap_rl import ClipCapRLModel
from configuration_clipcap_rl import ClipCapRLConfig
import open_clip
import json
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from typing import List

# python eval.py --data_dir ../mirs/.cache/prepare/merged --num_generations 5 --model_id ../clipcap-gpt2-medium --temperature 1.2

# bleu = evaluate.load("sacrebleu")
# rouge = evaluate.load("rouge")
# meteor = evaluate.load("meteor")

clip_model_name = 'ViT-B-32'
clip_pretrained = 'laion2b_s34b_b79k'

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else \
             "cpu"
    return torch.device(device)

@dataclass
class ScriptArguments:
    # Model name or path
    model_id: str
    data_dir: Optional[str] = "data"
    num_generations: Optional[int] = 1
    temperature: Optional[float] = 1.0

def critic(image_embeddings: torch.Tensor, candidates: List[str]):
    texts = clip_tokenizer(candidates).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_embeddings @ text_features.T).softmax(dim=-1)
    return text_probs

if __name__ == '__main__':

    device = get_device()

    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    config = ClipCapRLConfig.from_pretrained(args.model_id, local_files_only=True)
    model = ClipCapRLModel.from_pretrained(args.model_id, local_files_only=True, config=config).to(device)
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained, device=device)
    clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

    model.eval()

    samples = pd.read_parquet(args.data_dir, 
                    engine='pyarrow', filters=[('split', '=', 'test')]).sample(20)

    dataset_names = samples['dataset_name'].unique()
    print('dataset_names:', dataset_names)
    results = {}
    for dataset_name in dataset_names:
        print('Evaluating dataset:', dataset_name)
        results[dataset_name] = {
            "predictions": [],
            "references": [],
            "filepaths": [],
        }

        dataset_specific = samples[samples.dataset_name == dataset_name]

        for _, sample in tqdm(dataset_specific.iterrows(), total=dataset_specific.shape[0]):
            print('Image:', sample['filepath'])
            print('Reference caption:', sample['caption'])
                 
            image_embeddings = torch.from_numpy(sample['image_embeddings']).to(device)

            print('Predicted captions:')
            for seq in range(args.num_generations):
                tokens = model.generate_with_critic(critic, image_embeddings, temperature=args.temperature)
                prediction = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
                print(f'{seq+1}. ', prediction)
            print()

            results[dataset_name]["predictions"].append(prediction)
            results[dataset_name]["references"].append(sample['caption'])
            results[dataset_name]["filepaths"].append(sample['filepath'])
    
    # saving it
    # json.dump(results, open("critic_results.json", "w"))
    # for dataset_name in dataset_names:
    #     metrics = {}
    #     metrics.update(bleu.compute(predictions=results[dataset_name]["predictions"], references=results[dataset_name]["references"]))
    #     metrics.update(rouge.compute(predictions=results[dataset_name]["predictions"], references=results[dataset_name]["references"]))
    #     metrics.update(meteor.compute(predictions=results[dataset_name]["predictions"], references=results[dataset_name]["references"]))
    #     print(f"Dataset: {dataset_name}")
    #     print(metrics)
    #     print()
