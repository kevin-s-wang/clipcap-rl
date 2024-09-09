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

# python eval.py --data_dir ../mirs/.cache/prepare/merged --num_generations 5 --model_id ../clipcap-gpt2-medium --temperature 1.2

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
cider = evaluate.load("./metrics/CIDEr")

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
                    engine='pyarrow', filters=[('split', '=', 'test')]).sample(1000)

    dataset_names = samples['dataset_name'].unique()
    print('dataset_names:', dataset_names)
    results = {}
    for dataset_name in dataset_names:
        print('Evaluating dataset:', dataset_name)
        results[dataset_name] = {}

        dataset_specific = samples[samples.dataset_name == dataset_name]

        for _, sample in tqdm(dataset_specific.iterrows(), total=dataset_specific.shape[0]):

            if sample['filepath'] in results[dataset_name]:
                results[dataset_name][sample['filepath']]['references'].append(sample['caption'])
                continue

            print('Image:', sample['filepath'])
            print('Reference caption:', sample['caption'])
            print('Generated captions:')
            generated_captions = []
            image_embeddings = torch.from_numpy(sample['image_embeddings']).to(device)
            for seq in range(args.num_generations):
                tokens = model.generate(image_embeddings, temperature=args.temperature)
                generated_caption = model.tokenizer.decode(tokens[0], skip_special_tokens=True)
                generated_captions.append(generated_caption)
                print(f'{seq+1}. ', generated_caption)
        
            print()
            prediction = generated_captions[0]

            

            if args.num_generations > 1:
                text = clip_tokenizer(generated_captions).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_embeddings @ text_features.T).softmax(dim=-1)
                # print("CLIP scores:", text_probs)
                prediction = generated_captions[text_probs.argmax().item()]

            results[dataset_name][sample['filepath']] = {
                'prediction': prediction,
                'references': [sample['caption']],
            }
                
            # results[dataset_name]["predictions"].append(prediction)
            # results[dataset_name]["references"].append(sample['caption'])
            # results[dataset_name]["filepaths"].append(sample['filepath'])
    
    # saving it
    json.dump(results, open("results.json", "w"))
    for dataset_name in dataset_names:
        predictions = []
        references = []
        for _, val in results[dataset_name].items():
            predictions.append(val['prediction'])
            references.append(val['references'])
        
        metrics = {}
        # metrics.update(bleu.compute(predictions=predictions, references=references))
        metrics.update(rouge.compute(predictions=predictions, references=references))
        metrics.update(meteor.compute(predictions=predictions, references=references))
        metrics.update(cider.compute(predictions=predictions, references=references))
        print(f"Dataset: {dataset_name}")
        print(metrics)
        print()
