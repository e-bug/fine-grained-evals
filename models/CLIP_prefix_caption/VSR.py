import argparse
import os

import time
import datetime
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import torch
from torch.nn import functional as nnf

import clip
from transformers import GPT2Tokenizer

from model_utils import ClipCaptionModel, pad_tokens


@torch.no_grad()
def evaluation(model, clip_model, clip_preprocess, prefix_length, data, tokenizer, device, data_dir):
    model.eval()
   
    start_time = time.time()
    itm_scores = []
    for example_id, example in tqdm(enumerate(data)):
        image_path = os.path.join(data_dir, 'images', example['image'])
        image = Image.open(image_path).convert('RGB')
        image = clip_preprocess(image).unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        
        caption = example['caption']
        tokens, mask = pad_tokens(caption, tokenizer, prefix_length)
        tokens, mask = tokens.to(device), mask.to(device)

        outputs = model(tokens.unsqueeze(0), prefix, mask)
        logits = outputs.logits[:, prefix_length-1: -1]
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
        itm_score = 1 / loss.item()

        itm_scores += [
            {'label': example['label'], 'score': itm_score, 'relation': example['relation'], 'id': example_id}, 
        ]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return itm_scores


def main(args):
    device = torch.device(args.device)

    print("Creating model", flush=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Creating VSR benchmark", flush=True)
    with jsonlines.open(f'{args.data_dir}/annotations/{args.data_variant}/dev.jsonl') as f:
        val_data = [i for i in f]
    with jsonlines.open(f'{args.data_dir}/annotations/{args.data_variant}/test.jsonl') as f:
        test_data = [i for i in f]
    print("Start evaluating", flush=True)
    val_scores = evaluation(model, clip_model, preprocess, prefix_length,
                            val_data, tokenizer, device, args.data_dir)
    test_scores = evaluation(model, clip_model, preprocess, prefix_length,
                             test_data, tokenizer, device, args.data_dir)

    with open(os.path.join(args.output_dir, f'{args.output_name}_dev.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, val_scores)))
    with open(os.path.join(args.output_dir, f'{args.output_name}_test.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_variant', type=str, default='random')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='clipcap')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
