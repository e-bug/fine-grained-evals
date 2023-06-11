import argparse
import os

import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
from torch.nn import functional as nnf

import clip
from datasets import load_dataset
from transformers import GPT2Tokenizer

from model_utils import ClipCaptionModel, pad_tokens


@torch.no_grad()
def evaluation(model, clip_model, clip_preprocess, prefix_length, dset, tokenizer, device):
    model.eval()

    start_time = time.time()
    scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_0, image_1 = example['image_0'].convert('RGB'), example['image_1'].convert('RGB')
        
        image_0 = clip_preprocess(image_0).unsqueeze(0).to(device)
        image_1 = clip_preprocess(image_1).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix_0 = clip_model.encode_image(image_0).to(device, dtype=torch.float32)
            prefix_1 = clip_model.encode_image(image_1).to(device, dtype=torch.float32)
        
        caption_0, caption_1 = example['caption_0'], example['caption_1']
        tokens_0, mask_0 = pad_tokens(caption_0, tokenizer, prefix_length)
        tokens_0, mask_0 = tokens_0.to(device), mask_0.to(device)

        tokens_1, mask_1 = pad_tokens(caption_1, tokenizer, prefix_length)
        tokens_1, mask_1 = tokens_1.to(device), mask_1.to(device)

        outputs_c0_i0 = model(tokens_0.unsqueeze(0), prefix_0, mask_0)
        logits_c0_i0 = outputs_c0_i0.logits[:, prefix_length-1: -1]
        loss_c0_i0 = nnf.cross_entropy(logits_c0_i0.reshape(-1, logits_c0_i0.shape[-1]),
                                       tokens_0.flatten(), ignore_index=0)

        outputs_c0_i1 = model(tokens_0.unsqueeze(0), prefix_1, mask_0)
        logits_c0_i1 = outputs_c0_i1.logits[:, prefix_length-1: -1]
        loss_c0_i1 = nnf.cross_entropy(logits_c0_i1.reshape(-1, logits_c0_i1.shape[-1]),
                                       tokens_0.flatten(), ignore_index=0)

        outputs_c1_i0 = model(tokens_1.unsqueeze(0), prefix_0, mask_1)
        logits_c1_i0 = outputs_c1_i0.logits[:, prefix_length-1: -1]
        loss_c1_i0 = nnf.cross_entropy(logits_c1_i0.reshape(-1, logits_c1_i0.shape[-1]),
                                       tokens_1.flatten(), ignore_index=0)

        outputs_c1_i1 = model(tokens_1.unsqueeze(0), prefix_1, mask_1)
        logits_c1_i1 = outputs_c1_i1.logits[:, prefix_length-1: -1]
        loss_c1_i1 = nnf.cross_entropy(logits_c1_i1.reshape(-1, logits_c1_i1.shape[-1]),
                                       tokens_1.flatten(), ignore_index=0)

        scores += [
                {'label': f'{example["id"]}_c0_i0', 'score': 1/loss_c0_i0.item()}, 
                {'label': f'{example["id"]}_c0_i1', 'score': 1/loss_c0_i1.item()},
                {'label': f'{example["id"]}_c1_i0', 'score': 1/loss_c1_i0.item()},
                {'label': f'{example["id"]}_c1_i1', 'score': 1/loss_c1_i1.item()},
            ]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return scores


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

    print("Creating Winoground dataset", flush=True)
    test_dset = load_dataset('facebook/winoground')['test']

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model, clip_model, preprocess, prefix_length,
                             test_dset, tokenizer, device)

    with open(os.path.join(args.output_dir, f'{args.output_name}.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='clipcap')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cache', default='', type=str)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
