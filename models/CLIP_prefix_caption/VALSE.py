import argparse
import os

import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import torch
from torch.nn import functional as nnf

import clip
from transformers import GPT2Tokenizer

from model_utils import ClipCaptionModel, pad_tokens


@torch.no_grad()
def evaluation(model, instrument2data, instrument2piece, clip_model, clip_preprocess, prefix_length, tokenizer, device, data_dir):
    model.eval()

    start_time = time.time()
    itm_scores = []
    example_id = -1
    for instrument, dset in instrument2data.items():
        piece = instrument2piece[instrument]
        for k, example in tqdm(dset.items()):
            example_id += 1
            if example["mturk"]["caption"] < 2:
                continue

            image_path = os.path.join(data_dir, 'images', example['image_file'])
            image = Image.open(image_path).convert('RGB')
            image = clip_preprocess(image).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)


            caption, foil = example['caption'], example['foil']
            tokens_c, mask_c = pad_tokens(caption, tokenizer, prefix_length)
            tokens_c, mask_c = tokens_c.to(device), mask_c.to(device)

            tokens_f, mask_f = pad_tokens(foil, tokenizer, prefix_length)
            tokens_f, mask_f = tokens_f.to(device), mask_f.to(device)

            outputs_c = model(tokens_c.unsqueeze(0), prefix, mask_c)
            logits_c = outputs_c.logits[:, prefix_length-1: -1]
            loss_c = nnf.cross_entropy(logits_c.reshape(-1, logits_c.shape[-1]), tokens_c.flatten(), ignore_index=0)

            outputs_f = model(tokens_f.unsqueeze(0), prefix, mask_f)
            logits_f = outputs_f.logits[:, prefix_length-1: -1]
            loss_f = nnf.cross_entropy(logits_f.reshape(-1, logits_f.shape[-1]), tokens_f.flatten(), ignore_index=0)

            itm_scores += [
                {'label': f'{example_id}_c', 'score': 1/loss_c.item(), 'instrument': instrument, 'piece': piece, 'type': 'caption', 'id': example_id}, 
                {'label': f'{example_id}_f', 'score': 1/loss_f.item(), 'instrument': instrument, 'piece': piece, 'type': 'foil', 'id': example_id},
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

    print("Creating VALSE benchmark", flush=True)
    instrument2piece = {
        'actant-swap': 'actions', 
        'action-replacement': 'actions', 
        'coreference-hard': 'coreference',
        'coreference-standard': 'coreference',
        'counting-adversarial': 'counting',
        'counting-hard': 'counting',
        'counting-small-quant': 'counting',
        'existence': 'existence',
        'foil-it': 'foil-it',
        'plurals': 'plurality',
        'relations': 'relations',
    }
    instrument2data = {k: json.load(open(os.path.join(args.data_dir, f'{k}.json')))
                       for k in instrument2piece}
    
    print("Start evaluating", flush=True)
    test_scores = evaluation(model, instrument2data, instrument2piece,
                             clip_model, preprocess, prefix_length, tokenizer,
                             device, args.data_dir)

    with open(os.path.join(args.output_dir, f'{args.output_name}.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='clipcap')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
