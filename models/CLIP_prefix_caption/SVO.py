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
def evaluation(model, clip_model, clip_preprocess, prefix_length, dset, tokenizer, device, data_dir):
    model.eval()

    start_time = time.time()
    scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_p = Image.open(os.path.join(data_dir, 'images', example['pos_image'])).convert('RGB')
        image_n = Image.open(os.path.join(data_dir, 'images', example['neg_image'])).convert('RGB')
        image_p, image_n = clip_preprocess(image_p).unsqueeze(0), clip_preprocess(image_n).unsqueeze(0)
        image_p, image_n = image_p.to(device), image_n.to(device)        
        with torch.no_grad():
            prefix_p = clip_model.encode_image(image_p).to(device, dtype=torch.float32)
            prefix_n = clip_model.encode_image(image_n).to(device, dtype=torch.float32)
        
        caption = example['caption']
        tokens, mask = pad_tokens(caption, tokenizer, prefix_length)
        tokens, mask = tokens.to(device), mask.to(device)

        outputs_p = model(tokens.unsqueeze(0), prefix_p, mask)
        logits_p = outputs_p.logits[:, prefix_length-1: -1]
        loss_p = nnf.cross_entropy(logits_p.reshape(-1, logits_p.shape[-1]), tokens.flatten(), ignore_index=0)

        outputs_n = model(tokens.unsqueeze(0), prefix_n, mask)
        logits_n = outputs_n.logits[:, prefix_length-1: -1]
        loss_n = nnf.cross_entropy(logits_n.reshape(-1, logits_n.shape[-1]), tokens.flatten(), ignore_index=0)

        scores += [
            {'label': f'{example["svo_index"]}_p', 'score': 1/loss_p.item(),
             'pos_triplet': example['pos_triplet'], 'neg_triplet': example['neg_triplet'],
             'subj_neg': example['subj_neg'], 'verb_neg': example['verb_neg'], 'obj_neg': example['obj_neg']},
            {'label': f'{example["svo_index"]}_n', 'score': 1/loss_n.item(),
             'pos_triplet': example['pos_triplet'], 'neg_triplet': example['neg_triplet'],
             'subj_neg': example['subj_neg'], 'verb_neg': example['verb_neg'], 'obj_neg': example['obj_neg']},
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

    print("Creating SVO Probes dataset", flush=True)
    with jsonlines.open(f'{args.data_dir}/annotations/test.jsonl') as f:
        test_data = [i for i in f]

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model, clip_model, preprocess, prefix_length,
                             test_data, tokenizer, device, args.data_dir)

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
