import argparse
import os

import time
import datetime
import json
import jsonlines
from tqdm import tqdm

import torch

import clip
from PIL import Image


@torch.no_grad()
def evaluation(model, preprocess, dset, data_dir, device):
    model.eval()
 
    start_time = time.time()

    scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_p = Image.open(os.path.join(data_dir, 'images', example['pos_image'])).convert('RGB')
        image_n = Image.open(os.path.join(data_dir, 'images', example['neg_image'])).convert('RGB')
        image_p, image_n = preprocess(image_p), preprocess(image_n)
        image_p, image_n = image_p.to(device, non_blocking=True), image_n.to(device, non_blocking=True)

        text_tokens = clip.tokenize(example['caption']).repeat(2, 1).to(device)
        image_input = torch.stack([image_p, image_n])

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        scores += [
            {'label': f'{example["svo_index"]}_p', 'score': float(similarity[0, 0]),
             'pos_triplet': example['pos_triplet'], 'neg_triplet': example['neg_triplet'],
             'subj_neg': example['subj_neg'], 'verb_neg': example['verb_neg'], 'obj_neg': example['obj_neg']},
            {'label': f'{example["svo_index"]}_n', 'score': float(similarity[1, 1]),
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
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("Creating SVO Probes dataset", flush=True)
    with jsonlines.open(f'{args.data_dir}/annotations/test.jsonl') as f:
        test_data = [i for i in f]

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model, preprocess, test_data, args.data_dir, device)

    with open(os.path.join(args.output_dir, f'{args.output_name}.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='clip')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    main(args)
