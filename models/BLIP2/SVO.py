import argparse
import os

import time
import datetime
import json
import jsonlines
from pathlib import Path
from tqdm import tqdm

import torch

from PIL import Image

from lavis.models import load_model_and_preprocess


@torch.no_grad()
def evaluation(model, dset, vis_processors, text_processors, device, data_dir):
    start_time = time.time()

    scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_p = Image.open(os.path.join(data_dir, 'images', example['pos_image'])).convert('RGB')
        image_n = Image.open(os.path.join(data_dir, 'images', example['neg_image'])).convert('RGB')
        image_p, image_n = vis_processors["eval"](image_p).unsqueeze(0), vis_processors["eval"](image_n).unsqueeze(0)
        image_p, image_n = image_p.to(device, non_blocking=True), image_n.to(device, non_blocking=True)        

        caption = text_processors["eval"](example['caption'])

        itm_logits_p = model({"image": image_p, "text_input": caption}, match_head="itm")
        itm_logits_n = model({"image": image_n, "text_input": caption}, match_head="itm")

        itm_scores_p = torch.nn.functional.softmax(itm_logits_p, dim=1)[0][1].item()
        itm_scores_n = torch.nn.functional.softmax(itm_logits_n, dim=1)[0][1].item()

        scores += [
            {'label': f'{example["svo_index"]}_p', 'score': itm_scores_p,
             'pos_triplet': example['pos_triplet'], 'neg_triplet': example['neg_triplet'],
             'subj_neg': example['subj_neg'], 'verb_neg': example['verb_neg'], 'obj_neg': example['obj_neg']},
            {'label': f'{example["svo_index"]}_n', 'score': itm_scores_n,
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
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", args.checkpoint, device=device, is_eval=True)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("Creating SVO Probes dataset", flush=True)
    with jsonlines.open(f'{args.data_dir}/annotations/test.jsonl') as f:
        test_data = [i for i in f]

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model, test_data, vis_processors, text_processors, device, args.data_dir)

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
    parser.add_argument('--output_name', type=str, default='albef')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
