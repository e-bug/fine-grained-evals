import argparse
import os

import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch

from datasets import load_dataset

from lavis.models import load_model_and_preprocess


@torch.no_grad()
def evaluation(model, dset, vis_processors, text_processors, device):
    start_time = time.time()

    itm_scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_0, image_1 = example['image_0'].convert('RGB'), example['image_1'].convert('RGB')

        image_0, image_1 = vis_processors["eval"](image_0).unsqueeze(0), vis_processors["eval"](image_1).unsqueeze(0)
        image_0, image_1 = image_0.to(device, non_blocking=True), image_1.to(device, non_blocking=True)
        
        caption_0, caption_1 = text_processors["eval"](example['caption_0']), text_processors["eval"](example['caption_1'])

        itm_logits_c0_i0 = model({"image": image_0, "text_input": caption_0}, match_head="itm")
        itm_logits_c0_i1 = model({"image": image_1, "text_input": caption_0}, match_head="itm")
        itm_logits_c1_i0 = model({"image": image_0, "text_input": caption_1}, match_head="itm")
        itm_logits_c1_i1 = model({"image": image_1, "text_input": caption_1}, match_head="itm")
        
        itm_scores_c0_i0 = torch.nn.functional.softmax(itm_logits_c0_i0, dim=1)[0][1].item()
        itm_scores_c0_i1 = torch.nn.functional.softmax(itm_logits_c0_i1, dim=1)[0][1].item()
        itm_scores_c1_i0 = torch.nn.functional.softmax(itm_logits_c1_i0, dim=1)[0][1].item()
        itm_scores_c1_i1 = torch.nn.functional.softmax(itm_logits_c1_i1, dim=1)[0][1].item()

        itm_scores += [
            {'label': f'{example["id"]}_c0_i0', 'score': itm_scores_c0_i0}, 
            {'label': f'{example["id"]}_c0_i1', 'score': itm_scores_c0_i1},
            {'label': f'{example["id"]}_c1_i0', 'score': itm_scores_c1_i0},
            {'label': f'{example["id"]}_c1_i1', 'score': itm_scores_c1_i1},
        ]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return itm_scores


def main(args):
    device = torch.device(args.device)

    print("Creating model", flush=True)
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", args.checkpoint, device=device, is_eval=True)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("Creating Winoground dataset", flush=True)
    test_dset = load_dataset('facebook/winoground')['test']

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model, test_dset, vis_processors, text_processors, device)

    with open(os.path.join(args.output_dir, f'{args.output_name}.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='albef')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
