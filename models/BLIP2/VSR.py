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
def evaluation(model, data, vis_processors, text_processors, device, data_dir):
    start_time = time.time()

    itm_scores = []
    for example_id, example in tqdm(enumerate(data)):
        image_path = os.path.join(data_dir, 'images', example['image'])
        image = Image.open(image_path).convert('RGB')
        image = vis_processors["eval"](image).unsqueeze(0).to(device, non_blocking=True)

        caption = text_processors["eval"](example['caption'])

        itm_logits = model({"image": image, "text_input": caption}, match_head="itm")
        itm_score = torch.nn.functional.softmax(itm_logits, dim=1)[0][1].item()

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
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", args.checkpoint, device=device, is_eval=True)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Creating VSR benchmark", flush=True)
    with jsonlines.open(f'{args.data_dir}/annotations/{args.data_variant}/dev.jsonl') as f:
        val_data = [i for i in f]
    with jsonlines.open(f'{args.data_dir}/annotations/{args.data_variant}/test.jsonl') as f:
        test_data = [i for i in f]
    print("Start evaluating", flush=True)
    val_scores = evaluation(model, val_data, vis_processors, text_processors, device, args.data_dir)
    test_scores = evaluation(model, test_data, vis_processors, text_processors, device, args.data_dir)

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
    parser.add_argument('--output_name', type=str, default='xvlm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
