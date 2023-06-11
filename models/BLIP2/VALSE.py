import argparse
import os

import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch

from PIL import Image

from lavis.models import load_model_and_preprocess


@torch.no_grad()
def evaluation(model, instrument2data, instrument2piece, vis_processors, text_processors, device, data_dir): 
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
            image = vis_processors["eval"](image).unsqueeze(0).to(device, non_blocking=True)

            caption, foil = text_processors["eval"](example['caption']), text_processors["eval"](example['foil'])

            itm_logits_c = model({"image": image, "text_input": caption}, match_head="itm")
            itm_logits_f = model({"image": image, "text_input": foil}, match_head="itm")

            itm_scores_c = torch.nn.functional.softmax(itm_logits_c, dim=1)[0][1].item()
            itm_scores_f = torch.nn.functional.softmax(itm_logits_f, dim=1)[0][1].item()

            itm_scores += [
                {'label': f'{example_id}_c', 'score': itm_scores_c, 'instrument': instrument, 'piece': piece, 'type': 'caption', 'id': example_id}, 
                {'label': f'{example_id}_f', 'score': itm_scores_f, 'instrument': instrument, 'piece': piece, 'type': 'foil', 'id': example_id},
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
    instrument2data = {k: json.load(open(os.path.join(args.data_dir, f'{k}.json'))) for k in instrument2piece}
    
    print("Start evaluating", flush=True)
    test_scores = evaluation(model, instrument2data, instrument2piece, vis_processors, text_processors, device, args.data_dir)

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
    parser.add_argument('--output_name', type=str, default='xvlm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
