import argparse
import os

import ruamel.yaml as yaml
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from dataset.utils import pre_caption

from models.model_pretrain import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer


@torch.no_grad()
def evaluation(model, instrument2data, instrument2piece, tokenizer, device, config):
    model.eval()

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    
    start_time = time.time()

    itm_scores = []
    example_id = -1
    for instrument, dset in instrument2data.items():
        piece = instrument2piece[instrument]
        for k, example in tqdm(dset.items()):
            example_id += 1
            if example["mturk"]["caption"] < 2:
                continue

            image_path = os.path.join(config['data_dir'], 'images', example['image_file'])
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).to(device, non_blocking=True)

            caption, foil = pre_caption(example['caption'], config['max_tokens']), pre_caption(example['foil'], config['max_tokens'])
            text_input_c = tokenizer(caption, padding='longest', max_length=config['max_tokens'], return_tensors='pt').to(device)
            text_input_f = tokenizer(foil, padding='longest', max_length=config['max_tokens'], return_tensors='pt').to(device)

            image_embeds, image_atts = model.get_vision_embeds(image.unsqueeze(0))
            text_embeds_c = model.get_text_embeds(text_input_c.input_ids, text_input_c.attention_mask)
            text_embeds_f = model.get_text_embeds(text_input_f.input_ids, text_input_f.attention_mask)

            cross_c = model.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds_c, text_atts=text_input_c.attention_mask)[:, 0, :]
            cross_f = model.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds_f, text_atts=text_input_f.attention_mask)[:, 0, :]

            itm_logits_c = model.itm_head(cross_c)
            itm_logits_f = model.itm_head(cross_f)

            itm_scores_c = F.softmax(itm_logits_c)[0][1].item()
            itm_scores_f = F.softmax(itm_logits_f)[0][1].item()

            itm_scores += [
                {'label': f'{example_id}_c', 'score': itm_scores_c, 'instrument': instrument, 'piece': piece, 'type': 'caption', 'id': example_id}, 
                {'label': f'{example_id}_f', 'score': itm_scores_f, 'instrument': instrument, 'piece': piece, 'type': 'foil', 'id': example_id},
            ]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return itm_scores


def main(args, config):
    device = torch.device(args.device)

    print("Creating model", flush=True)
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=True)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    model_without_ddp = model
    if config['use_roberta']:
        tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

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
    instrument2data = {k: json.load(open(os.path.join(config['data_dir'], f'{k}.json'))) for k in instrument2piece}
    
    print("Start evaluating", flush=True)
    test_scores = evaluation(model_without_ddp, instrument2data, instrument2piece, tokenizer, device, config)

    with open(os.path.join(args.output_dir, f'{args.output_name}.jsonl'), 'w') as f:
        f.write('\n'.join(map(json.dumps, test_scores)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='xvlm')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
