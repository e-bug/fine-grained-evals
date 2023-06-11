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
from data.utils import pre_caption

from models.blip_pretrain import blip_pretrain

from transformers import BertTokenizer


@torch.no_grad()
def evaluation(model, instrument2data, instrument2piece, tokenizer, device, config):
    model.eval()

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=Image.BICUBIC),
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

            image_embeds = model.visual_encoder(image.unsqueeze(0))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            cross_c = model.text_encoder(text_input_c.input_ids, attention_mask=text_input_c.attention_mask,
                                         encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,      
                                         return_dict=True)
            cross_f = model.text_encoder(text_input_f.input_ids, attention_mask=text_input_f.attention_mask,
                                         encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts,      
                                         return_dict=True)

            itm_logits_c = model.itm_head(cross_c.last_hidden_state[:,0,:])
            itm_logits_f = model.itm_head(cross_f.last_hidden_state[:,0,:])

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
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                          vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])
    model.visual_encoder_m = torch.nn.Identity()
    model.vision_proj_m = torch.nn.Identity()
    model.text_encoder_m = torch.nn.Identity()
    model.text_proj_m = torch.nn.Identity()
    state_dict = torch.load(args.checkpoint, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    model_without_ddp = model
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
    parser.add_argument('--output_name', type=str, default='blip')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
