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

from datasets import load_dataset

from models.blip_pretrain import blip_pretrain

from transformers import BertTokenizer


@torch.no_grad()
def evaluation(model, dset, tokenizer, device, config):
    model.eval()

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    
    start_time = time.time()

    itm_scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_0, image_1 = example['image_0'].convert('RGB'), example['image_1'].convert('RGB')
        image_0, image_1 = test_transform(image_0), test_transform(image_1)
        image_0, image_1 = image_0.to(device, non_blocking=True), image_1.to(device, non_blocking=True)

        caption_0, caption_1 = pre_caption(example['caption_0'], config['max_tokens']), pre_caption(example['caption_1'], config['max_tokens'])
        text_input_0 = tokenizer(caption_0, padding='longest', max_length=config['max_tokens'], return_tensors='pt').to(device)
        text_input_1 = tokenizer(caption_1, padding='longest', max_length=config['max_tokens'], return_tensors='pt').to(device)

        image_embeds_0 = model.visual_encoder(image_0.unsqueeze(0))
        image_embeds_1 = model.visual_encoder(image_1.unsqueeze(0))
        image_atts = torch.ones(image_embeds_0.size()[:-1], dtype=torch.long).to(image_0.device)
        
        cross_c0_i0 = model.text_encoder(text_input_0.input_ids, attention_mask=text_input_0.attention_mask,
                                         encoder_hidden_states=image_embeds_0, encoder_attention_mask=image_atts,      
                                         return_dict=True)
        cross_c0_i1 = model.text_encoder(text_input_0.input_ids, attention_mask=text_input_0.attention_mask,
                                         encoder_hidden_states=image_embeds_1, encoder_attention_mask=image_atts,      
                                         return_dict=True)
        cross_c1_i0 = model.text_encoder(text_input_1.input_ids, attention_mask=text_input_1.attention_mask,
                                         encoder_hidden_states=image_embeds_0, encoder_attention_mask=image_atts,      
                                         return_dict=True)
        cross_c1_i1 = model.text_encoder(text_input_1.input_ids, attention_mask=text_input_1.attention_mask,
                                         encoder_hidden_states=image_embeds_1, encoder_attention_mask=image_atts,      
                                         return_dict=True)

        itm_logits_c0_i0 = model.itm_head(cross_c0_i0.last_hidden_state[:,0,:])
        itm_logits_c0_i1 = model.itm_head(cross_c0_i1.last_hidden_state[:,0,:])
        itm_logits_c1_i0 = model.itm_head(cross_c1_i0.last_hidden_state[:,0,:])
        itm_logits_c1_i1 = model.itm_head(cross_c1_i1.last_hidden_state[:,0,:])

        itm_scores_c0_i0 = F.softmax(itm_logits_c0_i0)[0][1].item()
        itm_scores_c0_i1 = F.softmax(itm_logits_c0_i1)[0][1].item()
        itm_scores_c1_i0 = F.softmax(itm_logits_c1_i0)[0][1].item()
        itm_scores_c1_i1 = F.softmax(itm_logits_c1_i1)[0][1].item()

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
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    model_without_ddp = model
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("Creating Winoground dataset", flush=True)
    test_dset = load_dataset('facebook/winoground')['test']

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model_without_ddp, test_dset, tokenizer, device, config)

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
