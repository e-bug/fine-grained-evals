import argparse
import os

import ruamel.yaml as yaml
import time
import datetime
import json
import jsonlines
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
def evaluation(model, dset, tokenizer, device, config):
    model.eval()

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    
    start_time = time.time()

    scores = []
    for example in tqdm(dset):
        # Note that some images in winoground are RGBA and some are RGB. Need to convert all to RGB with .convert('RGB')
        image_p = Image.open(os.path.join(config['data_dir'], 'images', example['pos_image'])).convert('RGB')
        image_n = Image.open(os.path.join(config['data_dir'], 'images', example['neg_image'])).convert('RGB')
        image_p, image_n = test_transform(image_p), test_transform(image_n)
        image_p, image_n = image_p.to(device, non_blocking=True), image_n.to(device, non_blocking=True)

        caption = pre_caption(example['caption'], config['max_tokens'])
        text_input = tokenizer(caption, padding='longest', max_length=config['max_tokens'], return_tensors='pt').to(device)

        image_embeds_p, image_atts_p = model.get_vision_embeds(image_p.unsqueeze(0))
        image_embeds_n, image_atts_n = model.get_vision_embeds(image_n.unsqueeze(0))
        text_embeds = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)

        cross_p = model.get_cross_embeds(image_embeds_p, image_atts_p,
                                         text_embeds=text_embeds,
                                         text_atts=text_input.attention_mask)[:, 0, :]
        cross_n = model.get_cross_embeds(image_embeds_n, image_atts_n,
                                         text_embeds=text_embeds,
                                         text_atts=text_input.attention_mask)[:, 0, :]

        itm_logits_p = model.itm_head(cross_p)
        itm_logits_n = model.itm_head(cross_n)

        itm_scores_p = F.softmax(itm_logits_p)[0][1].item()
        itm_scores_n = F.softmax(itm_logits_n)[0][1].item()

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

    print("Creating SVO Probes dataset", flush=True)
    with jsonlines.open(f'{config["data_dir"]}/annotations/test.jsonl') as f:
        test_data = [i for i in f]

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    test_scores = evaluation(model_without_ddp, test_data, tokenizer, device, config)

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
