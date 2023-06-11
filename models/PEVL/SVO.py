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

from models.model_pretrain import PEVL_pretrain
from models.tokenization_bert import BertTokenizer


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

        image_embeds_p = model.visual_encoder(image_p.unsqueeze(0))
        image_embeds_n = model.visual_encoder(image_n.unsqueeze(0))
        image_atts = torch.ones(image_embeds_p.size()[:-1], dtype=torch.long).to(image_p.device)
        text_embeds = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask,
                                              return_dict=True, mode='text').last_hidden_state

        cross_p = model.text_encoder.bert(encoder_embeds=text_embeds, attention_mask=text_input.attention_mask,
                                          encoder_hidden_states=image_embeds_p, encoder_attention_mask=image_atts,      
                                          return_dict=True, mode='fusion')
        cross_n = model.text_encoder.bert(encoder_embeds=text_embeds, attention_mask=text_input.attention_mask,
                                          encoder_hidden_states=image_embeds_n, encoder_attention_mask=image_atts,      
                                          return_dict=True, mode='fusion')

        itm_logits_p = model.itm_head(cross_p.last_hidden_state[:,0,:])
        itm_logits_n = model.itm_head(cross_n.last_hidden_state[:,0,:])

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

    ## Tokenizer
    unus = ['[unused{}]'.format(x) for x in range(200,800)]
    pos_token = ['@@']
    pos_token.extend([f'[pos_{x}]' for x in range(512)])
    pos_token.append('##')
    postoken_dict = {}
    tokenizer = BertTokenizer.from_pretrained('configs/vocab.txt')
    for x,y in zip(unus, pos_token):
        un_index = tokenizer.vocab[x]
        tokenizer.vocab[y] = un_index
        postoken_dict[y] = un_index
        _ = tokenizer.vocab.pop(x)
        tokenizer.basic_tokenizer.never_split.add(y)
    postoken_dict.pop('@@')
    postoken_dict.pop('##')
    postoken_index = torch.randn(30522).bool()
    postoken_index[:] = False
    for x in postoken_dict.values():
        postoken_index[x]=True

    print("Creating model", flush=True)
    model = PEVL_pretrain(config=config, tokenizer=tokenizer, postoken_dict=postoken_dict, init_deit=False)
    state_dict = torch.load(args.checkpoint, map_location='cpu')['model']
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    model_without_ddp = model

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
    parser.add_argument('--output_name', type=str, default='albef')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
