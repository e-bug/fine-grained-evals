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

from datasets import load_dataset

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
        text_embeds_0 = model.text_encoder.bert(text_input_0.input_ids, attention_mask=text_input_0.attention_mask,
                                                return_dict=True, mode='text').last_hidden_state
        text_embeds_1 = model.text_encoder.bert(text_input_1.input_ids, attention_mask=text_input_1.attention_mask,
                                                return_dict=True, mode='text').last_hidden_state

        cross_c0_i0 = model.text_encoder.bert(encoder_embeds=text_embeds_0, attention_mask=text_input_0.attention_mask,
                                              encoder_hidden_states=image_embeds_0, encoder_attention_mask=image_atts,      
                                              return_dict=True, mode='fusion')
        cross_c0_i1 = model.text_encoder.bert(encoder_embeds=text_embeds_0, attention_mask=text_input_0.attention_mask,
                                              encoder_hidden_states=image_embeds_1, encoder_attention_mask=image_atts,      
                                              return_dict=True, mode='fusion')
        cross_c1_i0 = model.text_encoder.bert(encoder_embeds=text_embeds_1, attention_mask=text_input_1.attention_mask,
                                              encoder_hidden_states=image_embeds_0, encoder_attention_mask=image_atts,      
                                              return_dict=True, mode='fusion')
        cross_c1_i1 = model.text_encoder.bert(encoder_embeds=text_embeds_1, attention_mask=text_input_1.attention_mask,
                                              encoder_hidden_states=image_embeds_1, encoder_attention_mask=image_atts,      
                                              return_dict=True, mode='fusion')

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
    parser.add_argument('--output_name', type=str, default='pevl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
