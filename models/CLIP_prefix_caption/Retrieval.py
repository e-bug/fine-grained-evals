import argparse
import os

import numpy as np
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

from PIL import Image

import torch
from torch.nn import functional as nnf
from torch.cuda.amp import autocast

import clip
from transformers import GPT2Tokenizer

from model_utils import ClipCaptionModel, pad_tokens


@torch.no_grad()
def evaluation(model, clip_model, clip_preprocess, prefix_length, dset, tokenizer, k_test, device):
    model.eval()

    start_time = time.time()

    dset_text = []
    dset_image = []
    dset_txt2img = {}
    dset_img2txt = {}
    text_ids = []
    text_atts = []
    txt_id = 0
    for img_id, ann in enumerate(dset):
        dset_image.append(ann["image"])
        dset_img2txt[img_id] = []
        for i, caption in enumerate(ann["caption"]):
            tokens, mask = pad_tokens(caption, tokenizer, prefix_length)
            dset_text.append(caption)
            text_ids.append(tokens)
            text_atts.append(mask)
            dset_img2txt[img_id].append(txt_id)
            dset_txt2img[txt_id] = img_id
            txt_id += 1
    text_ids = torch.stack(text_ids, dim=0).to(device)
    text_atts = torch.stack(text_atts, dim=0).to(device)

    num_text = len(dset_text)
    text_bs = 256
    text_embeds = []
    for i in tqdm(range(0, num_text, text_bs), desc='Embedding texts'):
        text = dset_text[i : min(num_text, i + text_bs)]
        text = clip.tokenize(text, truncate=True).to(device)
        text_embed = clip_model.encode_text(text)
        text_embeds.append(text_embed)
    text_embeds = torch.cat(text_embeds, dim=0)
    
    num_image = len(dset_image)
    image_bs = 256
    image_embeds = []
    for i in tqdm(range(0, num_image, image_bs), desc='Embedding images'):
        image = dset_image[i : min(num_image, i + image_bs)]
        image = [Image.open(image_path).convert("RGB") for image_path in image]
        image = torch.stack([clip_preprocess(img) for img in image])
        image = image.to(device)
        image_embed = clip_model.encode_image(image)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=0)

    with autocast():
        sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = torch.full(
        (len(dset_image), len(dset_text)), -100.0
    ).to(device)
    for i, sims in tqdm(enumerate(sims_matrix), desc='score_matrix_i2t'):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        image_inputs = image_embeds[i].repeat(k_test, 1).to(device, dtype=torch.float32)
        tokens = text_ids[topk_idx]
        outputs = model(tokens, image_inputs, text_atts[topk_idx])
        logits = outputs.logits[:, prefix_length-1: -1]

        scores = []
        for l, t in zip(logits, tokens):
            loss = nnf.cross_entropy(l.reshape(-1, l.shape[-1]), t.flatten(), ignore_index=0)
            scores.append(1 / loss.item())
        score_matrix_i2t[i, topk_idx] = torch.tensor(scores, dtype=torch.float32, device=device)

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(dset_text), len(dset_image)), -100.0
    ).to(device)
    for i, sims in tqdm(enumerate(sims_matrix), desc='score_matrix_t2i'):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)

        image_inputs = image_embeds[topk_idx].to(device, dtype=torch.float32)
        tokens = text_ids[i].repeat(k_test, 1).to(device)
        outputs = model(tokens, image_inputs, text_atts[i].repeat(k_test, 1))
        logits = outputs.logits[:, prefix_length-1: -1]

        scores = []
        for l, t in zip(logits, tokens):
            loss = nnf.cross_entropy(l.reshape(-1, l.shape[-1]), t.flatten(), ignore_index=0)
            scores.append(1 / loss.item())
        score_matrix_t2i[i, topk_idx] = torch.tensor(scores, dtype=torch.float32, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), dset_txt2img, dset_img2txt

   
@torch.no_grad()
def compute_recall(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks_i2t = ranks
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result, ranks_i2t, ranks


def main(args):    
    device = torch.device(args.device)

    print("Creating model", flush=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("Creating Retrieval dataset", flush=True)
    with open(args.annotations_file) as f:
        test_data = json.load(f)
        for e in test_data:
            e['image'] = f'{args.images_dir}/{e["image"]}'

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    score_test_i2t, score_test_t2i, test_txt2img, test_img2txt = evaluation(
        model, clip_model, preprocess, prefix_length, test_data, tokenizer, args.k_test, device)
    test_result, test_i2t, test_t2i = compute_recall(
        score_test_i2t, score_test_t2i, test_txt2img, test_img2txt)

    log_stats = {**{f'test_{k}': v for k, v in test_result.items()}}
    print(log_stats)
    with open(os.path.join(args.output_dir, f'{args.output_name}.txt'), 'w') as f:
        f.write(json.dumps(log_stats) + "\n")
    np.save(os.path.join(args.output_dir, "score_test_i2t.txt"), score_test_i2t)
    np.save(os.path.join(args.output_dir, "score_test_t2i.txt"), score_test_t2i)
    np.save(os.path.join(args.output_dir, "test_i2t.txt"), test_i2t)
    np.save(os.path.join(args.output_dir, "test_t2i.txt"), test_t2i)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--annotations_file', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_name', type=str, default='clipcap')
    parser.add_argument('--k_test', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
