import argparse
import os

import numpy as np
import time
import datetime
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from PIL import Image

from lavis.models import load_model_and_preprocess


@torch.no_grad()
def evaluation(model, dset, vis_processors, text_processors, k_test):
    start_time = time.time()

    dset_text = []
    dset_image = []
    dset_txt2img = {}
    dset_img2txt = {}
    txt_id = 0
    for img_id, ann in enumerate(dset):
        dset_image.append(ann["image"])
        dset_img2txt[img_id] = []
        for i, caption in enumerate(ann["caption"]):
            dset_text.append(text_processors['eval'](caption))
            dset_img2txt[img_id].append(txt_id)
            dset_txt2img[txt_id] = img_id
            txt_id += 1

    num_text = len(dset_text)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs), desc='Embedding texts'):
        text = dset_text[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    
    num_image = len(dset_image)
    image_bs = 256
    vit_feats = []
    image_embeds = []
    for i in tqdm(range(0, num_image, image_bs), desc='Embedding images'):
        image = dset_image[i : min(num_image, i + image_bs)]
        image = [Image.open(image_path).convert("RGB") for image_path in image]
        image = torch.stack([vis_processors["eval"](img) for img in image])
        image = image.to(model.device)
        with autocast():
            image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)
    score_matrix_i2t = torch.full(
        (len(dset_image), len(dset_text)), -100.0
    ).to(model.device)
    for i, sims in tqdm(enumerate(sims_matrix), desc='score_matrix_i2t'):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(dset_text), len(dset_image)), -100.0
    ).to(model.device)
    for i, sims in tqdm(enumerate(sims_matrix), desc='score_matrix_t2i'):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[i].repeat(k_test, 1),
            text_atts=text_atts[i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[i, topk_idx] = score + topk_sim

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
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2", args.checkpoint, device=device, is_eval=True)
    print("### Total Params: ", sum(p.numel() for p in model.parameters()))

    print("Creating Retrieval dataset", flush=True)
    with open(args.annotations_file) as f:
        test_data = json.load(f)
        for e in test_data:
            e['image'] = f'{args.images_dir}/{e["image"]}'

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    print("Start evaluating", flush=True)
    score_test_i2t, score_test_t2i, test_txt2img, test_img2txt = evaluation(
        model, test_data, vis_processors, text_processors, args.k_test)
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
    parser.add_argument('--output_name', type=str, default='blip2')
    parser.add_argument('--k_test', default=128, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
