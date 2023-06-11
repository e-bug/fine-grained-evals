# Fine-grained Vision-and-Language Understanding Evaluations

This is the implementation of the approaches described in the paper:
> Emanuele Bugliarello, Laurent Sartran, Aishwarya Agrawal, Lisa Anne Hendricks and Aida Nematzadeh. [Measuring Progress in Fine-grained Vision-and-Language Understanding](arxiv.org/abs/2305.07558). _In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics_, Jul 2023.

We provide the code for reproducing our results on open source models.


## Models

[`models/`](models/) contains the source code for our evaluated models.

We added the `SVO.py`, `VALSE.py`, `VSR.py`, `Winoground.py` scripts to each model's source code to evaluate them on our set of fine-grained tasks.


## Evaluation

We provide all of our evaluation scripts in [experiments/](experiments).

For [ALBEF](models/ALBEF/configs_ours/), [BLIP](models/BLIP/configs_ours/), [PEVL](models/PEVL/configs_ours/) and [X-VLM](models/X-VLM/configs_ours/), task configuration files are stored in `configs_ours/` folders.

If you do not have the Winoground data, run the following first:
```python
from datasets import load_dataset

auth_token = ""  # FIXME: Set your HuggingFace authentication token.
test_dset = load_dataset('facebook/winoground', use_auth_token=auth_token)['test']
```


## Workdir

The following shows the structure of our working directory.

Ours is set to `BASE_DIR="/workdir"` in the scripts shared in [`experiments/`](experiments/).
Update it according to your setup.

<details>
<summary>Click to expand</summary>

```bash
checkpoints/
    | ALBEF/
    |   | ALBEF_4M.pth
    |   | ALBEF.pth
    | BLIP/
    |   | model_base.pth
    |   | model_base_14M.pth
    |   | model_base_capfilt_large.pth
    |   | model_large.pth
    | ClipCap/
    |   | clipcap_cc_weights.pt
    |   | clipcap_coco_weights.pt
    | PEVL/
    |   | grounding.pth
    |   | pevl_pretrain.pth
    |   | vrd.pth
    | X-VLM/
    |   | 16m_base_model_state_step_199999.th
    |   | 4m_base_model_state_step_199999.th
    | backbones/
    |   | huggingface/bert-base-uncased/
    |   |   | config.json
    |   |   | pytorch_model.bin
    |   |   | tokenizer_config.json
    |   |   | vocab.txt
    |   | hub/
    |   |   | swin_base_patch4_window7_224_22k.pth
data/
    | svo_probes/
    |   | annotations/test.jsonl
    |   | images/
    | VALSE/data/
    |   | images/
    |   | actant-swap.json
    |   | ...
    | vsr/
    |   | annotations/
    |   |   | dev.jsonl
    |   |   | test.jsonl
    |   | images/
envs/
    | albef/
    | blip/
    | lavis/
    | x-vlm/
fine-grained-evals/
```
</details>

For reference, [`data/`](data/) provides the text files used in our evaluation.
We remark that VSR has been [updated](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) after our experiments.

`envs/` contains venv environments based on the `requirements.txt` files of each model.

NB: You might need to `pip install datasets`.


## License

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{bugliarello-etal-2023-measuring,
    title = "Measuring Progress in Fine-grained Vision-and-Language Understanding",
    author = "Bugliarello, Emanuele   and
      Sartrain, Laurent  and
      Agrawal, Aishwarya  and
      Hendricks, Lisa Anne  and
      Nematzadeh, Aida",
    booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2305.07558",
}
```
