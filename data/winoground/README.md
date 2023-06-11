---
pretty_name: Winoground
task_categories:
- image-to-text
- text-to-image
- image-classification
extra_gated_prompt: "By clicking on “Access repository” below, you also agree that you are using it solely for research purposes. The full license agreement is available in the dataset files."
---
# Dataset Card for Winoground

## Dataset Description
Winoground is a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning. Given two images and two captions, the goal is to match them correctly—but crucially, both captions contain a completely identical set of words/morphemes, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing model performance. In our accompanying paper, we probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. In the paper, we perform an extensive analysis to obtain insights into how future work might try to mitigate these models’ shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field.

## Data
The captions and tags are located in `data/examples.jsonl` and the images are located in `data/images.zip`. You can load the data as follows:
```
from datasets import load_dataset
examples = load_dataset('facebook/winoground', use_auth_token=<YOUR USER ACCESS TOKEN>)
```
You can get `<YOUR USER ACCESS TOKEN>` by following these steps:
1) log into your Hugging Face account
2) click on your profile picture
3) click "Settings"
4) click "Access Tokens"
5) generate an access token

## Model Predictions and Statistics
The image-caption model scores from our paper are saved in `statistics/model_scores`. To compute many of the tables and graphs from our paper, run the following commands:
```
git clone https://huggingface.co/datasets/facebook/winoground
cd winoground
pip install -r statistics/requirements.txt
python statistics/compute_statistics.py
```

## FLAVA Colab notebook code for Winoground evaluation
https://colab.research.google.com/drive/1c3l4r4cEA5oXfq9uXhrJibddwRkcBxzP?usp=sharing

## CLIP Colab notebook code for Winoground evaluation
https://colab.research.google.com/drive/15wwOSte2CjTazdnCWYUm2VPlFbk2NGc0?usp=sharing

## Paper FAQ

### Why is the group score for a random model equal to 16.67%?

<details>
  <summary>Click for a proof!</summary>
  
  Intuitively, we might think that we can multiply the probabilities from the image and text score to get 1/16 = 6.25%. But, these scores are not conditionally independent. We can find the correct probability with combinatorics:

  For ease of notation, let:
  - a = s(c_0, i_0)
  - b = s(c_1, i_0)
  - c = s(c_1, i_1)
  - d = s(c_0, i_1)
  
  The group score is defined as 1 if a > b, a > d, c > b, c > d and 0 otherwise.
  
  As one would say to GPT-3, let's think step by step:
  
  1. There are 4! = 24 different orderings of a, c, b, d.
  2. There are only 4 orderings for which a > b, a > d, c > b, c > d:
  - a, c, b, d
  - a, c, d, b
  - c, a, b, d
  - c, a, d, b
  3. No ordering is any more likely than another because a, b, c, d are sampled from the same random distribution.
  4. We can conclude that the probability of a group score of 1 is 4/24 = 0.166...
</details>

## Citation Information

[https://arxiv.org/abs/2204.03162](https://arxiv.org/abs/2204.03162)

Tristan Thrush and Candace Ross contributed equally.
```
@inproceedings{thrush_and_ross2022winoground,
  author = {Tristan Thrush and Ryan Jiang and Max Bartolo and Amanpreet Singh and Adina Williams and Douwe Kiela and Candace Ross},
  title = {Winoground: Probing vision and language models for visio-linguistic compositionality},
  booktitle = {CVPR},
  year = 2022,
}
```