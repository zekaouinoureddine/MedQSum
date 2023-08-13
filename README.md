![](https://img.shields.io/badge/Python-3.10-brightgreen.svg)
![](https://img.shields.io/badge/PyTorch-2.0-orange.svg)

# MedQSum
This GitHub repository presents the code source of our paper "**Paper Title**", which introduces a solution to get the most out of LLM, when answering health-related questions. We address the challenge of crafting accurate prompts by summarizing consumer health questions (CHQs) to generate clear and concise medical questions. Our approach involves fine-tuning Transformer-based models, including Flan-T5 in resource-constrained environments and three medical question summarization datasets.

### Datasets
To fine-tune and evaluate our implemented models, we used three question summarization datasets:




| Dataset  | Reference                                                        | Examples | Download                    | comments         |
|:--------:|:----------------------------------------------------------------:|:--------:|:---------------------------:|:----------------:|
| MeQ-Sum  | [Asma Ben Abacha et al](https://aclanthology.org/P19-1215/)      | 1000     | [download](./data/meq_sum/) |                  |
| HCM      | [Khalil Mrini et al](https://aclanthology.org/2021.bionlp-1.28/) | 1643     | [download](./data/hcm_sum/) |                  |
| CHQ-Summ | [Shweta Yadav et al](https://arxiv.org/abs/2206.06581)           | 1507     | [download](./data/chq_sum/) | 693 examples were chosen as outlined [here](https://github.com/shwetanlp/Yahoo-CHQ-Summ#data-preparation) |



### MedQSum Arch.
Our implemented models were fine-tuned using the following architecture:

<p align="center">
  <img src="./assets/models.png" style="box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.5);" />
</p>

### Results
The main validation results of fine-Tuned models for question summarization on three diverse datasets:

| Dataset | Model                  | R-1    | R-2    | R-L    | R-L-SUM |
|---------|------------------------|--------|--------|--------|---------|
| MeQ-Sum | T5 Base                | 41.78  | 25.88  | 39.90  | 39.97   |
|         | BART Large XSum        | 50.76* | 33.94* | 48.87* | 48.83*  |
|         | Pegasus XSum           | 46.59  | 30.46  | 44.60  | 44.78   |
|         | Flan-T5 XXL            | 45.74  | 24.87  | 43.16  | 43.09   |
| HCM     | T5 Base                | 38.49  | 19.82  | 37.64  | 37.71   |
|         | BART Large XSum        | 38.50  | 21.86* | 37.64  | 37.67   |
|         | Pegasus XSum           | 38.68* | 21.48  | 38.24* | 38.20*  |
|         | Flan-T5 XXL            | 38.34  | 19.35  | 36.94  | 36.89   |
| CHQ-Summ| T5 Base                | 38.31  | 20.36  | 36.05  | 36.10   |
|         | BART Large XSum        | 39.95* | 20.43* | 37.46* | 37.36*  |
|         | Pegasus XSum           | 37.16  | 18.76  | 34.96  | 34.86   |
|         | Flan-T5 XXL            | 36.78  | 17.02  | 35.08  | 35.05   |
| MeQ+HCM | T5 Base                | 37.90  | 20.11  | 36.75  | 36.75   |
|         | BART Large XSum        | 41.39* | 24.12* | 40.24* | 40.23*  |
|         | Pegasus XSum           | 41.14  | 22.13  | 40.03  | 39.96   |
|         | Flan-T5 XXL            | 41.31  | 22.41  | 39.74  | 39.73   |
| MeQ+HCM+CHQ | T5 Base            | 37.22  | 18.58  | 35.93  | 35.88   |
|            | BART Large XSum     | 41.10  | 23.06  | 39.17  | 39.20   |
|            | Pegasus XSum        | 41.66  | 23.51* | 40.27  | 40.32   |
|            | Flan-T5 XXL         | 42.69* | 23.28  | 40.88* | 40.87*  |



Ablations results showcasing the effects of generative configuration choices and instruction iine-Tuning on the MeQ-Sum dataset

| Model                                                   | R-1   | R-2   | R-L   | R-L-SUM |
|---------------------------------------------------------|-------|-------|-------|---------|
| Flan-T5 Standard Fine-tuning                            | 45.74 | 24.87 | 43.16 | 43.09   |
| Flan-T5 Instruction Fine-Tuning                         | 46.94*| 27.09*| 43.40*| 43.72*  |
| BART Large XSum                                         | 50.76 | 33.94 | 48.87 | 48.83   |
| BART Large XSum (top_p=.95, top_k=50, and temp.=.6)     | 54.32*| 38.08*| 51.98*| 51.99*  |


### Requirements
The code requires Python3 and the following dependencies:

```bash
pip install requirements.txt
```

### Fine-tuning
To fine-tune our implemented models and reproduce the results, copy and paste the following command:

```bash
python train.py \
      --train_data_path ../data/meq_sum/train.json \  # Training data
      --valid_data_path ../data/meq_sum/valid.json \  # Validation data
      --lr 3e-5 \                                     # Learning rate
      --epochs 1 \                                    # Epochs
      --model_checkpoint facebook/bart-large-xsum     # HF Model checkpoint
```

### Ineference
To perform inference and generate an intelligible CHQ, use the following command with your custom configuration.

```bash
python inference.py 
      --model_checkpoint t5-base \                   # Choose the saved model via its Hugging Face checkpoint
      --input_chq_text "type your CHQ input text" \  # Input text
```


<!-- ### Cite Us
If you are using this repo's code for your reseach work, please cite our paper:

```
@article{
    author = {Nour Eddine Zekaoui},
    title = {},
    journal
    year = {2023},
    month = {},
    doi = {}
    url = {}
}
``` -->

### Contact Us
For help or issues using the paper's code, please submit a GitHub **[issue](https://github.com/zekaouinoureddine/MedQSum/issues)**. For personal communication related to the paper, please contact: `{nour-eddine.zekaoui}@esi.ac.ma`.

> If you like it, give it a ⭐, then follow me on:
> - LinkedIn: [Nour Eddine ZEKAOUI](https://www.linkedin.com/in/nour-eddine-zekaoui-ba43b1177/)
> - Twitter: [@NZekaoui](https://twitter.com/NZekaoui)