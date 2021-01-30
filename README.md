# InterpretableEntityRepresentation

> [**Interpretable Entity Representations through Large-Scale Typing**](https://www.aclweb.org/anthology/2020.findings-emnlp.54.pdf)<br/>
> Yasumasa Onoe and Greg Durrett<br/>
> Findings of EMNLP 2020

## Getting Started 

### Dependencies

```bash
$ git clone https://github.com/yasumasaonoe/InterpretableEntityRepresentation.git
```

This code has been tested with Python 3.6 and the following dependencies:

- `torch==1.6.0`
- `tqdm==4.48.2`
- `transformers==3.1.0`

If using a conda environment, please use the following commands:

```bash
$ conda create -n et python=3.6
$ conda activate et
$ pip install  [package name]
```

### File Descriptions

- `run_et.py`: Main script for training and evaluating models, and writing predictions to an output file.
- `models.py`: Defines a Transformer-based entity typing model.
- `transformer_data_utils.py`: Contains data loader and utility functions.
- `transformer_constant.py`: Defines paths etc.
- `train.sh`: Sample training command.

## Datasets / Models

This code assumes 3 directories listed below. Paths to these directories are specified in `transformer_constant.py`.
- `./data`: This directory contains train/dev data files. [[Download](https://drive.google.com/file/d/1zcNZ-Ng4yARjwCsIVQZ5AoPdSADBbknO/view?usp=sharing)]
- `./ontology`: This directory contains type vocab files. [[Download](https://drive.google.com/file/d/1KD5Oz62Wel38rFggcuuBs_VuHtMppGmc/view?usp=sharing)]
- `./model`: Trained models will be saved in this directory. When you run `run_et.py` with the test mode, the trained model is loaded from here.

Data file patterns look like below:

- Train (`./data/train/wiki_context/train_*.json`)
- Dev (`./data/validation/wiki_context_dev_999.json`)

```

----------------------------------------------------------------------------------------------------

The data files are formated as jsonlines. Here is a single training example:

{
  "ex_id": "08_4210795",
  "wikiurl": "http://en.wikipedia.org/wiki/Nova_Scotia",
  "wikiId": "21184",
  "y_wikiurl_dump": "https://en.wikipedia.org/wiki?curid=21184",
  "left_context_text": "Transit Cape Breton is a public transport agency operating buses in the Cape Breton Regional Municipality (CBRM), in ",
  "word": "Nova Scotia",
  "right_context_text": ", Canada.",
  "left_context": ["Transit", "Cape", "Breton", "is", "a", "public", "transport", "agency", "operating", "buses", "in", "the", "Cape", "Breton", "Regional", "Municipality", "(", "CBRM", ")", ",", "in"],
  "mention_as_list": ["Nova", "Scotia"],
  "right_context": [",", "Canada", "."],
  "y_category": ["british", "territories", "states and territories established in 1867", "british north america", "atlantic canada", "the maritimes", "provinces of canada", "1867", "established", "of canada", "former british colonies and protectorates in the americas", "colonies", "1867 establishments in canada", "in the americas", "former", "provinces", "former scottish colonies", "states", "establishments", "protectorates", "in 1867", "nova scotia", "acadia", "in canada"], 
  "y_title": "Nova Scotia"
}
```

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `ex_id`                   | Unique example ID.                                                                       |
| `wikiurl`                 | Wikipedia url of the gold Wiki entity.                                                   |
| `wikiId`                  | Wiki page ID of the gold entity.                                                         |
| `y_wikiurl_dump`          | Another Wikipedia url of the gold Wiki entity.                                           |
| `left_context_text`       | Left context of a mention.                                                               |
| `word`                    | A mention.                                                                               |
| `right_context_text`      | Right context of a mention.                                                              |
| `left_context`            | Tokenized left context of a mention.                                                     |
| `mention_as_list`         | A tokenized mention. |                                                                   |
| `right_context`           | Tokenized right context of a mention.                                                    |
| `y_category`              | The gold entity types derived from Wikipedia categories.                                 |
| `y_title`                 | Wikipedia title of the gold Wiki entity.                                                 |


## Entity Typing Training and Evaluation

### Training

`run_et.py` is the primary script for training and evaluating models. Here is an example of training:

```bash
$ python3 -u run_et.py \
-model_id bert_base_uncased_1 \
-model_type bert-base-uncased \
-mode train \
-goal 60k \
-learning_rate_enc 2e-5 \
-learning_rate_cls 1e-3 \
-per_gpu_train_batch_size 8 \
-per_gpu_eval_batch_size 8 \
-gradient_accumulation_steps 4 \
-log_period 1000 \
-eval_period 2000 \
-eval_after 2000 \
-save_period 10000000 \
-train_data train/wiki_context/train_*.json \
-dev_data validation/wiki_context_dev_999.json
```

Descriptions for command line arguments above: 

| Flag                             | Description                                                                       |
|----------------------------------|-----------------------------------------------------------------------------------|
| `-model_id`                      | Experiment name.                                                                  |
| `-model_type`                    | Pretrained MLM type. Currently, BERT and RoBERTa are supported.                   |
| `-mode`                          | Whether to train or test. This can be either `train` or `test`.                   |
| `-goal`                          | Type vocab.                                                                       |
| `-learning_rate_enc`             | Initial learning rate for the Transformer encoder.                                |
| `-learning_rate_cls`             | Initial learning rate for the type embeddings.                                    |
| `-per_gpu_train_batch_size`      | The batch size per GPU in the train mode.                                         |
| `-per_gpu_eval_batch_size`       | The batch size per GPU in the eval mode.                                          |
| `-gradient_accumulation_steps`   | Number of updates steps to accumulate before performing a backward/update pass.   |
| `-log_period`                    | How often to save.                                                                |
| `-eval_period`                   | How often to run dev.                                                             |
| `-eval_after`                    | When to start to run dev.                                                         |
| `-save_period`                   | How often to save.                                                                |
| `-train_data`                    | Train data file pattern.                                                          |
| `-dev_data`                      | Dev data file pattern.                                                            |
  
Descriptions for all command line arguments are provided in `run_et.py`. 

### Evaluation

If you would like to evaluate the trained model on another dataset, simply set `-mode` to `test` and point to the test data using `-eval_data`. Make sure put `-load` so that the trained parameters will be loaded. 

```bash
$ python3 -u run_et.py \
-model_id bert_large_wwm_1 \
-model_type bert-large-uncased-whole-word-masking \
-load \
-mode test \
-goal 60k \
-eval_data validation/wiki_context_dev_999.json
```

## Downstream Task Evaluation

Work in progress....
