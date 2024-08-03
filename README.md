# FineRadScore

This repository contains code to evaluate a pair of reports (candidate and ground truth) using the FineRadScore evaluation framework. 
- `datasets/`: contains csv headers for the ReFiSco-v0, ReFiSco-v1, and ReXVal datasets. You will need to download these datasets after signing a PhysioNet agreement and/or replace these files with the pairs of candidate/ground truth reports you want FineRadScore to evaluate on. Descriptions for each dataset can be found below:
    - `refisco-v0.csv`: [ReFiSco-v0 dataset](https://physionet.org/content/refisco/0.0/)
    - `refisco-v1.csv`: ReFiSco-v1 dataset
    - `refisco-v1-paraphrased.csv`: contains paraphrased versions in the column `corrected_paraphrase` of the generated reports of a subset of the ReFiSco-v1 dataset
    - `rexval_full.csv`: full [ReXVal dataset](https://physionet.org/content/rexval-dataset/1.0.0/)
    - `ReXVal_test_40.csv`: test split of the ReXVal dataset used to evaluate RadCliQ


## Creating an Environment

This repository was setup using conda. To create an environment, run `conda create -n testenv python=3.9`. Then, run `conda activate testenv` and `pip install -r requirements.txt` to install required packages.

## Preprocess Datasets

Run `python preprocess_datasets.py` to preprocess datasets. You should see new files appear in the `datasets/` folder.

## Adding API Keys

Run `export OPENAI_API_KEY=<api key>` to add your OpenAI API key. Also modify lines 8-10 in `gpt4_generations.py` accordingly to match your api type, version, and base information.

Run `export ANTHROPIC_API_KEY=<api key>` to add your Anthropic API key.

## Run Experiments on ReFiSco datasets

Run `python run_refisco_experiments.py <version> <setting> <model>`
- version: v0, v1
- setting: zeroshot, original, shuffled, paraphrased
- model: gpt4, claude3

Original, shuffled, and paraphrased settings are all using the few-shot prompt. For example, `python run_refisco_experiments.py v1 original gpt4`.

## Run Experiments on ReXVal datasets

Run `python run_rexval_experiments.py <version> <setting> <model>`
- version: test, full
- setting: zeroshot, fewshot
- model: gpt4, claude3

For example, `python run_rexval_experiments.py test fewshot gpt4`.
