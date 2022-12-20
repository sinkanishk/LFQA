# Investigating Summarization Techniques for Long Format Question Answering(LFQA)
E6998: Natural Language Generation and Summarization Course




## Overview

This repository contains code for running the experiments on the QuALITY data which can be used for running experiments with any MCQ dataset.

### Dataset Format

```
{
    "context":  ...
    "query":    ...
    "option_0"  ...
      ..
    "option_N"  ...
    "label":    ...
}
```

where the data will be formatted into inputs as:

```
[CLS] context [SEP] query option 
```


For custom datasets, you should provide a path to a folder containing files such as
```
config.json
train.jsonl
validation.jsonl
test.jsonl
```

The individual phases are optional. Each phase is contained in a single `jsonl` file, with the keys as specified above. `config.json` specifies some configurations for the task, such as the number of choices. 

### Models

Models can be broken down into 2 categories:

**Encoder-based** models are based on `transformers.AutoModelForMultipleChoice`. 

**Generation-based** models are based on `transformers.AutoModelForCausalLM`. 


## Usage setup

Create a conda env using 
```
conda env create -f lfqa.yml
```
It is recommended to add this package to your python site-packages.


## Sample usage

#### Training + Evaluating an Encoder

```bash
EXPDIR=/path/to/experiment
python run_lrqa.py \
    --model_name_or_path roberta-base \
    --model_mode mc \
    --task_name cosmosqa \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end \
    --num_train_epochs 1
```

#### Evaluating a Generator

```bash
EXPDIR=/path/to/experiment
python run_lrqa.py \
    --model_name_or_path gpt2 \
    --model_mode generation \
    --task_name cosmosqa \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end \
    --num_train_epochs 1
```



## Extractive Preprocessing

For creating datasets for extractive methodologies, run the extraction.py. 

```bash
python scripts/extraction.py \
    --input_base_path /path/to/input/data \
    --output_base_path /path/to/output/data \
    --summary hierarchical \
    --scorer rouge \
    --query_type question
```

This will prepare the data in a format that is suitable for the above `run_lrqa` scripts.

## Query-Summarization
```bash
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate
```
## Data Format

### CSV Format
```csv
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```
### JSON Format
```json
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

