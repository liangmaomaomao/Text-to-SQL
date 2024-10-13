from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import nltk
import torch
import evaluate
from rouge_score import rouge_scorer, scoring
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
import gc
import sys
from spider import evaluation_hub

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME ='model/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

file_path="train_all.csv"
train_english_df = pd.read_csv(file_path)
# batch=train_english_df['Chinese_input'].iloc[:2].tolist()
# inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
# outputs = model.generate(**inputs)
#
# decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
# generated_texts.extend(decoded_texts)
# outputs = model.generate(
#         **inputs,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#         max_new_tokens=4000,
#         do_sample=False,
#         num_beams=2,
#     )

def get_all_cuda_tensors():
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append(obj)
        except:
            pass
    return cuda_tensors
data_input = train_english_df.iloc[1000:1180]
texts = data_input['English_input']

def evaluation_rouge(generated_texts,labels,db_id):
    formated_preds = [[pred] for pred in generated_texts]

    formated_labels = [[labels[i],db_id[i]] for i in range(len(generated_texts))]
    metric = evaluate.load("rouge")
    result_rouge = metric.compute(predictions=generated_texts, references=labels, use_stemmer=True)
    etype = "all"
    # gold = 'spider/evaluation_examples/gold_example.txt'
    # pred = 'spider/evaluation_examples/pred_example.txt'
    db_dir = 'spider/database'
    table = 'spider/evaluation_examples/examples/tables.json'

    kmaps = evaluation_hub.build_foreign_key_map_from_json(table)
    # result_rouge = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result_test = evaluation_hub.evaluate(formated_labels, formated_preds, db_dir, etype, kmaps)
    result = {**result_rouge, **result_test}
    return result


def generate_text(texts, batch_size=8):
    generated_texts = []
    model.eval() # set model to evaluation mode

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            if i+batch_size<= len(texts):
                batch = texts[i:i+batch_size].tolist()
            else:
                batch = texts[i:len(texts)].tolist()
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            print('input',torch.cuda.memory_allocated())
            outputs = model.generate(**inputs,num_beams=1,num_return_sequences=1,max_new_tokens=400)
            print('output',torch.cuda.memory_allocated())
            decoded_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            generated_texts.extend(decoded_texts)
            del outputs
            # del decoded_texts
            del inputs
            print('del',torch.cuda.memory_allocated())
            cuda_tensors = get_all_cuda_tensors()
            print('tensor_num',len(cuda_tensors))
    return generated_texts

generated_texts=generate_text(texts)
labels=data_input['query'].tolist()
db_id=data_input['db_id'].tolist()
evaluation_rouge = evaluation_rouge(generated_texts, data_input['query'].tolist(),data_input['db_id'].tolist())
print(evaluation_rouge)
