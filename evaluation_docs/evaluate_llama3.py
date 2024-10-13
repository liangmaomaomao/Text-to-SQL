import sqlparse
import torch
import evaluate
from rouge_score import rouge_scorer, scoring
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import gc
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import sys
import os

import evaluation_metrics
#import spider
from evaluation_metrics import evaluation_hub


def get_all_cuda_tensors():
    cuda_tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                cuda_tensors.append(obj)
        except:
            pass
    return cuda_tensors

def generate_query_no_use(file_path,lan,batch_size):
    MODEL_NAME = 'model/Meta-Llama-3-8B'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_memory = torch.cuda.get_device_properties(0).total_memory

    ##model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if available_memory > 15e9:
        # if you have atleast 15GB of GPU memory, run load the model in float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16).to(device)
    else:
        # else, load in 8 bits – this is a bit slower
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # torch_dtype=torch.float16,
            load_in_8bit=True).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ##data
    train_df = pd.read_csv(file_path)
    train_english_df = Dataset.from_pandas(train_df).select(range(1000,1010))
    if lan == "English":
        texts = train_english_df['English_input']
    else:
        texts = train_english_df['Chinese_input']
    # if torch.cuda.device_count() > 1:
    #     model=torch.nn.DataParallel(model)
    ##test
    #batch = texts.iloc[:2].tolist()
    #batch = texts

    generated_texts = []
    model.eval()  # set model to evaluation mode
    #accumulate=[]

    with (torch.no_grad()):
        for i in tqdm(range(0, len(texts), batch_size)):
            if i + batch_size <= len(texts):
                batch = texts[i:i + batch_size]
                # batch = texts[i:i+batch_size]
            else:
                batch = texts[i:len(texts)]
                # batch = texts[i:len(texts)]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            print('input', torch.cuda.memory_allocated())
            # model.eval()
            outputs = model.generate(
                **inputs,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=400,
                do_sample=False,
                #num_beams=1
                )
            print('output', torch.cuda.memory_allocated())
            decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            query_g = [sqlparse.format(decoded_text.split("[SQL]")[-1].split("[/SQL]")[0], reindent=True) for decoded_text in decoded_texts]

            del inputs
            del outputs
            print('del', torch.cuda.memory_allocated())
            # torch.cuda.empty_cache()  # Free unused memory from PyTorch
            #print(torch.cuda.memory_allocated())
            # gc.collect()  # Additional garbage collection in Python
            cuda_tensors = get_all_cuda_tensors()
            print(len(cuda_tensors))
            # for tensor in cuda_tensors:
            #     print(tensor.size)
            generated_texts.extend(query_g)

    #train_english_df['pred_query']=generated_texts
    #train_english_df.to_csv("llama3"+lan+file_path.split('/')[-1])
    return generated_texts, train_english_df['query'], train_english_df['db_id']

def generate_query(file_path,lan,batch_size):
    #file_path = "train_all.csv"
    MODEL_NAME = 'Meta-Llama-3-8B'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    available_memory = torch.cuda.get_device_properties(0).total_memory

    ##model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if available_memory > 15e9:
        # if you have atleast 15GB of GPU memory, run load the model in float16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16).to(device)
    else:
        # else, load in 8 bits – this is a bit slower
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # torch_dtype=torch.float16,
            load_in_8bit=True).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ##data
    train_df = pd.read_csv(file_path)
    # train_english_df = Dataset.from_pandas(train_df).select(range(1000,1010))
    train_english_df = Dataset.from_pandas(train_df)
    if lan == "English":
        texts = train_english_df['English_input']
    else:
        texts = train_english_df['Chinese_input']
    # if torch.cuda.device_count() > 1:
    #     model=torch.nn.DataParallel(model)
    ##test
    #batch = texts.iloc[:2].tolist()
    #batch = texts

    generated_texts = []
    model.eval()  # set model to evaluation mode
    #accumulate=[]

    with (torch.no_grad()):
        for i in tqdm(range(0, len(texts), batch_size)):
            if i + batch_size <= len(texts):
                batch = texts[i:i + batch_size]
                # batch = texts[i:i+batch_size]
            else:
                batch = texts[i:len(texts)]

            pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer,device=device)
            outputs = pipe(
                batch,
                max_new_tokens=400,
                do_sample=False,
                # num_beams=1,
                num_return_sequences=1,
                # return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            query_g = [sqlparse.format(decoded_text[0]["generated_text"].split("[SQL]")[-1].split("[/SQL]")[0], reindent=True) for decoded_text in outputs]
            del outputs
            del pipe
            print('del', torch.cuda.memory_allocated())
            # torch.cuda.empty_cache()  # Free unused memory from PyTorch
            #print(torch.cuda.memory_allocated())
            # gc.collect()  # Additional garbage collection in Python
            cuda_tensors = get_all_cuda_tensors()
            print(len(cuda_tensors))
            # for tensor in cuda_tensors:
            #     print(tensor.size)
            generated_texts.extend(query_g)

    train_df['pred_query']=generated_texts
    # train_english_df.to_csv("llama3.csv")
    train_df.to_csv("data/spider/test_data/llama3_"+lan+file_path.split('/')[-1])
    return generated_texts, train_english_df['query'], train_english_df['db_id']

def evaluation_rouge(generated_texts,labels,db_id):
    formated_preds = [[pred] for pred in generated_texts]

    formated_labels = [[labels[i],db_id[i]] for i in range(len(generated_texts))]
    metric = evaluate.load("rouge")
    result_rouge = metric.compute(predictions=generated_texts, references=labels, use_stemmer=True)
    etype = "all"
    # gold = 'spider/evaluation_examples/gold_example.txt'
    # pred = 'spider/evaluation_examples/pred_example.txt'
    db_dir = '/data/spider/test_database'
    table = 'data/spider/test_data/tables.json'

    kmaps = evaluation_hub.build_foreign_key_map_from_json(table)
    # result_rouge = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result_test = evaluation_hub.evaluate(formated_labels, formated_preds, db_dir, etype, kmaps)
    result = {**result_rouge, **result_test}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file',  type=str, default="data/spider/test_data/test_all.csv")
    parser.add_argument('--lan',  type=str, default="English")
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()
    file_path = args.file
    lan = args.lan
    batch_size = args.batch

    file_path = "data/spider/test_data/test_all.csv"
    lan = "English"
    batch_size = 8

    generated_texts, labels, db_id = generate_query(file_path, lan, batch_size)
    print(generated_texts)
    evaluation_rouge = evaluation_rouge(generated_texts, labels,db_id)
    print(evaluation_rouge)

