# Evaluate final test set on 1 network:
# DISTILBERT

import torch
from transformers import *
import run_glue_gap
import json
import pandas as pd
import numpy as np
import pickle
import shutil
from sklearn.model_selection import KFold
import time
import os
import glob
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
import tqdm
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer)
from utils_glue_gap import (compute_metrics,
                        output_modes, processors, simple_accuracy)
#
all_preds=[]
all_finalpreds=[]
all_actuals=[]
all_accs=[]
all_times=[]
#networka=0
#
parser = run_glue_gap.argparse.ArgumentParser()
args = run_glue_gap.get_args(parser)
#
model_typea=3 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert, 4: XLM, 5:TransformerXL
small_or_big=0 # 0: small, 1: big
args.con_or_lib=0 # 0: con, 1: lib, 2: original
batch_size=32
#
args.num_train_epochs=5.0
args.do_train=True
args.do_eval=True
#
#args.max_steps=3
args.overwrite_output_dir=True
args.overwrite_cache = True
args.evaluate_during_training=False
args.logging_steps =50
args.save_steps=1000
args.max_seq_length = 512
args.seed=round(time.time())
#
args.learning_rate =6.3e-5
args.adam_epsilon=1e-8
args.warmup_steps= 97  
args.weight_decay= 0.0
#
if small_or_big==0:
    args.per_gpu_eval_batch_size=8
    args.per_gpu_train_batch_size=8
    if batch_size==8:
        args.gradient_accumulation_steps=1
    elif batch_size==16:
        args.gradient_accumulation_steps=2
    elif batch_size==32:
        args.gradient_accumulation_steps=4
elif small_or_big==1:
    args.per_gpu_eval_batch_size=1
    args.per_gpu_train_batch_size=1
    if batch_size==8:
        args.gradient_accumulation_steps=8
    elif batch_size==16:
        args.gradient_accumulation_steps=16
    elif batch_size==32:
        args.gradient_accumulation_steps=32
#
#
if model_typea==0:
    args.model_type='bert'
    args.do_lower_case=True
    if small_or_big==0:
        args.model_name_or_path='bert-base-uncased'
    else:
        args.model_name_or_path='bert-large-uncased'
elif model_typea==1:
    args.model_type='xlnet'
    args.do_lower_case=False
    #args.model_name_or_path='xlnet-base-cased'
    if small_or_big==0:
        args.model_name_or_path='xlnet-base-cased'
    else:
        args.model_name_or_path='xlnet-large-cased'
elif model_typea==2:
    args.model_type='roberta'
    args.do_lower_case=False
    #args.model_name_or_path='roberta-base'
    if small_or_big==0:
        args.model_name_or_path='roberta-base'
    else:
        args.model_name_or_path='roberta-large'
elif model_typea==3:
    args.model_type='distilbert'
    args.model_name_or_path='distilbert-base-uncased'
    args.do_lower_case=True
elif model_typea==4:
    args.model_type='xlm'
    args.model_name_or_path='xlm-mlm-en-2048'
    args.do_lower_case=False
#
#
args.task_name='entityrel'
args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_con_eval2'
args.data_dir='/home/projects/data/entrel_data_con_eval2'
args.output_dir='/tmp/entrel/'
#
results = {}
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}
# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
#
args.device = device
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
checkpoints = [args.output_dir]
checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
args.output_mode = output_modes[args.task_name]
checkpoint=checkpoints[-1]
global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
model = model_class.from_pretrained(checkpoint)
model.to(args.device)
#result = run_glue_gap.evaluate(args, model, tokenizer, prefix=prefix)
eval_task=args.task_name
eval_dataset = run_glue_gap.load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
start_time=time.time()
#for batch in tqdm(eval_dataloader, desc="Evaluating"):
for batch in eval_dataloader:
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)
    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[3]}
        if args.model_type != 'distilbert':
            inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if preds is None:
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
#
eval_loss = eval_loss / nb_eval_steps
final_preds = np.argmax(preds, axis=1)
result = simple_accuracy(final_preds, out_label_ids)
nupreds=[list(map(float,list(preds[:,0]))),list(map(float, list(preds[:,1])))]
all_preds.append(nupreds)
all_finalpreds.append(list(map(float,list(final_preds))))
all_actuals.append(list(map(float,list(out_label_ids))))
all_accs.append(result)
all_times.append(time.time()-start_time)


the_dic={'all_outputs':all_preds,'all_preds':all_finalpreds,'all_actuals':all_actuals,'all_accs':all_accs,'all_times':all_times}
filename='/home/ec2-user/SageMaker/Data/distilbert_testdata_nooverlap_outputs.json'
with open(filename, 'w') as outfile:
    json.dump(the_dic, outfile)
