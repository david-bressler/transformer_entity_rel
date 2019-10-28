
#Run one


# #Distilbert best settings
# f1: 0.875
# con_or_lib: 0
# learning_rate: 6.297632766696337e-05
# num_train_epochs: 5
# per_gpu_train_batch_size: 8
# gradient_accumulation_steps: 4
# warmup_steps: 97
# model_name_or_path: distilbert-base-uncased

# #XLNET best settings
# f1: 0.8895705521472393
# con_or_lib: 0
# learning_rate: 7.550040585288487e-06
# num_train_epochs: 9
# per_gpu_train_batch_size: 8
# gradient_accumulation_steps: 2
# warmup_steps: 97
# model_name_or_path: xlnet-base-cased


import torch
from transformers import *
import run_glue_gap
import json
import pandas as pd
import numpy as np
import pickle
import shutil
import time
import os



parser = run_glue_gap.argparse.ArgumentParser()
args = run_glue_gap.get_args(parser)

#datasetta=1 # 0: Gap, 1: entrel
model_typea=1 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert, 4: XLM, 5:TransformerXL
small_or_big=0 # 0: small, 1: big
args.con_or_lib=0 # 0: con, 1: lib, 2: original
batch_size=16

args.num_train_epochs=9.0
args.do_train=True
args.do_eval=True

#args.max_steps=3
args.overwrite_output_dir=True
args.overwrite_cache = True
args.evaluate_during_training=False
args.logging_steps =50
args.save_steps=1000
args.max_seq_length = 512
args.seed=round(time.time())

args.learning_rate =1e-5
args.adam_epsilon=1e-8
args.warmup_steps= 100  
args.weight_decay= 0.0

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

# if datasetta==0:
#     args.data_dir='/home/ec2-user/SageMaker/Data/gap_data/'
#     args.output_dir='/tmp/gap/'
#     args.task_name='gap'
# elif datasetta==1:
    
if args.con_or_lib==0:
    args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_con'
elif args.con_or_lib==1:
    args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_lib'
elif args.con_or_lib==2:
    args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data'


args.output_dir='/tmp/entrel/'
args.task_name='entityrel'

#REMOVE EXISTING WEIGHTS:
if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)

#args.per_gpu_eval_batch_size=2
#args.per_gpu_train_batch_size=2
#args.gradient_accumulation_steps=16

results=run_glue_gap.run_main(parser,args)



