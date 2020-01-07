

#cp tokenization_utils.py /home/ec2-user/anaconda3/envs/JupyterSystemEnv/lib/python3.6/site-packages/transformers/tokenization_utils.py

import torch
from transformers import *
import run_glue_gap
import json
import pandas as pd
import numpy as np
import pickle
import pdb
import os
import shutil
import time


args_list=[]
results_list=[]

num_trials=1000

for trial_num in range(num_trials):
    
    parser = run_glue_gap.argparse.ArgumentParser()
    args = run_glue_gap.get_args(parser)
    
    #model_typea=2 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert
    model_typea=run_glue_gap.gen_grid_val([1,2],'sel')
    #small_or_big=run_glue_gap.gen_grid_val([0,1],'sel') # 0: small, 1: big
    small_or_big=0 # 0: small, 1: big
    #args.con_or_lib=run_glue_gap.gen_grid_val([0,1],'sel') # 0: con, 1: lib
    args.con_or_lib=0 # 0: con, 1: lib
    batch_size=run_glue_gap.gen_grid_val([8,16,32],'sel')
    
    args.num_train_epochs=run_glue_gap.gen_grid_val([5,7,9,11],'sel')
    #args.num_train_epochs=run_glue_gap.gen_grid_val([1,2],'sel')
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
    
    #args.learning_rate =run_glue_gap.gen_grid_val([1e-6,5e-5],'exp')
    args.learning_rate =run_glue_gap.gen_grid_val([4e-6,6e-5],'exp')
    args.adam_epsilon= 1e-8
    args.warmup_steps= run_glue_gap.gen_grid_val([50,150],'lin_round')
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
    
    if args.con_or_lib==0:
        #args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_con'
        args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_cheat2'
    else:
        args.data_dir='/home/ec2-user/SageMaker/Data/entrel_data_lib'
    args.output_dir='/tmp/entrel/'
    args.task_name='entityrel'
    
    results=run_glue_gap.run_main(parser,args)
    
    args_list.append(args)
    results_list.append(results)
    
    outputta=(results_list,args_list)
    pickle.dump(outputta, open("outputta.pickle","wb"))
    shutil.rmtree(args.output_dir)

   