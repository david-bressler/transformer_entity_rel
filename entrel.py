



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
import sklearn
import pdb
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


#from entrel import EntityRelevance
#entreller=EntityRelevance()

class EntityRelevance:
    def __init__(self):
        parser = run_glue_gap.argparse.ArgumentParser()
        args = run_glue_gap.get_args(parser)
        args.task_name='entityrel'
        args.base_dir=os.getcwd()
        args.output_dir=args.base_dir+'/data/tmp/'
        self.company_train='contexts_samples_MT_train_new'
        self.company_test='contexts_samples_MT_test_new'
        self.person_train='contexts_samples_PERSON_train'
        self.person_test='contexts_samples_PERSON_test'
        self.args=args
        self.parser=parser
    
    def open_file(self,fn):
        with open(fn,'r') as f:
            the_dic=json.loads(f.read())
        return the_dic
    
    def select_sequence(self, article):
        con_body = article["conservative-marked-body"].replace("\n", "")
        con_seq = article["conservative-sequence"].replace("\n", "")
        lib_body = article["liberal-marked-body"].replace("\n", "")
        lib_seq = article["liberal-sequence"].replace("\n", "")
        lib_choice = ""
        con_choice = ""
        if (
            len(lib_seq) > 0 and len(lib_body) > 0
        ):  # if both lib_seq and lib_body not empty
            if len(lib_body.split()) < len(
                lib_seq.split()
            ):  # choose the shorter bw lib_body and lib_seq to append to lib
                lib_choice = lib_body
            else:
                lib_choice = lib_seq
        else:  # if only one or neither are not empty
            lib_choice = (
                lib_seq
                if len(lib_seq) > 0
                else lib_body
                if len(lib_body) > 0
                else lib_choice
            )
        if (
            len(con_seq) > 0 and len(con_body) > 0
        ):  # if con_seq and con_body not empty
            if len(con_body.split()) < len(
                con_seq.split()
            ):  # choose the shorter bw con_body and con_seq to append to con
                con_choice = con_body
            else:
                con_choice = con_seq
        else:  # if only one or neither are not empty
            con_choice = (
                con_seq
                if len(con_seq) > 0
                else con_body
                if len(con_body) > 0
                else con_choice
            )
        if len(con_choice) == 0:
            con_choice = lib_choice
        if len(lib_choice) == 0:
            lib_choice = con_choice
        if len(con_choice) == 0 and len(lib_choice) == 0:
            con_choice = "asdfasdfasdfasdf"
            lib_choice = "asdfasdfasdfasdf"
        return con_choice

    def display_contents(self,fn):
        print('FILENAME: ' + fn)
        the_dic=self.open_file(fn)
        print('NUMBER OF DATAPOINTS: ' + str(len(the_dic)))
        print('KEYS: ')
        print(the_dic[0].keys())
        print('conservative-sequence: ')
        print(the_dic[0]['conservative-sequence'])
        print('liberal-sequence: ')
        print(the_dic[0]['liberal-sequence'])
        print('conservative-marked-body: ')
        print(the_dic[0]['conservative-marked-body'])
        print('liberal-marked-body: ')
        print(the_dic[0]['liberal-marked-body'])
        print('........................................')

    def explore_data(self):
        # fn=self.args.base_dir+'/data/contexts_samples_person_train.json'
        # self.display_contents(fn)
        fn=self.args.base_dir+'/data/contexts_samples_PERSON_train.json'
        self.display_contents(fn)
        # fn=self.args.base_dir+'/data/contexts_samples_person_test.json'
        # self.display_contents(fn)
        fn=self.args.base_dir+'/data/contexts_samples_PERSON_test.json'
        self.display_contents(fn)
        fn=self.args.base_dir+'/data/contexts_samples_MT_train_new.json'
        self.display_contents(fn)
        fn=self.args.base_dir+'/data/contexts_samples_MT_test_new.json'
        self.display_contents(fn)

    def create_dataset(self,in_fn,out_fn,splitta):
        the_dic=self.open_file(in_fn)
        the_targets=[docca['entity_relevance'] for docca in the_dic]
        conservative_inputs=[]
        for doc_ind, docca in enumerate(the_dic):
            conservative_inputs.append(self.select_sequence(docca))
        con_data = pd.DataFrame(list(zip(the_targets, conservative_inputs)), columns =['targets', 'texts']) 
        print(con_data.head())
        if splitta>0:
            con_data = con_data.sample(frac=1).reset_index(drop=True)#shuffle
            msk = np.random.rand(len(con_data)) < splitta
            con_train_data = con_data[msk].reset_index(drop=True)
            con_test_data = con_data[~msk].reset_index(drop=True)
        else:
            con_train_data = con_data.copy()
            con_test_data = con_data.copy()
        #
        print(np.shape(con_test_data))
        print(np.shape(con_train_data))
        if os.path.isdir(out_fn):
            shutil.rmtree(out_fn)
        os.mkdir(out_fn)
        con_train_data.to_csv(out_fn+'train.tsv',sep='\t',index=False,header=False)
        con_test_data.to_csv(out_fn+'dev.tsv',sep='\t',index=False,header=False)

    def create_datasets(self):
        self.create_dataset(self.args.base_dir+'/data/'+self.company_train+'.json',self.args.base_dir+'/data/company/', 0.9)
        print('CREATED COMPANY TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_train+'.json',self.args.base_dir+'/data/person/', 0.9)
        print('CREATED PERSON TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.company_test+'.json',self.args.base_dir+'/data/company_test/', 0)
        print('CREATED COMPANY TEST TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_test+'.json',self.args.base_dir+'/data/person_test/', 0)
        print('CREATED PERSON TEST TSV')

    def train_one(self, ds_name):
        args=self.args
        args.local_rank=-1 #this sets which gpu to use; set to -1 for default
        args.data_dir=args.base_dir+'/data/' + ds_name
        model_typea=2 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert, 4: XLM, 5:TransformerXL
        small_or_big=0 # 0: small, 1: big
        args.con_or_lib=0 # 0: con, 1: lib, 2: original
        batch_size=8
        args.num_train_epochs=2.0
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
        args.learning_rate =2.27e-5
        args.adam_epsilon=1e-8
        args.warmup_steps= 86  
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
        if os.path.isdir(args.output_dir):
            shutil.rmtree(args.output_dir)
        results=run_glue_gap.run_main(self.parser,args)
        shutil.move(args.output_dir, args.base_dir+'/data/' + ds_name + '_model')
        return results



