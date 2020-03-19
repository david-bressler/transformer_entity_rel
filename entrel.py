



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
        self.company_train='contexts_samples_MT_train_new'
        self.company_test='contexts_samples_MT_test_new'
        self.person_train='contexts_samples_PERSON_train'
        self.person_test='contexts_samples_PERSON_test'
        self.company_train_mixed='mixmodel_samples_company_train'
        self.company_test_mixed='mixmodel_samples_company_test'
        self.person_train_mixed='mixmodel_samples_person_train'
        self.person_test_mixed='mixmodel_samples_person_test'
    
    def open_file(self,fn):
        with open(fn,'r') as f:
            the_dic=json.loads(f.read())
        return the_dic
    
    def select_sequence(self, article, add_tag=0, tagga='' ):
        con_body = article["conservative-marked-body"].replace("\n", "")
        con_seq = article["conservative-sequence"].replace("\n", "")
        lib_body = article["liberal-marked-body"].replace("\n", "")
        lib_seq = article["liberal-sequence"].replace("\n", "")
        if add_tag==1:
            if len(con_body)>0:
                con_body= '<' + tagga + ' CON BOD> ' + con_body
            if len(con_seq)>0:
                con_seq= '<' + tagga + ' CON SEQ> ' + con_seq
            if len(lib_body)>0:
                lib_body= '<' + tagga + ' CON BOD> ' + lib_body
            if len(lib_seq)>0:
                lib_seq= '<' + tagga + ' LIB SEQ> ' + lib_seq
        lib_choice = ""
        con_choice = ""
        if (len(lib_seq) > 0 and len(lib_body) > 0):  # if both lib_seq and lib_body not empty
            if len(lib_body.split()) < len(lib_seq.split()):  # choose the shorter bw lib_body and lib_seq to append to lib
                lib_choice = lib_body
            else:
                lib_choice = lib_seq
        else:  # if only one or neither are not empty
            lib_choice=lib_seq if len(lib_seq)>0 else lib_body if len(lib_body)>0 else lib_choice
        if (len(con_seq) > 0 and len(con_body) > 0):  # if con_seq and con_body not empty
            if len(con_body.split()) < len(con_seq.split()):  # choose the shorter bw con_body and con_seq to append to con
                con_choice = con_body
            else:
                con_choice = con_seq
        else:  # if only one or neither are not empty
            con_choice=con_seq if len(con_seq)>0 else con_body if len(con_body)>0 else con_choice
        if len(con_choice) == 0:
            con_choice = lib_choice
        if len(lib_choice) == 0:
            lib_choice = con_choice
        if len(con_choice) == 0 and len(lib_choice) == 0:
            con_choice = "<BAD INPUT> asdfasdfasdfasdf"
            lib_choice = "<BAD INPUT> asdfasdfasdfasdf"
        return con_choice

    def display_contents(self,fn):
        print('FILENAME: ' + fn)
        the_dic=self.open_file(fn)
        print('NUMBER OF DATAPOINTS: ' + str(len(the_dic)))
        print('KEYS: ')
        print(the_dic[0].keys())
        print('QUERY_NAME:')
        print(the_dic[0]['query_name'])
        print('LABEL')
        print(the_dic[0]['entity_relevance'])
        print('ARTICLE BODY:')
        print(the_dic[0]['article_body'])
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
        self.args=self.get_args()
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
        fn=self.args.base_dir+'/data/mixmodel_samples_company_train.json'
        self.display_contents(fn)
        fn=self.args.base_dir+'/data/mixmodel_samples_company_test.json'
        self.display_contents(fn)
        fn=self.args.base_dir+'/data/mixmodel_samples_person_train.json'
        self.display_contents(fn)
        fn=self.args.base_dir+'/data/mixmodel_samples_person_test.json'
        self.display_contents(fn)
        

    def create_dataset(self,in_fn,out_fn,splitta,do_shuffle=0,sampla=1,add_tag=0,tagga=''):
        the_dic=self.open_file(in_fn)
        the_targets=[docca['entity_relevance'] for docca in the_dic]
        conservative_inputs=[]
        for doc_ind, docca in enumerate(the_dic):
            conservative_inputs.append(self.select_sequence(docca,add_tag,tagga))
        con_data = pd.DataFrame(list(zip(the_targets, conservative_inputs)), columns =['targets', 'texts']) 
        if do_shuffle==1:
            con_data = con_data.sample(frac=1).reset_index(drop=True)#shuffle
        if splitta>0:
            msk = np.random.rand(len(con_data)) < splitta
            con_train_data = con_data[msk].reset_index(drop=True)
            con_test_data = con_data[~msk].reset_index(drop=True)
        else:
            con_train_data = con_data.copy()
            con_test_data = con_data.copy()
        #
        if sampla<1:
            con_train_data=con_train_data.sample(frac=sampla)
            con_test_data=con_test_data.sample(frac=sampla)
        print(np.shape(con_test_data))
        print(np.shape(con_train_data))
        if os.path.isdir(out_fn):
            shutil.rmtree(out_fn)
        os.mkdir(out_fn)
        con_train_data.to_csv(out_fn+'train.tsv',sep='\t',index=False,header=False)
        con_test_data.to_csv(out_fn+'dev.tsv',sep='\t',index=False,header=False)
        print(con_train_data.head())
    
    def create_composite_ds(self,train_path,test_path,target):
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.mkdir(target)
        shutil.copyfile(train_path+'train.tsv', target+'train.tsv')
        shutil.copyfile(test_path+'dev.tsv', target+'dev.tsv')

    def combine_datasets(self,path1,path2,target):
        if os.path.isdir(target):
            shutil.rmtree(target)
        os.mkdir(target)
        df1=pd.read_csv(path1+'train.tsv',sep='\t',header=None)
        df2=pd.read_csv(path2+'train.tsv',sep='\t',header=None)
        df3 = df1.append(df2, ignore_index=True)
        df3 = df3.sample(frac=1).reset_index(drop=True)#shuffle
        df3.to_csv(target+'train.tsv',sep='\t',index=False,header=False)
        print(np.shape(df1),np.shape(df2),np.shape(df3))
        df1=pd.read_csv(path1+'dev.tsv',sep='\t',header=None)
        df2=pd.read_csv(path2+'dev.tsv',sep='\t',header=None)
        df3 = df1.append(df2, ignore_index=True)
        df3.to_csv(target+'dev.tsv',sep='\t',index=False,header=False)
        print(np.shape(df1),np.shape(df2),np.shape(df3))

    def create_datasets_mixed(self):
        self.args=self.get_args()
        self.create_dataset(self.args.base_dir+'/data/'+self.company_train_mixed+'.json',self.args.base_dir+'/data/company_mixed/', 0,do_shuffle=0,add_tag=1,tagga='COM')
        print('CREATED COMPANY TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_train_mixed+'.json',self.args.base_dir+'/data/person_mixed/', 0,do_shuffle=0,add_tag=1,tagga='PER')
        print('CREATED PERSON TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.company_test_mixed+'.json',self.args.base_dir+'/data/company_test_mixed/', 0,do_shuffle=0,add_tag=1,tagga='COM')
        print('CREATED COMPANY TEST TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_test_mixed+'.json',self.args.base_dir+'/data/person_test_mixed/', 0,do_shuffle=0,add_tag=1,tagga='PER')
        print('CREATED PERSON TEST TSV')
        path1=self.args.base_dir+'/data/company_mixed/'
        path2=self.args.base_dir+'/data/person_mixed/'
        target=self.args.base_dir+'/data/company_person_mixed/'
        self.combine_datasets(path1,path2,target)
        print('CREATED COMBINED COMPANY PERSON TRAIN/EVAL TSV')
        path1=self.args.base_dir+'/data/company_test_mixed/'
        path2=self.args.base_dir+'/data/person_test_mixed/'
        target=self.args.base_dir+'/data/company_person_test_mixed/'
        self.combine_datasets(path1,path2,target)
        print('CREATED COMBINED COMPANY PERSON TEST TSV')
        train_path=self.args.base_dir+'/data/company_person_mixed/'
        test_path=self.args.base_dir+'/data/company_person_test_mixed/'
        target=self.args.base_dir+'/data/company_person_fulltrainwtest_mixed/'
        self.create_composite_ds(train_path,test_path,target)
        print('CREATED COMPOSITE COMPANY_PERSON FULL TRAIN TSV WITH TEST TSV')

    def create_datasets(self):
        self.args=self.get_args()
        self.create_dataset(self.args.base_dir+'/data/'+self.company_train+'.json',self.args.base_dir+'/data/company/', 0,do_shuffle=0)
        print('CREATED COMPANY TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_train+'.json',self.args.base_dir+'/data/person/', 0,do_shuffle=0)
        print('CREATED PERSON TRAIN/EVAL TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.company_test+'.json',self.args.base_dir+'/data/company_test/', 0,do_shuffle=0)
        print('CREATED COMPANY TEST TSV')
        self.create_dataset(self.args.base_dir+'/data/'+self.person_test+'.json',self.args.base_dir+'/data/person_test/', 0,do_shuffle=0)
        print('CREATED PERSON TEST TSV')
        path1=self.args.base_dir+'/data/company/'
        path2=self.args.base_dir+'/data/person/'
        target=self.args.base_dir+'/data/company_person/'
        self.combine_datasets(path1,path2,target)
        print('CREATED COMBINED COMPANY PERSON TRAIN/EVAL TSV')
        path1=self.args.base_dir+'/data/company_test/'
        path2=self.args.base_dir+'/data/person_test/'
        target=self.args.base_dir+'/data/company_person_test/'
        self.combine_datasets(path1,path2,target)
        print('CREATED COMBINED COMPANY PERSON TEST TSV')
        train_path=self.args.base_dir+'/data/company_person/'
        test_path=self.args.base_dir+'/data/company_person_test/'
        target=self.args.base_dir+'/data/company_person_fulltrainwtest/'
        self.create_composite_ds(train_path,test_path,target)
        print('CREATED COMPOSITE COMPANY_PERSON FULL TRAIN TSV WITH TEST TSV')

    def get_args(self):
        self.parser = run_glue_gap.argparse.ArgumentParser()
        args = run_glue_gap.get_args(self.parser)
        args.task_name='entityrel'
        args.base_dir=os.getcwd()
        args.output_dir=args.base_dir+'/data/tmp/'
        args.local_rank=6 #this sets which gpu to use; set to -1 for default
        #args.cuda_devices=[6,7] #DWB: couldnt get this working
        model_typea=2 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert, 4: XLM, 5:TransformerXL
        small_or_big=0 # 0: small, 1: big
        args.con_or_lib=0 # 0: con, 1: lib, 2: original
        batch_size=8
        args.num_train_epochs=3.0
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
        args.learning_rate = 1.95e-5# 2.83e-5 #2.27e-5
        args.adam_epsilon=1e-8
        args.warmup_steps= 50 #67 # 86  
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
        return args

    def eval_mult(self,ds_names, eval_names):
        the_dics=[]
        for ds_name in ds_names:
            for eval_name in eval_names:
                the_dic=self.eval_one(ds_name, eval_name)
                the_dic['matchup']= ds_name + ' EVALS ' + eval_name
                the_dics.append(the_dic)
        for dicca in the_dics:
            print(dicca['matchup'])
            print('F1' + str(dicca['all_f1s']))
            print('Acc' + str(dicca['all_accs']))
            print('Counts' + str(dicca['all_counts']))
            print('Time' + str(dicca['all_times']))
                
    def eval_one(self, ds_name, eval_name):
        args=self.get_args()
        args.data_dir=args.base_dir+'/data/' + eval_name #this is where we want to eval the 'dev.tsv'
        args.output_dir=args.base_dir+'/data/' + ds_name + '_model' #this is where we want to load the model from
        #
        all_preds=[]
        all_finalpreds=[]
        all_actuals=[]
        all_accs=[]
        all_f1s=[]
        all_times=[]
        all_counts=[]
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
            args.n_gpu = 1
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            #torch.distributed.init_process_group(backend='nccl') #DWB.... this wasn't working on Gibson
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
        #eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset) #DWB
        eval_sampler = SequentialSampler(eval_dataset) 
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
        all_f1s.append(sklearn.metrics.f1_score(all_actuals[-1],all_finalpreds[-1],average='macro'))
        TPs=[inda for inda,bla in enumerate(all_actuals[-1]) if (bla==1.0 and all_finalpreds[-1][inda]==1.0)]
        FNs=[inda for inda,bla in enumerate(all_actuals[-1]) if (bla==1.0 and all_finalpreds[-1][inda]==0.0)]
        TNs=[inda for inda,bla in enumerate(all_actuals[-1]) if (bla==0.0 and all_finalpreds[-1][inda]==0.0)]
        FPs=[inda for inda,bla in enumerate(all_actuals[-1]) if (bla==0.0 and all_finalpreds[-1][inda]==1.0)]
        all_counts.append((len(TPs),len(TNs),len(FPs),len(FNs)))
        all_times.append(time.time()-start_time)
        #sklearn.metrics.f1_score( list(map(float,list(out_label_ids))), list(map(float,list(preds))), average='macro')
        #
        argsbla=args.__dict__.copy()
        argsbla['device']=str(argsbla['device']) #
        the_dic={'all_outputs':all_preds,'all_preds':all_finalpreds,'all_actuals':all_actuals,'all_accs':all_accs, \
                'all_times':all_times, 'all_f1s':all_f1s, 'all_counts':all_counts, 'args':argsbla}
        filename=args.base_dir + '/data/' + ds_name + '_EVALS_' + eval_name + '.json'
        with open(filename, 'w') as outfile:
            json.dump(the_dic, outfile)
        return the_dic

    def analyze_hyper(self,fn):
        outputta = pickle.load( open( fn, "rb" ) )
        #print([outputta[0][inda]['f1_'] for inda in range(len(outputta[0]))])
        f1_list=[outputta[0][inda]['macrof1_'] for inda in range(len(outputta[0]))]
        print(f1_list)
        #max_inds=[inda for inda in range(len(outputta[0])) if outputta[0][inda]['f1_']>0.87]
        max_inds=[inda for inda in range(len(outputta[0])) if outputta[0][inda]['macrof1_']>0.9]
        min_inds=[inda for inda in range(len(outputta[0])) if outputta[0][inda]['macrof1_']<0.9]
        for inda in max_inds:
            #print(outputta[1][inda])
            print('macrof1_: ' + str(outputta[0][inda]['macrof1_']))
            #print('con_or_lib: ' + str(outputta[1][inda].con_or_lib))
            #print('the_dataset: ' + str(outputta[1][inda].the_dataset))
            print('learning_rate: ' + str(outputta[1][inda].learning_rate))
            print('num_train_epochs: ' + str(outputta[1][inda].num_train_epochs))
            print('per_gpu_train_batch_size: ' + str(outputta[1][inda].per_gpu_train_batch_size))
            print('gradient_accumulation_steps: ' + str(outputta[1][inda].gradient_accumulation_steps))
            print('warmup_steps: ' + str(outputta[1][inda].warmup_steps))
            print('model_name_or_path: ' + str(outputta[1][inda].model_name_or_path))
            print('...........')
        for inda in min_inds:
            #print(outputta[1][inda])
            print('macrof1_: ' + str(outputta[0][inda]['macrof1_']))
            #print('con_or_lib: ' + str(outputta[1][inda].con_or_lib))
            #print('the_dataset: ' + str(outputta[1][inda].the_dataset))
            print('learning_rate: ' + str(outputta[1][inda].learning_rate))
            print('num_train_epochs: ' + str(outputta[1][inda].num_train_epochs))
            print('per_gpu_train_batch_size: ' + str(outputta[1][inda].per_gpu_train_batch_size))
            print('gradient_accumulation_steps: ' + str(outputta[1][inda].gradient_accumulation_steps))
            print('warmup_steps: ' + str(outputta[1][inda].warmup_steps))
            print('model_name_or_path: ' + str(outputta[1][inda].model_name_or_path))
            print('...........')

    # def test_hyper_selection(self): #testing if the parameters that are produced look ok
    #     args=self.get_args()
    #     #args.data_dir=args.base_dir+'/data/' + ds_name #this is where we want to load training data and run evaluation
    #     args_list=[]
    #     results_list=[]
    #     num_trials=100
    #     for trial_num in range(num_trials):
    #         args=self.get_args()
    #         #args.n_gpu=0 #this is just necessary to run set_seed. It is re-set later in the code
    #         #run_glue_gap.set_seed(args)
    #         model_typea=2 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert
    #         #model_typea=run_glue_gap.gen_grid_val([1,2],'sel')
    #         #small_or_big=run_glue_gap.gen_grid_val([0,1],'sel') # 0: small, 1: big
    #         small_or_big=0 # 0: small, 1: big
    #         #args.con_or_lib=run_glue_gap.gen_grid_val([0,1],'sel') # 0: con, 1: lib
    #         args.con_or_lib=0 # 0: con, 1: lib
    #         batch_size=run_glue_gap.gen_grid_val([8,16],'sel')
    #         #print(batch_size)
    #         args.num_train_epochs=run_glue_gap.gen_grid_val([1,2],'sel')
    #         #args.num_train_epochs=run_glue_gap.gen_grid_val([1,2],'sel')
    #         args.do_train=True
    #         args.do_eval=True
    #         #args.max_steps=3
    #         args.overwrite_output_dir=True
    #         args.overwrite_cache = True
    #         args.evaluate_during_training=False
    #         args.logging_steps =50
    #         args.save_steps=1000
    #         args.max_seq_length = 512
    #         args.seed=round(time.time())
    #         #args.learning_rate =run_glue_gap.gen_grid_val([1e-6,5e-5],'exp')
    #         args.learning_rate =run_glue_gap.gen_grid_val([4e-6,6e-5],'exp')
    #         print(args.learning_rate)
    #         args.adam_epsilon= 1e-8
    #         args.warmup_steps= run_glue_gap.gen_grid_val([50,150],'lin_round')
    #         args.weight_decay= 0.0
    #         if small_or_big==0:
    #             args.per_gpu_eval_batch_size=8
    #             args.per_gpu_train_batch_size=8
    #             if batch_size==8:
    #                 args.gradient_accumulation_steps=1
    #             elif batch_size==16:
    #                 args.gradient_accumulation_steps=2
    #             elif batch_size==32:
    #                 args.gradient_accumulation_steps=4
    #         elif small_or_big==1:
    #             args.per_gpu_eval_batch_size=1
    #             args.per_gpu_train_batch_size=1
    #             if batch_size==8:
    #                 args.gradient_accumulation_steps=8
    #             elif batch_size==16:
    #                 args.gradient_accumulation_steps=16
    #             elif batch_size==32:
    #                 args.gradient_accumulation_steps=32
    #         if model_typea==0:
    #             args.model_type='bert'
    #             args.do_lower_case=True
    #             if small_or_big==0:
    #                 args.model_name_or_path='bert-base-uncased'
    #             else:
    #                 args.model_name_or_path='bert-large-uncased'
    #         elif model_typea==1:
    #             args.model_type='xlnet'
    #             args.do_lower_case=False
    #             #args.model_name_or_path='xlnet-base-cased'
    #             if small_or_big==0:
    #                 args.model_name_or_path='xlnet-base-cased'
    #             else:
    #                 args.model_name_or_path='xlnet-large-cased'
    #         elif model_typea==2:
    #             args.model_type='roberta'
    #             args.do_lower_case=False
    #             #args.model_name_or_path='roberta-base'
    #             if small_or_big==0:
    #                 args.model_name_or_path='roberta-base'
    #             else:
    #                 args.model_name_or_path='roberta-large'
    #         elif model_typea==3:
    #             args.model_type='distilbert'
    #             args.model_name_or_path='distilbert-base-uncased'
    #             args.do_lower_case=True
    #         elif model_typea==4:
    #             args.model_type='xlm'
    #             args.model_name_or_path='xlm-mlm-en-2048'
    #             args.do_lower_case=False
    #         #results=run_glue_gap.run_main(self.parser,args)
    #         results=[]
    #         args_list.append(args)
    #         results_list.append(results)
    #         outputta=(results_list,args_list)
    #         pickle.dump(outputta, open("outputta.pickle","wb"))
    #         #shutil.rmtree(args.output_dir)



    def run_hypersearch(self,ds_name):
        args_list=[]
        results_list=[]
        num_trials=1000
        for trial_num in range(num_trials):
            args=self.get_args()
            args.data_dir=args.base_dir+'/data/' + ds_name #this is where we want to load training data and run evaluation
            # args.n_gpu=0 #this is just necessary to run set_seed. It is re-set later in the code
            # run_glue_gap.set_seed(args)
            model_typea=2 # 0: BERT, 1: XLNET, 2: Roberta, 3: Distilbert
            #model_typea=run_glue_gap.gen_grid_val([1,2],'sel')
            #small_or_big=run_glue_gap.gen_grid_val([0,1],'sel') # 0: small, 1: big
            small_or_big=0 # 0: small, 1: big
            #args.con_or_lib=run_glue_gap.gen_grid_val([0,1],'sel') # 0: con, 1: lib
            args.con_or_lib=0 # 0: con, 1: lib
            batch_size=run_glue_gap.gen_grid_val([8,16],'sel')
            args.num_train_epochs=run_glue_gap.gen_grid_val([3,5,7],'sel')
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
            #print('LEARNING RATE')
            print(args.learning_rate)
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
            results=run_glue_gap.run_main(self.parser,args)
            args_list.append(args)
            results_list.append(results)
            outputta=(results_list,args_list)
            pickle.dump(outputta, open("outputta.pickle","wb"))
            shutil.rmtree(args.output_dir)





    def train_one(self, ds_name):
        args=self.get_args()
        args.data_dir=args.base_dir+'/data/' + ds_name
        if os.path.isdir(args.output_dir):
            shutil.rmtree(args.output_dir)
        results=run_glue_gap.run_main(self.parser,args)
        model_path=args.base_dir+'/data/' + ds_name + '_model'
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.move(args.output_dir, model_path)
        return results



