import os
import torch
import copy
import math
import numpy as np
import itertools
from tensorboardX import SummaryWriter
from joblib import Parallel, delayed
from tqdm import tqdm
import torch.nn.functional as F
from src.asr import Seq2Seq
from src.dataset import LoadDataset
from src.postprocess import Mapper,cal_acc,cal_cer,draw_att

import editdistance as ed
import json
import time

VAL_STEP = 30        # Additional Inference Timesteps to run during validation (to calculate CER)
TRAIN_WER_STEP = 250 # steps for debugging info.
GRAD_CLIP = 5


class Solver():
    ''' Super class Solver for all kinds of tasks'''
    def __init__(self,config,paras):
        # General Settings
        self.config = config
        self.paras = paras
        self.device = torch.device('cuda') if (self.paras.gpu and torch.cuda.is_available()) else torch.device('cpu')

        self.exp_name = paras.name
        if self.exp_name is None:
            self.exp_name = '_'.join([paras.config.split('/')[-1].replace('.yaml',''),'sd'+str(paras.seed)])
        if not os.path.exists(paras.ckpdir):os.makedirs(paras.ckpdir)
        self.ckpdir = os.path.join(paras.ckpdir,self.exp_name)
        if not os.path.exists(self.ckpdir):os.makedirs(self.ckpdir)
        
        # Load Mapper for idx2token
        self.mapper = Mapper(config['solver']['data_path'])

    def verbose(self,msg):
        ''' Verbose function for print information to stdout'''
        if self.paras.verbose:
            print('[INFO]',msg)
   
    def progress(self,msg):
        ''' Verbose function for updating progress on stdout'''
        if self.paras.verbose:
            print(msg+'                              ',end='\r')


class Trainer(Solver):
    ''' Handler for complete training progress'''
    def __init__(self,config,paras):
        super(Trainer, self).__init__(config,paras)
        # Logger Settings
        self.logdir = os.path.join(paras.logdir,self.exp_name)
        self.log = SummaryWriter(self.logdir)
        self.valid_step = config['solver']['dev_step']
        self.best_val_ed = 2.0
        # Add
        self.best_total_loss = 10000000.0

        # Training details
        self.step = 0
        self.n_epochs = 0
        self.max_step = config['solver']['total_steps']
        self.tf_start = config['solver']['tf_start']
        self.tf_end = config['solver']['tf_end']
        self.apex = config['solver']['apex']

    def load_data(self):
        ''' Load date for training/validation'''
        # self.verbose('Loading data from '+self.config['solver']['data_path'])
        setattr(self,'train_set',LoadDataset('train',text_only=False,spec_aug=self.paras.spec_aug,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,spec_aug=False,use_gpu=self.paras.gpu,**self.config['solver']))

        # Get 1 example for auto constructing model
        for self.sample_x,_ in getattr(self,'train_set'):break
        if len(self.sample_x.shape)==4: self.sample_x=self.sample_x[0]

    def set_model(self):
        ''' Setup ASR '''
        self.verbose('Init ASR model. Note: validation is done through greedy decoding w/ attention decoder.')

        # Build attention end-to-end ASR
        self.asr_model = Seq2Seq(self.sample_x,self.mapper.get_dim(),self.config['asr_model']).to(self.device)
        self.verbose('ASR model : {}.'.format(self.asr_model))

        if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
            self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')

        # Involve CTC
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']

        # TODO: load pre-trained model
        if self.paras.load:
            raise NotImplementedError

        # Setup optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)

    def load_pre_model(self):
        ''' Load saved ASR'''
        self.verbose('Load pre ASR model from ' + self.paras.continue_from)
        self.asr_model = torch.load(self.paras.continue_from)
        self.verbose('Load ASR model : {}.'.format(self.asr_model))
        # self.asr_model = Seq2Seq(self.sample_x, self.mapper.get_dim(), self.config['asr_model']).to(self.device)
        # self.verbose('Set ASR model : {}.'.format(self.asr_model))

        if 'VGG' in self.config['asr_model']['encoder']['enc_type']:
            self.verbose('VCC Extractor in Encoder is enabled, time subsample rate = 4.')
        self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(self.device)#, reduction='none')

        # Involve CTC
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
        self.ctc_weight = self.config['asr_model']['optimizer']['joint_ctc']


        # Setup optimizer
        if self.apex and self.config['asr_model']['optimizer']['type']=='Adam':
            import apex
            self.asr_opt = apex.optimizers.FusedAdam(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'])
        else:
            self.asr_opt = getattr(torch.optim,self.config['asr_model']['optimizer']['type'])
            self.asr_opt = self.asr_opt(self.asr_model.parameters(), lr=self.config['asr_model']['optimizer']['learning_rate'],eps=1e-8)
        self.asr_model.train()

    def exec(self):
        ''' Training End-to-end ASR system'''
        self.verbose('Training set total '+str(len(self.train_set))+' batches.')

        while self.step< self.max_step:
            # self.verbose('epochs '+ str(n_epochs))
            for x,y in self.train_set:
                self.progress('Training step - '+str(self.step) + '  ' + str(self.n_epochs))

                # Perform teacher forcing rate decaying
                tf_rate = self.tf_start - self.step*(self.tf_start-self.tf_end)/self.max_step
                
                # Hack bucket, record state length for each uttr, get longest label seq for decode step
                assert len(x.shape)==4,'Bucketing should cause acoustic feature to have shape 1xBxTxD'
                assert len(y.shape)==3,'Bucketing should cause label have to shape 1xBxT'
                x = x.squeeze(0).to(device = self.device,dtype=torch.float32)
                y = y.squeeze(0).to(device = self.device,dtype=torch.long)
                state_len = np.sum(np.sum(x.cpu().data.numpy(),axis=-1)!=0,axis=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # ASR forwarding 
                self.asr_opt.zero_grad()
                ctc_pred, state_len, att_pred, _ =  self.asr_model(x, ans_len,tf_rate=tf_rate,teacher=y,state_len=state_len)

                # Calculate loss function
                loss_log = {}
                label = y[:,1:ans_len+1].contiguous()
                ctc_loss = 0
                att_loss = 0
                
                # CE loss on attention decoder
                if self.ctc_weight<1:
                    b,t,c = att_pred.shape
                    att_loss = self.seq_loss(att_pred.view(b*t,c),label.view(-1))
                    att_loss = torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(y!=0,dim=-1)\
                               .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                    att_loss = torch.mean(att_loss) # Mean by batch
                    loss_log['train_att'] = att_loss

                # CTC loss on CTC decoder
                if self.ctc_weight>0:
                    target_len = torch.sum(y!=0,dim=-1)
                    ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, torch.LongTensor(state_len), target_len)
                    loss_log['train_ctc'] = ctc_loss
                
                asr_loss = (1-self.ctc_weight)*att_loss+self.ctc_weight*ctc_loss
                loss_log['train_full'] = asr_loss

                # Backprop
                asr_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.asr_model.parameters(), GRAD_CLIP)
                if math.isnan(grad_norm):
                    if self.step > 200000:
                        self.verbose('Error : grad norm is NaN @ step ' + str(self.step))
                else:
                    self.asr_opt.step()
                
                # Logger
                self.write_log('loss',loss_log)
                if self.ctc_weight<1:
                    self.write_log('acc',{'train':cal_acc(att_pred,label)})
                if self.step % TRAIN_WER_STEP ==0:
                    self.write_log('error rate',
                                   {'train':cal_cer(att_pred,label,mapper=self.mapper)})

                # Validation
                if self.step%self.valid_step == 0:
                    self.asr_opt.zero_grad()
                    self.valid()

                self.step+=1
                if self.step > self.max_step:break

            # Save each epoch
            torch.save(self.asr_model, os.path.join(self.ckpdir, 'asr_{}'.format(self.n_epochs)))
            self.n_epochs += 1

            if self.paras.spec_aug and (self.n_epochs%10 == 0):
                # self.verbose('Spec data')
                self.load_data()
    

    def write_log(self,val_name,val_dict):
        '''Write log to TensorBoard'''
        if 'att' in val_name:
            self.log.add_image(val_name,val_dict,self.step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, self.step)
        else:
            self.log.add_scalars(val_name,val_dict,self.step)


    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding with Attention decoder only)'''
        self.asr_model.eval()
        
        # Init stats
        val_loss, val_ctc, val_att, val_acc, val_cer = 0.0, 0.0, 0.0, 0.0, 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        
        # Perform validation
        for cur_b,(x,y) in enumerate(self.dev_set):
            self.progress(' '.join(['Valid step -',str(self.step),'(',str(cur_b),'/',str(len(self.dev_set)),')']))

            # Prepare data
            if len(x.shape)==4: x = x.squeeze(0)
            if len(y.shape)==3: y = y.squeeze(0)
            x = x.to(device = self.device,dtype=torch.float32)
            y = y.to(device = self.device,dtype=torch.long)
            state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
            state_len = [int(sl) for sl in state_len]
            ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))
            
            # Forward
            ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)

            # Compute attention loss & get decoding results
            label = y[:,1:ans_len+1].contiguous()
            if self.ctc_weight<1:
                seq_loss = self.seq_loss(att_pred[:,:ans_len,:].contiguous().view(-1,att_pred.shape[-1]),label.view(-1))
                seq_loss = torch.sum(seq_loss.view(x.shape[0],-1),dim=-1)/torch.sum(y!=0,dim=-1)\
                           .to(device = self.device,dtype=torch.float32) # Sum each uttr and devide by length
                seq_loss = torch.mean(seq_loss) # Mean by batch
                val_att += seq_loss.detach()*int(x.shape[0])
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_acc += cal_acc(att_pred,label)*int(x.shape[0])
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
            
            # Compute CTC loss
            if self.ctc_weight>0:
                target_len = torch.sum(y!=0,dim=-1)
                ctc_loss = self.ctc_loss( F.log_softmax( ctc_pred.transpose(0,1),dim=-1), label, 
                                         torch.LongTensor(state_len), target_len)
                val_ctc += ctc_loss.detach()*int(x.shape[0])

            val_len += int(x.shape[0])
        
        # Logger
        val_loss = (1-self.ctc_weight)*val_att + self.ctc_weight*val_ctc
        loss_log = {}
        for k,v in zip(['dev_full','dev_ctc','dev_att'],[val_loss, val_ctc, val_att]):
            if v > 0.0: loss_log[k] = v/val_len
        self.write_log('loss',loss_log)
 
        if self.ctc_weight<1:
            # Plot attention map to log
            val_hyp,val_txt = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
            val_attmap = draw_att(att_maps,att_pred)

            # Record loss
            self.write_log('error rate',{'dev':val_cer/val_len})
            self.write_log('acc',{'dev':val_acc/val_len})
            for idx,attmap in enumerate(val_attmap):
                self.write_log('att_'+str(idx),attmap)
                self.write_log('hyp_'+str(idx),val_hyp[idx])
                self.write_log('txt_'+str(idx),val_txt[idx])

            # Save model by val er.
            if val_cer/val_len  < self.best_val_ed:
                self.best_val_ed = val_cer/val_len
                self.verbose('Best val er       : {:.4f}       @ step {}\t  LOSS: {:.6f}\t ctc:{:.6f}\t att:{:.6f} '.format(self.best_val_ed,self.step,val_loss, val_ctc, val_att))
                torch.save(self.asr_model, os.path.join(self.ckpdir,'asr'))
                # add
                torch.save(self.asr_model, os.path.join(self.ckpdir, 'asr_{}'.format(self.step)))
                # Save hyps.
                with open(os.path.join(self.ckpdir,'best_hyp.txt'),'w') as f:
                    for t1,t2 in zip(all_pred,all_true):
                        f.write(t1+','+t2+'\n')


        self.asr_model.train()


class Tester(Solver):
    ''' Handler for complete inference progress'''
    def __init__(self,config,paras):
        super(Tester, self).__init__(config,paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']
        
        self.decode_file = "_".join(['decode','beam',str(self.config['solver']['decode_beam_size']),
                                     'len',str(self.config['solver']['max_decode_step_ratio'])])

    def load_data(self):
        self.verbose('Loading testing data '+str(self.config['solver']['test_set'])\
                     +' from '+self.config['solver']['data_path'])
        setattr(self,'test_set',LoadDataset('test',text_only=False,spec_aug=False,use_gpu=self.paras.gpu,**self.config['solver']))
        setattr(self,'dev_set',LoadDataset('dev',text_only=False,spec_aug=False,use_gpu=self.paras.gpu,**self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from '+os.path.join(self.ckpdir))
        self.asr_model = torch.load(os.path.join(self.ckpdir,'asr'))

        # Enable joint CTC decoding
        self.asr_model.joint_ctc = self.config['solver']['decode_ctc_weight'] >0
        if self.config['solver']['decode_ctc_weight'] >0:
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.verbose('Joint CTC decoding is enabled with weight = '+str(self.config['solver']['decode_ctc_weight']))
            self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
            self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']
        
        # Check models dev performance before inference
        self.asr_model.eval()
        self.asr_model.clear_att()
        self.asr_model = self.asr_model.to(self.device)
        self.verbose('Checking models performance on dev set '+str(self.config['solver']['dev_set'])+'...')
        self.valid()
        self.asr_model = self.asr_model.to('cpu') # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        start_time = time.time()
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = '+str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set),self.njobs))
        ## self.test_set = [(x,y) for (x,y) in self.test_set][::10]
        _ = Parallel(n_jobs=self.njobs)(delayed(self.beam_decode)(x[0],y[0].tolist()[0]) for x,y in tqdm(self.test_set))
        
        self.verbose('Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir,self.decode_file+'.txt'))))
        
        self.verbose('Top {} results at {}.'.format(self.config['solver']['decode_beam_size'],
                                                    str(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'))))
        end_time = time.time()
        time_elapsed = end_time - start_time
        print('time_elapsed ',time_elapsed )
        
    def write_hyp(self,hyps,y):
        '''Record decoding results'''
        gt = self.mapper.translate(y,return_string=True)
        # Best
        with open(os.path.join(self.ckpdir,self.decode_file+'.txt'),'a') as f:
            best_hyp = self.mapper.translate(hyps[0].outIndex,return_string=True)
            f.write(gt+'\t'+best_hyp+'\n')
        # N best
        with open(os.path.join(self.ckpdir,self.decode_file+'_nbest.txt'),'a') as f:
            for hyp in hyps:
                best_hyp = self.mapper.translate(hyp.outIndex,return_string=True)
                f.write(gt+'\t'+best_hyp+'\n')
        

    def beam_decode(self,x,y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device = self.device,dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step =  int(np.ceil(state_len[0]*self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model
        
        self.write_hyp(hyps,y)
        del hyps
        
        return 1

    
    def valid(self):
        '''Perform validation step (!!!NOTE!!! greedy decoding on Attention decoder only)'''
        val_cer = 0.0
        val_len = 0    
        all_pred,all_true = [],[]
        ctc_results = []
        with torch.no_grad():
            for cur_b,(x,y) in enumerate(self.dev_set):
                self.progress(' '.join(['Valid step - (',str(cur_b),'/',str(len(self.dev_set)),')']))

                # Prepare data
                if len(x.shape)==4: x = x.squeeze(0)
                if len(y.shape)==3: y = y.squeeze(0)
                x = x.to(device = self.device,dtype=torch.float32)
                y = y.to(device = self.device,dtype=torch.long)
                state_len = torch.sum(torch.sum(x.cpu(),dim=-1)!=0,dim=-1)
                state_len = [int(sl) for sl in state_len]
                ans_len = int(torch.max(torch.sum(y!=0,dim=-1)))

                # Forward
                ctc_pred, state_len, att_pred, att_maps = self.asr_model(x, ans_len+VAL_STEP,state_len=state_len)
                ctc_pred = torch.argmax(ctc_pred,dim=-1).cpu() if ctc_pred is not None else None
                ctc_results.append(ctc_pred)

                # Result
                label = y[:,1:ans_len+1].contiguous()
                t1,t2 = cal_cer(att_pred,label,mapper=self.mapper,get_sentence=True)
                all_pred += t1
                all_true += t2
                val_cer += cal_cer(att_pred,label,mapper=self.mapper)*int(x.shape[0])
                val_len += int(x.shape[0])
        
        
        # Dump model score to ensure model is corrected
        self.verbose('Validation Error Rate of Current model : {:.4f}      '.format(val_cer/val_len)) 
        self.verbose('See {} for validation results.'.format(os.path.join(self.ckpdir,'dev_att_decode.txt'))) 
        with open(os.path.join(self.ckpdir,'dev_att_decode.txt'),'w') as f:
            for hyp,gt in zip(all_pred,all_true):
                f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')
        
        # Also dump CTC result if available
        if ctc_results[0] is not None:
            ctc_results = [i for ins in ctc_results for i in ins]
            ctc_text = []
            for pred in ctc_results:
                p = [i for i in pred.tolist() if i != 0]
                p = [k for k, g in itertools.groupby(p)]
                ctc_text.append(self.mapper.translate(p,return_string=True))
            self.verbose('Also, see {} for CTC validation results.'.format(os.path.join(self.ckpdir,'dev_ctc_decode.txt'))) 
            with open(os.path.join(self.ckpdir,'dev_ctc_decode.txt'),'w') as f:
                for hyp,gt in zip(ctc_text,all_true):
                    f.write(gt.lstrip()+'\t'+hyp.lstrip()+'\n')


class Test_iter(Solver):
    ''' Handler for complete inference progress'''

    def __init__(self, config, paras):
        super(Test_iter, self).__init__(config, paras)
        self.verbose('During beam decoding, batch size is set to 1, please speed up with --njobs.')
        self.njobs = self.paras.njobs
        self.decode_step_ratio = config['solver']['max_decode_step_ratio']

        self.decode_file = "_".join(['decode', 'beam', str(self.config['solver']['decode_beam_size']),
                                     'len', str(self.config['solver']['max_decode_step_ratio']),str(self.paras.model)])
        self.cer = []
        self.wer = []
        self.time_elapsed = 0

    def load_data(self):
        self.verbose('Loading testing data ' + str(self.config['solver']['test_set']) \
                     + ' from ' + self.config['solver']['data_path'])
        setattr(self, 'test_set',
                LoadDataset('test', text_only=False, spec_aug=False, use_gpu=self.paras.gpu, **self.config['solver']))

    def set_model(self):
        ''' Load saved ASR'''
        self.verbose('Load ASR model from ' + os.path.join(self.ckpdir, self.paras.model))
        self.asr_model = torch.load(os.path.join(self.ckpdir, self.paras.model))
        self.asr_model.eval()

        # Enable joint CTC decoding
        self.asr_model.joint_ctc = self.config['solver']['decode_ctc_weight'] > 0
        if self.config['solver']['decode_ctc_weight'] > 0:
            assert self.asr_model.joint_ctc, "The ASR was not trained with CTC"
            self.verbose(
                'Joint CTC decoding is enabled with weight = ' + str(self.config['solver']['decode_ctc_weight']))
            self.decode_file += '_ctc{:}'.format(self.config['solver']['decode_ctc_weight'])
            self.asr_model.ctc_weight = self.config['solver']['decode_ctc_weight']

        self.asr_model = self.asr_model.to('cpu')  # move origin model to cpu, clone it to GPU for each thread

    def exec(self):
        '''Perform inference step with beam search decoding.'''
        start_time = time.time()
        test_cer = 0.0
        self.decode_beam_size = self.config['solver']['decode_beam_size']
        self.verbose('Start decoding with beam search, beam size = ' + str(self.config['solver']['decode_beam_size']))
        self.verbose('Number of utts to decode : {}, decoding with {} threads.'.format(len(self.test_set), self.njobs))

        _ = Parallel(n_jobs=self.njobs)(
            delayed(self.beam_decode)(x[0], y[0].tolist()[0]) for x, y in tqdm(self.test_set))

        self.verbose(
            'Decode done, best results at {}.'.format(str(os.path.join(self.ckpdir, self.decode_file + '.txt'))))

        end_time = time.time()
        self.time_elapsed = end_time - start_time
        print('time_elapsed ', self.time_elapsed)
        self.write_cer()

    def write_hyp(self, hyps, y):
        '''Record decoding results'''
        gt = self.mapper.translate(y, return_string=True)

        decode_path = os.path.join(self.ckpdir, 'decode')
        if not os.path.exists(decode_path): os.makedirs(decode_path)
        with open(os.path.join(decode_path,self.decode_file + '.txt'), 'a') as f:
            best_hyp = self.mapper.translate(hyps[0].outIndex, return_string=True)
            f.write(gt + '\t' + best_hyp + '\n')

        with open(self.paras.C2E) as label_file:
            C2E = json.load(label_file)

        result_path = os.path.join(self.ckpdir, 'CER')
        if not os.path.exists(result_path): os.makedirs(result_path)
        with open(os.path.join(result_path, 'CER_' + self.paras.model + '.txt'), 'a') as f:
            for j in C2E.keys():
                if j in gt:
                    gt = gt.replace(j, C2E[j])
                if j in best_hyp:
                    best_hyp = best_hyp.replace(j, C2E[j])
            f.write("gt:" + gt.replace(" ","") + '\n')
            f.write("pd:" + best_hyp.replace(" ","") + '\n')
            f.write("CER:" + str(ed.eval(best_hyp.split(' '), gt.split(' ')) / len(gt.split(' '))) + '\n')
            print('gt: ', gt.replace(" ",""))
            print('pd: ', best_hyp.replace(" ",""))
            print("CER:" + str(ed.eval(best_hyp.split(' '), gt.split(' ')) / len(gt.split(' '))))

            self.wer.append(ed.eval(best_hyp.split(' '), gt.split(' ')) / len(gt.split(' ')))
            self.cer.append(ed.eval(best_hyp, gt) / len(gt))

    def write_cer(self):
        with open(os.path.join(self.ckpdir, 'CER', 'CER_' + self.paras.model + '.txt'), 'a') as f:
            print('CER : {:.6f}'.format(sum(self.cer) / len(self.cer)))
            print('WER : {:.6f}'.format(sum(self.wer) / len(self.wer)))
            f.write('Test Summary \t' +
                    'Average CER {:.6f}\t'.format(sum(self.wer) / len(self.wer)) + '\n')
            f.write('time_elapsed ' + str(self.time_elapsed))

    def beam_decode(self, x, y):
        '''Perform beam decoding with end-to-end ASR'''
        # Prepare data
        x = x.to(device=self.device, dtype=torch.float32)
        state_len = torch.sum(torch.sum(x.cpu(), dim=-1) != 0, dim=-1)
        state_len = [int(sl) for sl in state_len]

        # Forward
        with torch.no_grad():
            max_decode_step = int(np.ceil(state_len[0] * self.decode_step_ratio))
            model = copy.deepcopy(self.asr_model).to(self.device)
            hyps = model.beam_decode(x, max_decode_step, state_len, self.decode_beam_size)
        del model

        self.write_hyp(hyps, y)
        del hyps

        return 1