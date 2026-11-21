import logging
import time
from contextlib import nullcontext
import torch
from torch .nn .utils import clip_grad_norm_
class Executor :
    def __init__ (self ):
        self .step =0
    def train (self ,model ,optimizer ,scheduler ,data_loader ,device ,writer ,
    args ,scaler ):
        model .train ()
        clip =args .get ('grad_clip',50.0 )
        log_interval =args .get ('log_interval',10 )
        rank =args .get ('rank',0 )
        accum_grad =args .get ('accum_grad',1 )
        is_distributed =args .get ('is_distributed',True )
        use_amp =args .get ('use_amp',False )
        logging .info ('при использовании градиента накопления размер нового пакета равен {} раз'
        'больше, чем раньше'.format (accum_grad ))
        if use_amp :
            assert scaler is not None
        num_seen_utts =0
        num_total_batch =len (data_loader )
        data_start =time .time ()
        for batch_idx ,batch in enumerate (data_loader ):
            train_start =time .time ()
            data_dur =train_start -data_start
            key ,feats ,target ,feats_lengths ,target_lengths =batch
            feats =feats .to (device )
            target =target .to (device )
            feats_lengths =feats_lengths .to (device )
            target_lengths =target_lengths .to (device )
            num_utts =target_lengths .size (0 )
            if num_utts ==0 :
                continue
            context =None
            if is_distributed and batch_idx %accum_grad !=0 :
                context =model .no_sync
            else :
                context =nullcontext
            with context ():
                with torch .cuda .amp .autocast (scaler is not None ):
                    loss ,loss_att ,loss_ctc =model (feats ,feats_lengths ,
                    target ,target_lengths )
                    loss =loss /accum_grad
                if use_amp :
                    scaler .scale (loss ).backward ()
                else :
                    loss .backward ()
            num_seen_utts +=num_utts
            if batch_idx %accum_grad ==0 :
                if rank ==0 and writer is not None :
                    writer .add_scalar ('train_loss',loss ,self .step )
                if use_amp :
                    scaler .unscale_ (optimizer )
                    grad_norm =clip_grad_norm_ (model .parameters (),clip )
                    scaler .step (optimizer )
                    scaler .update ()
                else :
                    grad_norm =clip_grad_norm_ (model .parameters (),clip )
                    if torch .isfinite (grad_norm ):
                        optimizer .step ()
                optimizer .zero_grad ()
                scheduler .step ()
                self .step +=1
            train_dur =time .time ()-data_start
            logging .info (f"rank: {rank }, avg_reader_cost: {data_dur }, avg_batch_cost: {train_dur }, avg_samples: {num_utts }, avg_ips: {num_utts /train_dur }")
            if batch_idx %1 ==0 :
                lr =optimizer .param_groups [0 ]['lr']
                log_str ='TRAIN Пакетная потеря {}/{} {:.6f} '.format (
                batch_idx ,num_total_batch ,
                loss .item ()*accum_grad )
                if loss_att is not None :
                    log_str +='потеря_в {:.6f} '.format (loss_att .item ())
                if loss_ctc is not None :
                    log_str +='loss_ctc {:.6f} '.format (loss_ctc .item ())
                log_str +='лр {:.8f} ранг {}'.format (lr ,rank )
                logging .debug (log_str )
            data_start =time .time ()
    def cv (self ,model ,data_loader ,device ,args ):
        model .eval ()
        log_interval =args .get ('log_interval',10 )
        num_seen_utts =1
        total_loss =0.0
        num_total_batch =len (data_loader )
        with torch .no_grad ():
            for batch_idx ,batch in enumerate (data_loader ):
                key ,feats ,target ,feats_lengths ,target_lengths =batch
                feats =feats .to (device )
                target =target .to (device )
                feats_lengths =feats_lengths .to (device )
                target_lengths =target_lengths .to (device )
                num_utts =target_lengths .size (0 )
                if num_utts ==0 :
                    continue
                loss ,loss_att ,loss_ctc =model (feats ,feats_lengths ,target ,
                target_lengths )
                if torch .isfinite (loss ):
                    num_seen_utts +=num_utts
                    total_loss +=loss .item ()*num_utts
                if batch_idx %log_interval ==0 :
                    log_str ='Потеря пакета CV {}/{} {:.6f} '.format (
                    batch_idx ,num_total_batch ,loss .item ())
                    if loss_att is not None :
                        log_str +='потеря_в {:.6f} '.format (loss_att .item ())
                    if loss_ctc is not None :
                        log_str +='loss_ctc {:.6f} '.format (loss_ctc .item ())
                    log_str +='потеря истории {:.6f}'.format (total_loss /
                    num_seen_utts )
                    logging .debug (log_str )
        return total_loss ,num_seen_utts