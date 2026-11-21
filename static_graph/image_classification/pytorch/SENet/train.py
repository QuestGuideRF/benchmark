from __future__ import print_function ,division
import torch
import torch .nn as nn
import torch .optim as optim
from torch .optim import lr_scheduler
from torch .autograd import Variable
import time
import os
import argparse
from read_ImageNetData import ImageNetData
import se_resnet
import se_resnext
from torchvision import models
def train_model (args ,model ,criterion ,optimizer ,scheduler ,num_epochs ,dataset_sizes ):
    since =time .time ()
    resumed =False
    best_model_wts =model .state_dict ()
    for epoch in range (args .start_epoch +1 ,num_epochs ):
        for phase in ['train','val']:
            if phase =='train':
                if args .start_epoch >0 and (not resumed ):
                    scheduler .step (args .start_epoch +1 )
                    resumed =True
                else :
                    scheduler .step (epoch )
                model .train (True )
            else :
                model .train (False )
            running_loss =0.0
            running_corrects =0
            tic_batch =time .time ()
            for i ,(inputs ,labels )in enumerate (dataloders [phase ]):
                if use_gpu :
                    inputs =Variable (inputs .cuda ())
                    labels =Variable (labels .cuda ())
                else :
                    inputs ,labels =Variable (inputs ),Variable (labels )
                optimizer .zero_grad ()
                outputs =model (inputs )
                _ ,preds =torch .max (outputs .data ,1 )
                loss =criterion (outputs ,labels )
                if phase =='train':
                    loss .backward ()
                    optimizer .step ()
                running_loss +=loss .item ()
                running_corrects +=torch .sum (preds ==labels .data )
                batch_loss =running_loss /((i +1 )*args .batch_size )
                batch_acc =running_corrects /((i +1 )*args .batch_size )
                if phase =='train'and i %args .print_freq ==0 :
                    print ('[Эпоха {}/{}]-[пакет:{}/{}] lr:{:.4f} {} Потеря: {:.6f} Акк: {:.4f} Время: {:.4f}пакет/сек'.format (
                    epoch ,num_epochs -1 ,i ,round (dataset_sizes [phase ]/args .batch_size )-1 ,scheduler .get_lr ()[0 ],phase ,batch_loss ,batch_acc ,args .print_freq /(time .time ()-tic_batch )),flush =True )
                    tic_batch =time .time ()
            epoch_loss =running_loss /dataset_sizes [phase ]
            epoch_acc =running_corrects /dataset_sizes [phase ]
            print ('{} Потеря: {:.4f} Акк: {:.4f}'.format (
            phase ,epoch_loss ,epoch_acc ))
        if (epoch +1 )%args .save_epoch_freq ==0 :
            if not os .path .exists (args .save_path ):
                os .makedirs (args .save_path )
            torch .save (model ,os .path .join (args .save_path ,"epoch_"+str (epoch )+".pth.tar"))
    time_elapsed =time .time ()-since
    print ('Обучение завершено через {:.0f} мин. {:.0f} с.'.format (
    time_elapsed //60 ,time_elapsed %60 ))
    model .load_state_dict (best_model_wts )
    return model
if __name__ =='__main__':
    parser =argparse .ArgumentParser (description ="PyTorch реализация SENet")
    parser .add_argument ('--data-dir',type =str ,default ="./imagenet/")
    parser .add_argument ('--batch-size',type =int ,default =16 )
    parser .add_argument ('--num-class',type =int ,default =1000 )
    parser .add_argument ('--num-epochs',type =int ,default =100 )
    parser .add_argument ('--lr',type =float ,default =0.045 )
    parser .add_argument ('--num-workers',type =int ,default =0 )
    parser .add_argument ('--gpus',type =str ,default =0 )
    parser .add_argument ('--print-freq',type =int ,default =10 )
    parser .add_argument ('--save-epoch-freq',type =int ,default =1 )
    parser .add_argument ('--save-path',type =str ,default ="output")
    parser .add_argument ('--resume',type =str ,default ="",help ="Для обучения с одной КПП")
    parser .add_argument ('--start-epoch',type =int ,default =0 ,help ="Соответствующее эпохе резюме ")
    parser .add_argument ('--network',type =str ,default ="se_resnext_50",help ="")
    args =parser .parse_args ()
    dataloders ,dataset_sizes =ImageNetData (args )
    use_gpu =torch .cuda .is_available ()
    print ("use_gpu:{}".format (use_gpu ))
    script_name ='_'.join ([args .network .strip ().split ('_')[0 ],args .network .strip ().split ('_')[1 ]])
    if script_name =="se_resnet":
        print ("модель: se_resnet.{}".format (args .network ))
        model =getattr (se_resnet ,args .network )(num_classes =args .num_class )
    elif script_name =="se_resnext":
        print ("модель: se_resnext.{}".format (args .network ))
        model =getattr (se_resnext ,args .network )(num_classes =args .num_class )
    elif args .network .startswith ("resnet"):
        network =args .network .replace ("_","");
        print ("модель: models.resnet.{}".format (network ))
        model =getattr (models .resnet ,network )(num_classes =args .num_class )
    else :
        raise Exception ("Укажите правильное имя сети, например se_resnet_xx или se_rexnext_xx.")
    print (model )
    if args .resume :
        if os .path .isfile (args .resume ):
            print (("=> загрузка контрольной точки '{}'".format (args .resume )))
            checkpoint =torch .load (args .resume )
            base_dict ={'.'.join (k .split ('.')[1 :]):v for k ,v in list (checkpoint .state_dict ().items ())}
            model .load_state_dict (base_dict )
        else :
            print (("=> контрольная точка в '{}' не найдена".format (args .resume )))
    if use_gpu :
        model =model .cuda ()
        model =torch .nn .DataParallel (model ,device_ids =[int (i )for i in args .gpus .strip ().split (',')])
    criterion =nn .CrossEntropyLoss ()
    optimizer_ft =optim .SGD (model .parameters (),lr =args .lr ,momentum =0.9 ,weight_decay =0.00004 )
    exp_lr_scheduler =lr_scheduler .StepLR (optimizer_ft ,step_size =1 ,gamma =0.98 )
    model =train_model (args =args ,
    model =model ,
    criterion =criterion ,
    optimizer =optimizer_ft ,
    scheduler =exp_lr_scheduler ,
    num_epochs =args .num_epochs ,
    dataset_sizes =dataset_sizes )