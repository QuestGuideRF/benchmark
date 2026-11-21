from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import distutils .util
import numpy as np
from paddle .fluid import core
import paddle .fluid as fluid
import six
def print_arguments (args ):
    print ("----------- Аргументы конфигурации -----------")
    for arg ,value in sorted (six .iteritems (vars (args ))):
        print ("%s: %s"%(arg ,value ))
    print ("------------------------------------------------")
def add_arguments (argname ,type ,default ,help ,argparser ,**kwargs ):
    type =distutils .util .strtobool if type ==bool else type
    argparser .add_argument (
    "--"+argname ,
    default =default ,
    type =type ,
    help =help +' По умолчанию: %(по умолчанию)с.',
    **kwargs )
def to_lodtensor (data ,place ):
    seq_lens =[len (seq )for seq in data ]
    cur_len =0
    lod =[cur_len ]
    for l in seq_lens :
        cur_len +=l
        lod .append (cur_len )
    flattened_data =np .concatenate (data ,axis =0 ).astype ("int32")
    flattened_data =flattened_data .reshape ([len (flattened_data ),1 ])
    res =core .DenseTensor ()
    res .set (flattened_data ,place )
    res .set_lod ([lod ])
    return res
def get_ctc_feeder_data (data ,place ,need_label =True ):
    pixel_tensor =core .DenseTensor ()
    pixel_data =None
    pixel_data =np .concatenate (
    list (map (lambda x :x [0 ][np .newaxis ,:],data )),axis =0 ).astype ("float32")
    pixel_tensor .set (pixel_data ,place )
    label_tensor =to_lodtensor (list (map (lambda x :x [1 ],data )),place )
    if need_label :
        return {"pixel":pixel_tensor ,"label":label_tensor }
    else :
        return {"pixel":pixel_tensor }
def get_attention_feeder_data (data ,place ,need_label =True ):
    pixel_tensor =core .DenseTensor ()
    pixel_data =None
    pixel_data =np .concatenate (
    list (map (lambda x :x [0 ][np .newaxis ,:],data )),axis =0 ).astype ("float32")
    pixel_tensor .set (pixel_data ,place )
    label_in_tensor =to_lodtensor (list (map (lambda x :x [1 ],data )),place )
    label_out_tensor =to_lodtensor (list (map (lambda x :x [2 ],data )),place )
    if need_label :
        return {
        "pixel":pixel_tensor ,
        "label_in":label_in_tensor ,
        "label_out":label_out_tensor
        }
    else :
        return {"pixel":pixel_tensor }
def get_attention_feeder_for_infer (data ,place ,need_label =False ):
    batch_size =len (data )
    init_ids_data =np .array ([0 for _ in range (batch_size )],dtype ='int64')
    init_scores_data =np .array (
    [1. for _ in range (batch_size )],dtype ='float32')
    init_ids_data =init_ids_data .reshape ((batch_size ,1 ))
    init_scores_data =init_scores_data .reshape ((batch_size ,1 ))
    init_recursive_seq_lens =[1 ]*batch_size
    init_recursive_seq_lens =[init_recursive_seq_lens ,init_recursive_seq_lens ]
    init_ids =fluid .create_lod_tensor (init_ids_data ,init_recursive_seq_lens ,
    place )
    init_scores =fluid .create_lod_tensor (init_scores_data ,
    init_recursive_seq_lens ,place )
    pixel_tensor =core .DenseTensor ()
    pixel_data =None
    pixel_data =np .concatenate (
    list (map (lambda x :x [0 ][np .newaxis ,:],data )),axis =0 ).astype ("float32")
    pixel_tensor .set (pixel_data ,place )
    return {
    "pixel":pixel_tensor ,
    "init_ids":init_ids ,
    "init_scores":init_scores
    }