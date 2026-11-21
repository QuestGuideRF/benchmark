from __future__ import print_function
from collections import Counter
import logging
import numpy
import time
import sys
import os
import tensorflow as tf
def int_feature (v ):
    return tf .train .Feature (int64_list =tf .train .Int64List (value =v ))
def write_data_to_tf (filename ,func ,func_args ,writer ):
    with open (filename )as fin_data :
        for line in fin_data :
            example =func (line ,func_args )
            if not example :
                continue
            writer .write (example .SerializeToString ())
def parse_text_match_pointwise_pad_data (line ,func_args ):
    seq_len =func_args [0 ]
    pad_id =func_args [1 ]
    group =line .strip ().split ("\t")
    if len (group )!=3 :
        logging .warning (
        "строка не соответствует формату (left_ids, right_ids, label)")
        return
    label =[0 ,0 ]
    all_ids =[]
    for i in [0 ,1 ]:
        tmp_ids =[int (t )for t in group [i ].strip ().split (" ")]
        if len (tmp_ids )<seq_len :
            pad_len =seq_len -len (tmp_ids )
            tmp_ids =tmp_ids +[pad_id ]*pad_len
        all_ids .append (tmp_ids [:seq_len ])
    label [int (group [2 ])]=1
    example =tf .train .Example (features =tf .train .Features (
    feature ={"label":int_feature (label ),
    "left":int_feature (all_ids [0 ]),
    "right":int_feature (all_ids [1 ])}))
    return example
def parse_text_match_pairwise_pad_data (line ,func_args ):
    seq_len =func_args [0 ]
    pad_id =func_args [1 ]
    group =line .strip ().split ("\t")
    if len (group )!=3 :
        logging .warning (
        "строка не соответствует формату (query_terms, posttitle_terms, negtitle_terms)")
        return
    all_ids =[]
    for i in [0 ,1 ,2 ]:
        tmp_ids =[int (t )for t in group [i ].strip ().split (" ")]
        if len (tmp_ids )<seq_len :
            pad_len =seq_len -len (tmp_ids )
            tmp_ids =tmp_ids +[pad_id ]*pad_len
        all_ids .append (tmp_ids [:seq_len ])
    example =tf .train .Example (features =tf .train .Features (
    feature ={"left":int_feature (all_ids [0 ]),
    "pos_right":int_feature (all_ids [1 ]),
    "neg_right":int_feature (all_ids [2 ])}))
    return example
def usage ():
    print (sys .argv [0 ],"options")
    print ("options")
    print ("\ttype: тип данных включает поточечное или попарное")
    print ("\tinputfile: путь к входному файлу")
    print ("\trecordfile: выходной файл записи")
    print ("\tpad_id: идентификатор площадки")
    print ("\tmax_len: максимальная длина последовательности")
if __name__ =="__main__":
    if len (sys .argv )!=6 :
        usage ()
        sys .exit (-1 )
    input_data_format =sys .argv [1 ]
    filename =sys .argv [2 ]
    tfrecord_name =sys .argv [3 ]
    pad_id =int (sys .argv [4 ])
    max_len =int (sys .argv [5 ])
    data_format_func ={"pointwise":parse_text_match_pointwise_pad_data ,
    "pairwise":parse_text_match_pairwise_pad_data }
    if input_data_format in data_format_func :
        using_func =data_format_func [input_data_format ]
    else :
        logging .error ("формат_данных не поддерживается")
        sys .exit (1 )
    local_writer =tf .python_io .TFRecordWriter (tfrecord_name )
    write_data_to_tf (filename ,using_func ,[max_len ,pad_id ],local_writer )
    local_writer .close ()