import logging
import json
import time
import sys
import os
import tensorflow as tf
_WORK_DIR =os .path .split (os .path .realpath (__file__ ))[0 ]
_UPPER_DIR =os .path .split (_WORK_DIR )[0 ]
sys .path .append (_UPPER_DIR )
from static_graph .MMDNN .paddle .utils import datafeeds
def load_config (config_file ):
    with open (config_file ,"r")as f :
        try :
            conf =json .load (f )
        except Exception :
            logging .error ("ошибка загрузки json-файла %s"%config_file )
    conf_dict ={}
    unused =[conf_dict .update (conf [k ])for k in conf ]
    logging .debug ("\n".join (["%s=%s"%(u ,conf_dict [u ])for u in conf_dict ]))
    return conf_dict
def read_tfrecords_pointwise (config ):
    datafeed =datafeeds .TFPointwisePaddingData (config )
    input_l ,input_r ,label_y =datafeed .ops ()
    init_op =tf .group (tf .global_variables_initializer (),
    tf .local_variables_initializer ())
    start_time =time .time ()
    sess =tf .InteractiveSession ()
    sess .run (init_op )
    coord =tf .train .Coordinator ()
    threads =tf .train .start_queue_runners (sess =sess ,coord =coord )
    step =0
    while not coord .should_stop ():
        step +=1
        try :
            left_ ,right_ ,label_ =sess .run ([input_l ,input_r ,label_y ])
            print ("точечное чтение данных хорошее")
        except tf .errors .OutOfRangeError :
            print ("прочитать %d шагов"%step )
            coord .request_stop ()
    coord .join (threads )
    duration =time .time ()-start_time
    print ("продолжительность: %dс, шаг: %d"%(duration ,step ))
    sess .close ()
def read_tfrecords_pairwise (config ):
    datafeed =datafeeds .TFPairwisePaddingData (config )
    query ,pos ,neg =datafeed .ops ()
    init_op =tf .group (tf .global_variables_initializer (),
    tf .local_variables_initializer ())
    start_time =time .time ()
    sess =tf .InteractiveSession ()
    sess .run (init_op )
    coord =tf .train .Coordinator ()
    threads =tf .train .start_queue_runners (sess =sess ,coord =coord )
    step =0
    while not coord .should_stop ():
        step +=1
        try :
            query_ ,pos_ ,neg_ =sess .run ([query ,pos ,neg ])
            print ("попарное чтение данных — это хорошо")
        except tf .errors .OutOfRangeError :
            print ("прочитать %d шагов"%step )
            coord .request_stop ()
    coord .join (threads )
    duration =time .time ()-start_time
    print ("продолжительность: %dс, шаг: %d"%(duration ,step ))
    sess .close ()
def usage ():
    print (sys .argv [0 ],"options")
    print ("options")
    print ("\tconfig_path: настроить путь к файлу")
if __name__ =="__main__":
    if len (sys .argv )!=2 :
        usage ()
        sys .exit (1 )
    config_path =sys .argv [1 ]
    config =load_config (config_path )
    data_format_func ={"pointwise":read_tfrecords_pointwise ,
    "pairwise":read_tfrecords_pairwise }
    if config ["training_mode"]in data_format_func :
        using_func =data_format_func [config ["training_mode"]]
    else :
        logging .error ("формат_данных не поддерживается")
        sys .exit (1 )
    using_func (config )