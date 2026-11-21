import numpy as np
import argparse
import logging
import time
import sys
import os
import paddle
import paddle .fluid as fluid
import layers .paddle_layers as layers
import util .data_reader as data_reader
_WORK_DIR =os .path .split (os .path .realpath (__file__ ))[0 ]
sys .path .append (os .path .join (_WORK_DIR ,'../../../common'))
import utils
import log
def train (conf_dict ):
    data =layers .DataLayer ()
    net =utils .import_class (
    "nets",conf_dict ["net"]["module_name"],conf_dict ["net"]["class_name"])(conf_dict )
    loss =utils .import_class (
    "losses",conf_dict ["loss"]["module_name"],conf_dict ["loss"]["class_name"])(conf_dict )
    optimizer =utils .import_class (
    "optimizers","paddle_optimizers",conf_dict ["optimizer"]["class_name"])(conf_dict )
    if "use_cuda"in conf_dict and conf_dict ["use_cuda"]==1 :
        place =fluid .core .CUDAPlace (0 )
    else :
        place =fluid .core .CPUPlace ()
    if conf_dict ["task_mode"]=="pairwise":
        left =data .ops (name ="left",shape =[1 ],dtype ="int64",lod_level =1 )
        pos_right =data .ops (name ="right",shape =[
        1 ],dtype ="int64",lod_level =1 )
        neg_right =data .ops (name ="neg_right",shape =[
        1 ],dtype ="int64",lod_level =1 )
        left_feat ,pos_score =net .predict (left ,pos_right )
        _ ,neg_score =net .predict (left ,neg_right )
        avg_cost =loss .compute (pos_score ,neg_score )
        feeder =fluid .DataFeeder (place =place ,feed_list =[
        left .name ,pos_right .name ,neg_right .name ])
        reader =data_reader .get_reader (conf_dict ,False ,None )
    else :
        left =data .ops (name ="left",shape =[1 ],dtype ="int64",lod_level =1 )
        right =data .ops (name ="right",shape =[1 ],dtype ="int64",lod_level =1 )
        label =data .ops (name ="label",shape =[1 ],dtype ="int64",lod_level =0 )
        left_feat ,pred =net .predict (left ,right )
        avg_cost =loss .compute (pred ,label )
        avg_cost .persistable =True
        feeder =fluid .DataFeeder (place =place ,feed_list =[
        left .name ,right .name ,label .name ])
        reader =data_reader .get_reader (conf_dict ,False ,None )
    infer_program =fluid .default_main_program ().clone ()
    optimizer .ops (avg_cost )
    executor =fluid .Executor (place )
    executor .run (fluid .default_startup_program ())
    parallel_executor =fluid .ParallelExecutor (
    use_cuda ="use_cuda"in conf_dict and conf_dict ["use_cuda"]==1 ,
    loss_name =avg_cost .name ,
    main_program =fluid .default_main_program ())
    device_count =parallel_executor .device_count
    logging .info ("количество устройств: %d"%device_count )
    logging .info ("начать процесс поезда...")
    for epoch_id in range (conf_dict ["epoch_num"]):
        losses =[]
        batch_data =paddle .batch (reader ,conf_dict ["batch_size"],drop_last =False )
        start_time =time .time ()
        total_loss =0.0
        for iter ,data in enumerate (batch_data ()):
            if len (data )<device_count :
                continue
            avg_loss =parallel_executor .run (
            [avg_cost .name ],feed =feeder .feed (data ))
            total_loss +=np .mean (avg_loss [0 ])
            if (iter +1 )%100 ==0 :
                print ("эпоха: %d, итер: %d, потеря: %f"%
                (epoch_id ,iter ,total_loss /100 ))
                total_loss =0.0
            losses .append (np .mean (avg_loss [0 ]))
        end_time =time .time ()
        print ("эпоха: %d, потеря: %f, время использования: %f"%
        (epoch_id ,np .mean (losses ),end_time -start_time ))
        model_save_dir =conf_dict ["model_path"]
        model_path =os .path .join (model_save_dir ,str (epoch_id ))
        if not os .path .exists (model_save_dir ):
            os .makedirs (model_save_dir )
        if conf_dict ["task_mode"]=="pairwise":
            feed_var_names =[left .name ,pos_right .name ]
            target_vars =[left_feat ,pos_score ]
        else :
            feed_var_names =[left .name ,right .name ]
            target_vars =[left_feat ,pred ]
        fluid .io .save_inference_model (
        model_path ,feed_var_names ,target_vars ,executor ,infer_program )
def predict (conf_dict ):
    with open ("samples.txt","w")as samples_file :
        with open ("predictions.txt","w")as predictions_file :
            model_save_dir =conf_dict ["model_path"]
            model_path =os .path .join (model_save_dir ,str (conf_dict ["use_epoch"]))
            if "use_cuda"in conf_dict and conf_dict ["use_cuda"]==1 :
                place =fluid .core .CUDAPlace (0 )
            else :
                place =fluid .core .CPUPlace ()
            executor =fluid .Executor (place =place )
            program ,feed_var_names ,fetch_targets =fluid .io .load_inference_model (
            model_path ,executor )
            if conf_dict ["task_mode"]=="pairwise":
                feeder =fluid .DataFeeder (
                place =place ,feed_list =feed_var_names ,program =program )
                reader =data_reader .get_reader (conf_dict ,True ,samples_file )
            else :
                feeder =fluid .DataFeeder (
                place =place ,feed_list =feed_var_names ,program =program )
                reader =data_reader .get_reader (conf_dict ,True ,samples_file )
            batch_data =paddle .batch (reader ,conf_dict ["batch_size"],drop_last =False )
            logging .info ("начать процесс тестирования...")
            bt =time .time ()
            for iter ,data in enumerate (batch_data ()):
                output =executor .run (program ,feed =feeder .feed (
                data ),fetch_list =fetch_targets )
                if conf_dict ["task_mode"]=="pairwise":
                    predictions_file .write (
                    "\n".join (map (lambda item :str (item [0 ]),output [1 ]))+"\n")
                else :
                    predictions_file .write (
                    "\n".join (map (lambda item :str (np .argmax (item )),output [1 ]))+"\n")
            if iter :
                print ("предсказать скорость: {} с/примеров".format ((time .time ()-bt )/iter ))
            else :
                print ("пожалуйста, проверьте данные, возможно, они нулевые!")
    utils .get_result_file (conf_dict ,"samples.txt","predictions.txt")
if __name__ =="__main__":
    log .init_log ("./log/paddle")
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--task_type",default ="train",
    help ="тип задачи включает обучение/предсказание, значение по умолчанию — поезд")
    parser .add_argument (
    "--conf_file_path",default ="examples/cnn_pointwise.json",help ="путь к файлу конфигурации")
    args =parser .parse_args ()
    conf_dict =utils .parse_json (args .conf_file_path )
    print (conf_dict )
    if args .task_type =="train":
        train (conf_dict )
    else :
        predict (conf_dict )