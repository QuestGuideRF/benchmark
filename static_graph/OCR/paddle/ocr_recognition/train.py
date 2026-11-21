from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle .fluid as fluid
from utility import add_arguments ,print_arguments ,get_ctc_feeder_data ,get_attention_feeder_data
import paddle .fluid .profiler as profiler
from static_graph .OCR .paddle .ocr_recognition .crnn_ctc_model import ctc_train_net
from static_graph .OCR .paddle .ocr_recognition .attention_model import attention_train_net
import data_reader
import argparse
import functools
import time
import os
import numpy as np
parser =argparse .ArgumentParser (description =__doc__ )
add_arg =functools .partial (add_arguments ,argparser =parser )
add_arg ('batch_size',int ,32 ,"Размер мини-партии.")
add_arg ('total_step',int ,720000 ,"Количество итераций. Ноль или меньше означает весь обучающий набор. Значение больше 0 означает, что обучающий набор может быть зациклен до тех пор, пока не будет достигнуто количество итераций.")
add_arg ('log_period',int ,1000 ,"Период журнала.")
add_arg ('save_model_period',int ,15000 ,"Сохраните модельный период. '-1' означает никогда не сохранять модель.")
add_arg ('eval_period',int ,15000 ,"Оцените период. '-1' означает, что модель никогда не будет оцениваться.")
add_arg ('save_model_dir',str ,"./models","Каталог, в котором будет сохранена модель.")
add_arg ('train_images',str ,None ,"Каталог изображений, которые будут использоваться для обучения.")
add_arg ('train_list',str ,None ,"Файл списка изображений, которые будут использоваться для обучения.")
add_arg ('test_images',str ,None ,"Каталог изображений, которые будут использоваться для теста.")
add_arg ('test_list',str ,None ,"Файл списка изображений, которые будут использоваться для обучения.")
add_arg ('model',str ,"crnn_ctc","Какой тип сети использовать. «crnn_ctc» или «внимание»")
add_arg ('init_model',str ,None ,"Файл модели инициализации каталога.")
add_arg ('use_gpu',bool ,True ,"Используете ли графический процессор для обучения.")
add_arg ('min_average_window',int ,10000 ,"Минимальное среднее окно.")
add_arg ('max_average_window',int ,12500 ,"Максимальное среднее окно. Предлагается задавать количество минипакетов за проход.")
add_arg ('average_window',float ,0.15 ,"Среднее окно.")
add_arg ('parallel',bool ,False ,"Используете ли параллельное обучение.")
add_arg ('profile',bool ,False ,"Использовать ли профилирование.")
add_arg ('skip_batch_num',int ,0 ,"Количество первых мини-партий, которые следует пропустить в качестве разминки для лучшего тестирования производительности.")
add_arg ('skip_test',bool ,False ,"Пропускать ли фазу тестирования.")
def train (args ):
    if args .model =="crnn_ctc":
        train_net =ctc_train_net
        get_feeder_data =get_ctc_feeder_data
    else :
        train_net =attention_train_net
        get_feeder_data =get_attention_feeder_data
    num_classes =None
    num_classes =data_reader .num_classes (
    )if num_classes is None else num_classes
    data_shape =data_reader .data_shape ()
    sum_cost ,error_evaluator ,inference_program ,model_average =train_net (
    args ,data_shape ,num_classes )
    train_reader =data_reader .train (
    args .batch_size ,
    train_images_dir =args .train_images ,
    train_list_file =args .train_list ,
    cycle =args .total_step >0 ,
    model =args .model )
    test_reader =data_reader .test (
    test_images_dir =args .test_images ,test_list_file =args .test_list ,model =args .model )
    place =fluid .CPUPlace ()
    if args .use_gpu :
        place =fluid .CUDAPlace (0 )
    exe =fluid .Executor (place )
    if 'ce_mode'in os .environ :
        fluid .default_startup_program ().random_seed =90
    exe .run (fluid .default_startup_program ())
    if args .init_model is not None :
        model_dir =args .init_model
        model_file_name =None
        if not os .path .isdir (args .init_model ):
            model_dir =os .path .dirname (args .init_model )
            model_file_name =os .path .basename (args .init_model )
        fluid .io .load_params (exe ,dirname =model_dir ,filename =model_file_name )
        print ("Инициализировать модель из: %s."%args .init_model )
    train_exe =exe
    error_evaluator .reset (exe )
    if args .parallel :
        train_exe =fluid .ParallelExecutor (
        use_cuda =True if args .use_gpu else False ,loss_name =sum_cost .name )
    fetch_vars =[sum_cost ]+error_evaluator .metrics
    def train_one_batch (data ):
        var_names =[var .name for var in fetch_vars ]
        if args .parallel :
            results =train_exe .run (var_names ,
            feed =get_feeder_data (data ,place ))
            results =[np .array (result ).sum ()for result in results ]
        else :
            results =train_exe .run (feed =get_feeder_data (data ,place ),
            fetch_list =fetch_vars )
            results =[result [0 ]for result in results ]
        return results
    def test (iter_num ):
        error_evaluator .reset (exe )
        for data in test_reader ():
            exe .run (inference_program ,feed =get_feeder_data (data ,place ))
        _ ,test_seq_error =error_evaluator .eval (exe )
        print ("\nВремя: %s; Итер[%d]; Ошибка последовательности теста: %s.\n"%(
        time .time (),iter_num ,str (test_seq_error [0 ])))
        print ("kpis test_acc %f"%(1 -test_seq_error [0 ]))
    def save_model (args ,exe ,iter_num ):
        filename ="параметры_%05d"%iter_num
        fluid .io .save_params (
        exe ,dirname =args .save_model_dir ,filename =filename )
        print ("Модель сохранена в: %s/%s."%(args .save_model_dir ,filename ))
    iter_num =0
    stop =False
    start_time =time .time ()
    while not stop :
        total_loss =0.0
        total_seq_error =0.0
        batch_times =[]
        for data in train_reader ():
            if args .total_step >0 and iter_num ==args .total_step +args .skip_batch_num :
                stop =True
                break
            if iter_num <args .skip_batch_num :
                print ("Итерация разминки")
            if iter_num ==args .skip_batch_num :
                profiler .reset_profiler ()
            start =time .time ()
            results =train_one_batch (data )
            batch_time =time .time ()-start
            fps =args .batch_size /batch_time
            batch_times .append (batch_time )
            total_loss +=results [0 ]
            total_seq_error +=results [2 ]
            iter_num +=1
            if iter_num %args .log_period ==0 :
                print ("\nВремя: %s; Итер[%d]; Средняя потеря: %.3f; Ошибка средней последовательности: %.3f"%(
                time .time (),iter_num ,
                total_loss /(args .log_period *args .batch_size ),
                total_seq_error /(args .log_period *args .batch_size )))
                print ("kpis train_cost %f"%(total_loss /(args .log_period *
                args .batch_size )))
                print ("kpis train_acc %f"%(
                1 -total_seq_error /(args .log_period *args .batch_size )))
                total_loss =0.0
                total_seq_error =0.0
            if not args .skip_test and iter_num %args .eval_period ==0 :
                if model_average :
                    with model_average .apply (exe ):
                        test (iter_num )
                else :
                    test (iter_num )
            if iter_num %args .save_model_period ==0 :
                if model_average :
                    with model_average .apply (exe ):
                        save_model (args ,exe ,iter_num )
                else :
                    save_model (args ,exe ,iter_num )
        end_time =time .time ()
        print ("kpis train_duration %f"%(end_time -start_time ))
        latencies =batch_times [args .skip_batch_num :]
        latency_avg =np .average (latencies )
        latency_pc99 =np .percentile (latencies ,99 )
        fpses =np .divide (args .batch_size ,latencies )
        fps_avg =np .average (fpses )
        fps_pc99 =np .percentile (fpses ,1 )
        print ('\nВсего примеров (включая разминку): %d'%
        (iter_num *args .batch_size ))
        print ('средняя задержка: %.5f сек., задержка 99 ПК: %.5f сек.'%(latency_avg ,
        latency_pc99 ))
        print ('средний кадр/с: %.5f, задержка кадров/с для 99 ПК: %.5f'%(fps_avg ,
        fps_pc99 ))
def main ():
    args =parser .parse_args ()
    print_arguments (args )
    if args .profile :
        if args .use_gpu :
            with profiler .cuda_profiler ("cuda_profiler.txt",'csv')as nvprof :
                train (args )
        else :
            with profiler .profiler ("CPU",sorted_key ='total')as cpuprof :
                train (args )
    else :
        train (args )
if __name__ =="__main__":
    main ()