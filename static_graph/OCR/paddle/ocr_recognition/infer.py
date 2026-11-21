from __future__ import print_function
import paddle .fluid as fluid
from utility import add_arguments ,print_arguments ,get_ctc_feeder_data ,get_attention_feeder_for_infer
import paddle .fluid .profiler as profiler
from crnn_ctc_model import ctc_infer
from attention_model import attention_infer
import numpy as np
import data_reader
import argparse
import functools
import os
import time
parser =argparse .ArgumentParser (description =__doc__ )
add_arg =functools .partial (add_arguments ,argparser =parser )
add_arg ('model',str ,"crnn_ctc","Какой тип сети использовать. «crnn_ctc» или «внимание»")
add_arg ('model_path',str ,None ,"Путь модели, который будет использоваться для вывода.")
add_arg ('input_images_dir',str ,None ,"Каталог изображений.")
add_arg ('input_images_list',str ,None ,"Файл списка изображений.")
add_arg ('dict',str ,None ,"Словарь. Результатом вывода будет индексная последовательность, если словарь был None.")
add_arg ('use_gpu',bool ,True ,"Используйте ли графический процессор для вывода.")
add_arg ('iterations',int ,0 ,"Количество итераций. Ноль или меньше означает весь набор тестов. Значение больше 0 означает, что набор тестов может быть зациклен до тех пор, пока не будет достигнуто количество итераций.")
add_arg ('profile',bool ,False ,"Использовать ли профилирование.")
add_arg ('skip_batch_num',int ,0 ,"Количество первых мини-партий, которые следует пропустить в качестве разминки для лучшего тестирования производительности.")
add_arg ('batch_size',int ,1 ,"Размер мини-пакета.")
def inference (args ):
    if args .model =="crnn_ctc":
        infer =ctc_infer
        get_feeder_data =get_ctc_feeder_data
    else :
        infer =attention_infer
        get_feeder_data =get_attention_feeder_for_infer
    eos =1
    sos =0
    num_classes =data_reader .num_classes ()
    data_shape =data_reader .data_shape ()
    images =fluid .layers .data (name ='pixel',shape =data_shape ,dtype ='float32')
    ids =infer (images ,num_classes ,use_cudnn =True if args .use_gpu else False )
    infer_reader =data_reader .inference (
    batch_size =args .batch_size ,
    infer_images_dir =args .input_images_dir ,
    infer_list_file =args .input_images_list ,
    cycle =True if args .iterations >0 else False ,
    model =args .model )
    place =fluid .CPUPlace ()
    if args .use_gpu :
        place =fluid .CUDAPlace (0 )
    exe =fluid .Executor (place )
    exe .run (fluid .default_startup_program ())
    dict_map =None
    if args .dict is not None and os .path .isfile (args .dict ):
        dict_map ={}
        with open (args .dict )as dict_file :
            for i ,word in enumerate (dict_file ):
                dict_map [i ]=word .strip ()
        print ("Загружен словарь из %s"%args .dict )
    model_dir =args .model_path
    model_file_name =None
    if not os .path .isdir (args .model_path ):
        model_dir =os .path .dirname (args .model_path )
        model_file_name =os .path .basename (args .model_path )
    fluid .io .load_params (exe ,dirname =model_dir ,filename =model_file_name )
    print ("Инициализировать модель из: %s."%args .model_path )
    batch_times =[]
    iters =0
    for data in infer_reader ():
        feed_dict =get_feeder_data (data ,place ,need_label =False )
        if args .iterations >0 and iters ==args .iterations +args .skip_batch_num :
            break
        if iters <args .skip_batch_num :
            print ("Итерация разминки")
        if iters ==args .skip_batch_num :
            profiler .reset_profiler ()
        start =time .time ()
        result =exe .run (fluid .default_main_program (),
        feed =feed_dict ,
        fetch_list =[ids ],
        return_numpy =False ,
        use_program_cache =True )
        indexes =prune (np .array (result [0 ]).flatten (),0 ,1 )
        batch_time =time .time ()-start
        fps =args .batch_size /batch_time
        batch_times .append (batch_time )
        if dict_map is not None :
            print ("Итерация %d, задержка: %.5f с, кадр/с: %f, результат: %s"%(
            iters ,
            batch_time ,
            fps ,
            [dict_map [index ]for index in indexes ],))
        else :
            print ("Итерация %d, задержка: %.5f с, кадр/с: %f, результат: %s"%(
            iters ,
            batch_time ,
            fps ,
            indexes ,))
        iters +=1
    latencies =batch_times [args .skip_batch_num :]
    latency_avg =np .average (latencies )
    latency_pc99 =np .percentile (latencies ,99 )
    fpses =np .divide (args .batch_size ,latencies )
    fps_avg =np .average (fpses )
    fps_pc99 =np .percentile (fpses ,1 )
    print ('\nВсего примеров (включая разминку): %d'%(iters *args .batch_size ))
    print ('средняя задержка: %.5f сек., задержка 99 ПК: %.5f сек.'%(latency_avg ,
    latency_pc99 ))
    print ('средний кадр/с: %.5f, задержка кадров/с для 99 ПК: %.5f'%(fps_avg ,fps_pc99 ))
def prune (words ,sos ,eos ):
    start_index =0
    end_index =len (words )
    if sos in words :
        start_index =np .where (words ==sos )[0 ][0 ]+1
    if eos in words :
        end_index =np .where (words ==eos )[0 ][0 ]
    return words [start_index :end_index ]
def main ():
    args =parser .parse_args ()
    print_arguments (args )
    if args .profile :
        if args .use_gpu :
            with profiler .cuda_profiler ("cuda_profiler.txt",'csv')as nvprof :
                inference (args )
        else :
            with profiler .profiler ("CPU",sorted_key ='total')as cpuprof :
                inference (args )
    else :
        inference (args )
if __name__ =="__main__":
    main ()