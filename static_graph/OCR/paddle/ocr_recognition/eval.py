import paddle .fluid as fluid
from utility import add_arguments ,print_arguments ,to_lodtensor ,get_ctc_feeder_data ,get_attention_feeder_data
from attention_model import attention_eval
from crnn_ctc_model import ctc_eval
import data_reader
import argparse
import functools
import os
parser =argparse .ArgumentParser (description =__doc__ )
add_arg =functools .partial (add_arguments ,argparser =parser )
add_arg ('model',str ,"crnn_ctc","Какой тип сети использовать. «crnn_ctc» или «внимание»")
add_arg ('model_path',str ,"","Путь модели, который будет использоваться для вывода.")
add_arg ('input_images_dir',str ,None ,"Каталог изображений.")
add_arg ('input_images_list',str ,None ,"Файл списка изображений.")
add_arg ('use_gpu',bool ,True ,"Использовать ли графический процессор для оценки.")
def evaluate (args ):
    if args .model =="crnn_ctc":
        eval =ctc_eval
        get_feeder_data =get_ctc_feeder_data
    else :
        eval =attention_eval
        get_feeder_data =get_attention_feeder_data
    num_classes =data_reader .num_classes ()
    data_shape =data_reader .data_shape ()
    evaluator ,cost =eval (data_shape ,num_classes )
    test_reader =data_reader .test (
    test_images_dir =args .input_images_dir ,
    test_list_file =args .input_images_list ,
    model =args .model )
    place =fluid .CPUPlace ()
    if args .use_gpu :
        place =fluid .CUDAPlace (0 )
    exe =fluid .Executor (place )
    exe .run (fluid .default_startup_program ())
    model_dir =args .model_path
    model_file_name =None
    if not os .path .isdir (args .model_path ):
        model_dir =os .path .dirname (args .model_path )
        model_file_name =os .path .basename (args .model_path )
    fluid .io .load_params (exe ,dirname =model_dir ,filename =model_file_name )
    print ("Инициализировать модель из: %s."%args .model_path )
    evaluator .reset (exe )
    count =0
    for data in test_reader ():
        count +=1
        exe .run (fluid .default_main_program (),feed =get_feeder_data (data ,place ))
    avg_distance ,avg_seq_error =evaluator .eval (exe )
    print ("Прочитайте образцы %d; avg_distance: %s; avg_seq_error: %s"%(
    count ,avg_distance ,avg_seq_error ))
def main ():
    args =parser .parse_args ()
    print_arguments (args )
    evaluate (args )
if __name__ =="__main__":
    main ()