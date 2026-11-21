import os
import copy
import json
import argparse
import shutil
import requests
parser =argparse .ArgumentParser (description =__doc__ )
parser .add_argument (
"--static_models",
type =str ,
default ='',
help ="запросить список статических моделей")
parser .add_argument (
"--dynamic_models",
type =str ,
default ='',
help ="запрос списка динамических моделей")
parser .add_argument (
"--save_dir",
type =str ,
default ='./diff',
help ="индексный результат, каталог сохранения")
parser .add_argument (
"--output_file",
type =str ,
default ='.rerun_list',
help ="выходной_файл, записывает, какую модель необходимо перезапустить")
def _query_latest_image_id ():
    url ="http://benchmark_server:8988/benchmark/image/"
    params ={
    "cuda_version":"10.1",
    "image_type":"2"
    }
    static_response =requests .get (url ,params =params ).json ()
    static_image_info =None
    dynamic_image_info =None
    for item in static_response :
        static_image_info ={
        "image_id":item ["image_id"],
        "version":item ["version"],
        "image_commit_id":item ["image_commit_id"]
        }
        break
    params ={
    "cuda_version":"10.1",
    "image_type":"3"
    }
    dynamic_response =requests .get (url ,params =params ).json ()
    for item in dynamic_response :
        dynamic_image_info ={
        "image_id":item ["image_id"],
        "version":item ["version"],
        "image_commit_id":item ["image_commit_id"]
        }
        break
    return static_image_info ,dynamic_image_info
def _check_result_file_is_need (model_type ,file_name ,model_list ):
    if model_type =="static_graph":
        model_list =[x .lower ()for x in model_list ]
        file_name =file_name .lower ()
    elif model_type =="dynamic_graph":
        model_list =[x [3 :].lower ()for x in model_list ]
        file_name =file_name .replace ("dynamic_","").lower ()
    for model_name in model_list :
        if file_name .startswith (model_name ):
            return True
    return False
def _copy_history_result_to_save_dir (save_dir ,image_version ,model_type ,model_list ):
    base_dir =os .environ .get ("base_dir")
    source_dir ="%s/logs/%s/%s/index/"%(base_dir ,image_version ,model_type )
    destination_dir =save_dir +"/"+model_type +"/index"
    os .makedirs (destination_dir ,exist_ok =True )
    if not os .path .exists (source_dir ):
        return
    for file_name in os .listdir (source_dir ):
        need_copy =_check_result_file_is_need (model_type ,file_name ,model_list )
        if need_copy :
            shutil .copy (os .path .join (source_dir ,file_name ),os .path .join (destination_dir ,file_name ))
def _check_model_result_file_exists (save_dir ,model_type ,model_name ):
    destination_dir =save_dir +"/"+model_type +"/index"
    if not os .path .exists (destination_dir ):
        return False
    for file_name in os .listdir (destination_dir ):
        if model_type =="static_graph":
            model_name =model_name .lower ()
            file_name =file_name .lower ()
        elif model_type =="dynamic_graph":
            model_name =model_name [3 :].lower ()
            file_name =file_name .replace ("dynamic_","").lower ()
        if file_name .startswith (model_name ):
            return True
    return False
def _calculate_remain_models (save_dir ,output_file ,static_models ,dynamic_models ,static_image_id ,dynamic_image_id ):
    current_dir =os .path .dirname (os .path .abspath (__file__ ))
    config_file =current_dir +"/models.json"
    with open (config_file ,"r")as f :
        data =json .loads (f .read ())
        all_static_graph =data ["static_graph"]
        all_dynamic_graph =data ["dynamic_graph"]
    if "all"in static_models :
        static_models =all_static_graph
    if "all"in dynamic_models :
        dynamic_models =all_dynamic_graph
    for model_name in copy .deepcopy (static_models ):
        is_exists =_check_model_result_file_exists (save_dir ,"static_graph",model_name )
        if is_exists :
            static_models .remove (model_name )
    for model_name in copy .deepcopy (dynamic_models ):
        is_exists =_check_model_result_file_exists (save_dir ,"dynamic_graph",model_name )
        if is_exists :
            dynamic_models .remove (model_name )
    with open (output_file ,"w")as f :
        if static_models :
            f .write ("static_graph %s %s\n"%(",".join (static_models ),static_image_id ))
        if dynamic_models :
            f .write ("динамический_граф %s %s\n"%(",".join (dynamic_models ),dynamic_image_id ))
def _query_latest_daily_result (args ):
    save_dir =os .path .abspath (args .save_dir )
    output_file =args .output_file
    if args .static_models .strip ():
        static_models =[x .strip ()for x in args .static_models .strip ().split (",")]
    else :
        static_models =[]
    if args .dynamic_models .strip ():
        dynamic_models =[x .strip ()for x in args .dynamic_models .strip ().split (",")]
    else :
        dynamic_models =[]
    os .makedirs (save_dir ,exist_ok =True )
    static_image_info ,dynamic_image_info =_query_latest_image_id ()
    print ("static_image_info: %s"%str (static_image_info ))
    print ("динамическое_изображение_информация: %s"%str (dynamic_image_info ))
    if static_models :
        _copy_history_result_to_save_dir (save_dir ,static_image_info ["version"],"static_graph",static_models )
    if dynamic_models :
        _copy_history_result_to_save_dir (save_dir ,dynamic_image_info ["version"],"dynamic_graph",dynamic_models )
    _calculate_remain_models (
    save_dir ,output_file ,static_models ,dynamic_models ,
    static_image_info ["version"],dynamic_image_info ["version"]
    )
if __name__ =="__main__":
    args =parser .parse_args ()
    _query_latest_daily_result (args )