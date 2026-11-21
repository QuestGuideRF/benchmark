from __future__ import print_function
import argparse
import os
import json
import sys
import warnings
import collections
import numpy as np
from common import utils
from common import system
from common import special_op_list
def _check_gpu_device (use_gpu ):
    gpu_devices =os .environ .get ("CUDA_VISIBLE_DEVICES",None )
    if use_gpu :
        assert gpu_devices ,"экспортируйте CUDA_VISIBLE_DEVICES=\"x\", чтобы проверить производительность графического процессора."
        assert len (gpu_devices .split (","))==1
    else :
        assert gpu_devices =="","экспортируйте CUDA_VISIBLE_DEVICES=\"\", чтобы проверить производительность процессора."
def parse_args ():
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument (
    '--filename',
    type =str ,
    default =None ,
    help ='Укажите имя файла теста')
    parser .add_argument (
    '--task',
    type =str ,
    default ="speed",
    help ='Укажите задачу: [скорость|точность|планирование]')
    parser .add_argument (
    '--testing_mode',
    type =str ,
    default ="static",
    help ='Укажите тип режима тестирования: [статический|динамический]')
    parser .add_argument (
    '--framework',
    type =str ,
    default ="paddle",
    help ='Укажите фреймворк: [paddle|tensorflow|tf|both]')
    parser .add_argument (
    '--json_file',type =str ,default =None ,help ='Файл параметров API')
    parser .add_argument (
    '--api_name',type =str ,default =None ,help ='Серия API')
    parser .add_argument (
    '--config_id',
    type =int ,
    default =None ,
    help ='Импортируйте параметры API только из файла json в указанной позиции [0|1|...]'
    )
    parser .add_argument (
    '--unknown_dim',
    type =int ,
    default =16 ,
    help ='Укажите неизвестное измерение.')
    parser .add_argument (
    '--profiler',
    type =str ,
    default ="none",
    help ='Выберите, какой профилировщик использовать [\"none\"|\"Default\"|\"OpDetail\"|\"AllOpDetail\"|\"pyprof\"]'
    )
    parser .add_argument (
    '--backward',
    type =system .str2bool ,
    default =False ,
    help ='Добавлять ли выпускные операции [True|False]')
    parser .add_argument (
    '--convert_to_fp16',
    type =system .str2bool ,
    default =False ,
    help ='Whether using gpu [True|False]')
    parser .add_argument (
    '--use_gpu',
    type =system .str2bool ,
    default =False ,
    help ='Whether using gpu [True|False]')
    parser .add_argument (
    '--gpu_time',
    type =float ,
    default =0 ,
    help ='Общее время ядра графического процессора, полученное из nvprof')
    parser .add_argument (
    '--scheduling_times',
    type =str ,
    default ="{}",
    help ='Планирование времени, анализируемое из nvprof')
    parser .add_argument (
    '--repeat',type =int ,default =1 ,help ='Итерации повторного бега')
    parser .add_argument (
    '--sync_interval',
    type =int ,
    default =80 ,
    help ='В задаче планирования синхронизацию необходимо выполнять «sync_interval» раз с интервалами.'
    )
    parser .add_argument (
    '--allow_adaptive_repeat',
    type =system .str2bool ,
    default =False ,
    help ='Использовать ли повтор значения в конфигурации json [True|False]')
    parser .add_argument (
    '--log_level',type =int ,default =0 ,help ='уровень логирования')
    parser .add_argument (
    '--get_status_without_running',
    type =system .str2bool ,
    default =False ,
    help ='Получите статус работы непосредственно, чтобы распечатать результат без повторного запуска программы.'
    )
    args =parser .parse_args ()
    if args .task not in ["speed","accuracy","scheduling"]:
        raise ValueError ("задачей должна быть скорость, точность или планирование")
    if args .framework not in [
    "paddle","tensorflow","tf","pytorch","torch","both"
    ]:
        raise ValueError (
        "задачей должно быть весло, тензорный поток, tf, pytorch, torch, оба")
    if args .get_status_without_running :
        assert args .task =="scheduling","задача должна быть «планируемой», если get_status_without_running имеет значение True."
        assert args .scheduling_times !="{}","scheduling_times не может быть {}, если задача находится в режиме планирования и get_status_without_running имеет значение True."
    if args .task =="accuracy":
        args .repeat =1
        args .profiler ="none"
    elif args .task =="scheduling":
        assert args .framework =="paddle","структура должна быть «веслом», если задача является «планированием»."
        assert args .testing_mode =="dynamic","test_mode должен быть «динамическим», если задача является «планирующей»."
        assert args .repeat >=10 ,"повтор должен быть больше 10, если задача запланирована, но получена {}.".format (
        args .repeat )
    _check_gpu_device (args .use_gpu )
    print (args )
    return args
def test_main (pd_obj =None ,
tf_obj =None ,
pd_dy_obj =None ,
torch_obj =None ,
config =None ):
    assert config is not None ,"Конфигурация API должна быть установлена."
    def _test_with_json_impl (filename ,config_id ,unknown_dim ,
    convert_to_fp16 ):
        config .init_from_json (filename ,config_id ,unknown_dim )
        if convert_to_fp16 :
            config .convert_to_fp16 ()
        if hasattr (config ,"api_list"):
            if args .api_name !=None :
                assert args .api_name in config .api_list ,"api_name должно быть одним значением в %s, но получено %s."%(
                config .api_list .keys (),args .api_name )
                config .api_name =args .api_name
                test_main_without_json (pd_obj ,tf_obj ,pd_dy_obj ,torch_obj ,
                config )
            else :
                for api_name in config .api_list .keys ():
                    config .api_name =api_name
                    test_main_without_json (pd_obj ,tf_obj ,pd_dy_obj ,
                    torch_obj ,config )
        else :
            test_main_without_json (pd_obj ,tf_obj ,pd_dy_obj ,torch_obj ,
            config )
    args =parse_args ()
    if args .json_file is not None :
        filename =config .alias_filename (args .json_file )
        if args .config_id is not None and args .config_id >=0 :
            _test_with_json_impl (filename ,args .config_id ,args .unknown_dim ,
            args .convert_to_fp16 )
        else :
            num_configs =0
            with open (filename ,'r')as f :
                num_configs =len (json .load (f ))
            for config_id in range (0 ,num_configs ):
                _test_with_json_impl (filename ,config_id ,args .unknown_dim ,
                args .convert_to_fp16 )
    else :
        test_main_without_json (pd_obj ,tf_obj ,pd_dy_obj ,torch_obj ,config )
def is_paddle_enabled (args ,config ):
    if args .task =="accuracy"or args .framework in ["paddle","both"]:
        return True
    return False
def is_tensorflow_enabled (args ,config ):
    if config .run_tf and args .testing_mode =="static":
        if args .task =="accuracy"or args .framework in [
        "tensorflow","tf","both"
        ]:
            return True
    return False
def is_torch_enabled (args ,config ):
    if config .run_torch and args .testing_mode =="dynamic":
        if args .task =="accuracy"or args .framework in [
        "torch","pytorch","both"
        ]:
            return True
    return False
def _adaptive_repeat (config ,args ):
    if args .task =="speed"and args .allow_adaptive_repeat and hasattr (
    config ,"repeat"):
        if args .use_gpu :
            args .repeat =config .repeat
def _check_disabled (config ,args ):
    if config .disabled ():
        status =collections .OrderedDict ()
        status ["name"]=config .api_name
        status ["device"]="GPU"if args .use_gpu else "CPU"
        status ["backward"]=args .backward if special_op_list .has_backward (
        config )else False
        status ["disabled"]=True
        status ["parameters"]=config .to_string ()
        print (json .dumps (status ))
        return True
    return False
def test_main_without_json (pd_obj =None ,
tf_obj =None ,
pd_dy_obj =None ,
torch_obj =None ,
config =None ):
    assert config is not None ,"Конфигурация API должна быть установлена."
    args =parse_args ()
    if _check_disabled (config ,args ):
        return
    _adaptive_repeat (config ,args )
    config .backward =args .backward
    use_feed_fetch =True if args .task =="accuracy"else False
    feeder_adapter =None
    if is_tensorflow_enabled (args ,config ):
        assert tf_obj is not None ,"Объект TensorFlow имеет значение None."
        tf_config =config .to_tensorflow ()
        print (tf_config )
        feeder_adapter =tf_obj .generate_random_feeder (tf_config ,
        use_feed_fetch )
        tf_outputs ,tf_stats =tf_obj .run (tf_config ,args ,use_feed_fetch ,
        feeder_adapter )
        if args .task =="speed":
            tf_stats ["gpu_time"]=args .gpu_time
            utils .print_benchmark_result (
            tf_stats ,
            task =args .task ,
            log_level =args .log_level ,
            config_params =config .to_string ())
    if is_paddle_enabled (args ,config )and args .testing_mode =="static":
        assert pd_obj is not None ,"Объект Paddle имеет значение None."
        print (config )
        pd_outputs ,pd_stats =pd_obj .run (config ,args ,use_feed_fetch ,
        feeder_adapter )
        if args .task =="speed":
            pd_stats ["gpu_time"]=args .gpu_time
            utils .print_benchmark_result (
            pd_stats ,
            task =args .task ,
            log_level =args .log_level ,
            config_params =config .to_string ())
        if pd_outputs ==False :
            sys .exit (1 )
    if is_torch_enabled (args ,config ):
        assert torch_obj is not None ,"Объект PyTorch имеет значение None."
        import torch
        try :
            import paddle
            flags =paddle .get_flags (["FLAGS_cudnn_exhaustive_search"])
            torch .backends .cudnn .benchmark =flags [
            "FLAGS_cudnn_exhaustive_search"]
        except Exception :
            torch .backends .cudnn .benchmark =os .environ .get (
            "FLAGS_cudnn_exhaustive_search",False )
        torch_config =config .to_pytorch ()
        print (torch_config )
        torch_outputs ,torch_stats =torch_obj .run (torch_config ,args )
        feeder_adapter =torch_obj .generate_random_feeder (torch_config )
        if args .task =="speed":
            torch_stats ["gpu_time"]=args .gpu_time
            utils .print_benchmark_result (
            torch_stats ,
            task =args .task ,
            log_level =args .log_level ,
            config_params =config .to_string ())
    if is_paddle_enabled (args ,config )and args .testing_mode =="dynamic":
        assert pd_dy_obj is not None ,"Динамический объект Paddle имеет значение None."
        print (config )
        pd_dy_outputs ,pd_dy_stats =pd_dy_obj .run (
        config ,args ,feeder_adapter =feeder_adapter )
        if args .task in ["speed","scheduling"]:
            if args .task =="speed":
                pd_dy_stats ["gpu_time"]=args .gpu_time
            if args .task =="scheduling":
                pd_dy_stats ["scheduling_times"]=args .scheduling_times
            utils .print_benchmark_result (
            pd_dy_stats ,
            task =args .task ,
            log_level =args .log_level ,
            config_params =config .to_string ())
        if pd_dy_outputs ==False :
            sys .exit (1 )
    if args .task =="accuracy":
        is_run_tf =config .run_tf and args .testing_mode =="static"
        is_run_torch =config .run_torch and args .testing_mode =="dynamic"
        if is_run_tf :
            base_outputs =pd_outputs
            compare_outputs =tf_outputs
            backward =pd_obj .backward
        elif is_run_torch :
            base_outputs =pd_dy_outputs
            compare_outputs =torch_outputs
            backward =pd_dy_obj .backward
        if is_run_tf or is_run_torch :
            if args .log_level ==1 :
                for i in range (len (base_outputs )):
                    out =base_outputs [i ]
                    if isinstance (out ,np .ndarray ):
                        print (
                        "{}-й вывод Paddle — это np.ndarray, форма — {}.".
                        format (i ,out .shape ))
            if args .log_level ==2 :
                print ("Выход весла: ",base_outputs )
                print ("Вывод TensorFlow: ",compare_outputs )
            utils .check_outputs (
            base_outputs ,
            compare_outputs ,
            args .testing_mode ,
            name =config .api_name ,
            atol =config .atol ,
            use_gpu =args .use_gpu ,
            backward =backward ,
            config_params =config .to_string ())