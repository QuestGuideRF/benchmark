from __future__ import print_function
import os
import sys
import json
import argparse
import time
import traceback
from collections import OrderedDict
import numpy as np
sys .path .append (os .path .dirname (os .path .dirname (os .path .abspath (__file__ ))))
sys .path .append (
os .path .abspath (os .path .join (os .path .dirname (__file__ ),"../../")))
from common import system
import op_benchmark_unit
TABLE_HEADER =["case_name","指标","标准值","当前值","波动范围"]
CHECK_KEY =OrderedDict ()
CHECK_KEY ["gpu_time"]="GPU正向内核"
CHECK_KEY ["gpu_time_backward"]="GPU反向内核"
CHECK_KEY ["paddle_gpu_perf"]="GPU正向"
CHECK_KEY ["paddle_gpu_perf_backwards"]="GPU反向"
CHECK_KEY ["paddle_cpu_perf"]="CPU正向"
CHECK_KEY ["paddle_cpu_perf_backwards"]="CPU反向"
def _is_json (line ):
    try :
        result =json .loads (line )
    except Exception :
        return False
    return True if isinstance (result ,dict )else False
def _read_last_line (inputfile ):
    def _read_last_block (f ,filesize ,blocksize ):
        if filesize >blocksize :
            maxseekpoint =filesize //blocksize
            f .seek ((maxseekpoint -1 )*blocksize )
        elif filesize :
            f .seek (0 ,0 )
        last_line =None
        lines =f .readlines ()
        if len (lines )>=2 :
            for line in reversed (lines ):
                if _is_json (line .strip ("\n")):
                    last_line =line .strip ("\n")
                    break
        return last_line
    filesize =os .path .getsize (inputfile )
    f =open (inputfile ,'r')
    last_line =None
    blocksize =1024
    while blocksize <filesize :
        last_line =_read_last_block (f ,filesize ,blocksize )
        if last_line is not None :
            break
        blocksize *=2
    if last_line is None :
        last_line =_read_last_block (f ,filesize ,blocksize )
    f .close ()
    return last_line
def _parse_disabled_status (res ,case_name ,last_line ):
    assert res .get (case_name ,None )is not None
    case_detail =res [case_name ]
    if case_detail .get ("disabled",None )is None :
        try :
            data =json .loads (last_line )
            disabled =data .get ("disabled",False )
            res [case_name ]["disabled"]=disabled
        except Exception :
            pass
def _parse_parameters (res ,case_name ,last_line ):
    assert res .get (case_name ,None )is not None
    case_detail =res [case_name ]
    if case_detail .get ("parameters",None )is None :
        try :
            data =json .loads (last_line )
            param =data ['parameters']
            res [case_name ]['parameters']=param .strip ("\n")
        except Exception :
            pass
def _parse_speed (res ,case_name ,statistic_type ,last_line ):
    assert res .get (case_name ,None )is not None
    speed_key_map ={
    "paddle_gpu_speed_forward":"%",
    "paddle_gpu_speed_backward":"%_backward",
    "tensorflow_gpu_speed_forward":"tf_%",
    "tensorflow_gpu_speed_backward":"tf_%_backward",
    "pytorch_gpu_speed_forward":"pytorch_%",
    "pytorch_gpu_speed_backward":"pytorch_%_backward"
    }
    speed_key =speed_key_map .get (statistic_type ,None )
    if speed_key is not None :
        gpu_time_key =speed_key .replace ("%","gpu_time")
        gflops_key =speed_key .replace ("%","gflops")
        gbs_key =speed_key .replace ("%","gbs")
    else :
        gpu_time_key =None
        gflops_key =None
        gbs_key =None
    try :
        data =json .loads (last_line )
        total =data ["speed"]["total"]
        total_str ="%.5f"%total
        res [case_name ][statistic_type ]=total_str
        if gpu_time_key and data ["speed"].get ("gpu_time",None )is not None :
            gpu_time_str ="%.5f"%data ["speed"]["gpu_time"]
            res [case_name ][gpu_time_key ]=gpu_time_str
        if gflops_key and data ["speed"].get ("gflops",None )is not None :
            gflops_str ="%.5f"%data ["speed"]["gflops"]
            res [case_name ][gflops_key ]=gflops_str
        if gbs_key and data ["speed"].get ("gbs",None )is not None :
            gbs_str ="%.5f"%data ["speed"]["gbs"]
            res [case_name ][gbs_key ]=gbs_str
    except Exception :
        res [case_name ][statistic_type ]="--"
        if gpu_time_key :
            res [case_name ][gpu_time_key ]="--"
        if gflops_key :
            res [case_name ][gflops_key ]="--"
        if gbs_key :
            res [case_name ][gbs_key ]="--"
def _parse_accuracy (res ,case_name ,statistic_type ,last_line ):
    assert res .get (case_name ,None )is not None
    difference_key =statistic_type .replace ("accuracy","difference")
    try :
        data =json .loads (last_line )
        res [case_name ][statistic_type ]=data ["consistent"]
        res [case_name ][difference_key ]=data ["diff"]
    except Exception :
        res [case_name ][statistic_type ]="--"
        res [case_name ][difference_key ]="--"
def get_job_res (res ,inputfile ,specified_op_list =None ):
    file_name =os .path .splitext (os .path .basename (inputfile ))[0 ]
    file_name =file_name .replace ("_fp32","").replace ("_fp16","")
    case_name =file_name .split ("-")[0 ]
    op_type =op_benchmark_unit .parse_op_type (case_name )
    if specified_op_list and op_type not in specified_op_list :
        return None
    print ("-- Анализ %s из %s"%(case_name ,inputfile ))
    if case_name not in res :
        res [case_name ]={}
    statistic_beg_idx =file_name .find ("-")
    statistic_type =file_name [statistic_beg_idx +1 :]
    framework =statistic_type .split ("_")[0 ]
    last_line =_read_last_line (inputfile )
    _parse_disabled_status (res ,case_name ,last_line )
    _parse_parameters (res ,case_name ,last_line )
    if last_line and "_speed_"in statistic_type :
        _parse_speed (res ,case_name ,statistic_type ,last_line )
    if last_line and "_accuracy_"in statistic_type :
        _parse_accuracy (res ,case_name ,statistic_type ,last_line )
    return framework
def check_results (op_record ,alarm_results ):
    from benchmark_op import models
    results =models .OpRecord2 .objects .filter (
    case_name =op_record .case_name ,
    version =op_record .version ).order_by ('-timestamp')[:10 :1 ]
    for key ,verbose in CHECK_KEY .items ():
        results_list =[]
        count =0
        for result in results :
            if count ==0 :
                count +=1
                continue
            if len (results_list )==5 :
                break
            try :
                if result and getattr (result ,key )!='--':
                    result =json .loads (getattr (result ,key ))
                    result =float (result )
                    results_list .append (result )
            except Exception as exc :
                print ("добавить ошибку данных истории {}".format (exc ))
        if not results_list :
            continue
        try :
            avg_values =round (np .array (results_list ).mean (),4 )
            if not avg_values :
                continue
            ranges =round (
            (float (getattr (op_record ,key ))-avg_values )/avg_values ,4 )
        except Exception as rw :
            print ("ошибка определения диапазона {}".format (rw ))
            traceback .print_exc ()
            ranges =-1
        if -0.05 <ranges <0.05 :
            continue
        if ranges >=0.05 :
            color ="red"
        elif ranges <=-0.05 :
            color ="green"
        current_html_result =[
        dict (value =op_record .case_name ),dict (value =verbose ),
        dict (value =avg_values ),dict (value =getattr (op_record ,key )),dict (
        value ='%.2f%%'%(ranges *100 ),color =color )
        ]
        alarm_results [verbose ]["data"].append (current_html_result )
def construct_alarm_email (timestamp ,alarm_results ):
    import scripts .template as template
    flag =False
    for k ,v in alarm_results .items ():
        if alarm_results [k ]["data"]:
            flag =True
            break
    if flag :
        title ="op_benchmark"
        env =dict (
        PADDLE_VERSION =timestamp ,
        DOCKER_IMAGES =os .getenv ('RUN_IMAGE_NAME'),
        CUDA_VERSION =os .getenv ('CUDA_VERSION'),
        CUDNN_VERSION =os .getenv ('CUDNN_VERSION'),
        PADDLE_COMMIT_ID =os .getenv ('PADDLE_COMMIT_ID'))
        email_t =template .EmailTemplate (title ,env ,alarm_results ,
        args .op_result_dir )
        email_t .construct_email_content ()
def dump_mysql (data ,version ,construct_email ):
    import models .benchmark_server .helper as helper
    from benchmark_op import models
    timestamp =os .getenv ("PADDLE_VERSION",time .time ())
    alarm_results =OrderedDict ()
    for k ,v in CHECK_KEY .items ():
        alarm_results [v ]={"header":TABLE_HEADER ,"data":[]}
    for i in range (len (data )):
        dic =data [i ]
        op_record =models .OpRecord2 ()
        op_record .timestamp =timestamp
        op_record .case_name =dic ['name']
        op_record .paddle_cpu_accuracy =dic .get ("paddle_cpu_accuracy_forward",
        "--")
        op_record .paddle_cpu_accuracy_backwards =dic .get (
        "paddle_cpu_accuracy_backward","--")
        op_record .paddle_gpu_accuracy =dic .get ("paddle_gpu_accuracy_forward",
        "--")
        op_record .paddle_gpu_accuracy_backwards =dic .get (
        "paddle_gpu_accuracy_backward","--")
        op_record .paddle_cpu_perf =dic .get ("paddle_cpu_speed_forward","--")
        op_record .tf_cpu_perf =dic .get ("tensorflow_cpu_speed_forward","--")
        op_record .paddle_gpu_perf =dic .get ("paddle_gpu_speed_forward","--")
        op_record .tf_gpu_perf =dic .get ("tensorflow_gpu_speed_forward","--")
        op_record .paddle_cpu_perf_backwards =dic .get (
        "paddle_cpu_speed_backward","--")
        op_record .tf_cpu_perf_backwards =dic .get (
        "tensorflow_cpu_speed_backward","--")
        op_record .paddle_gpu_perf_backwards =dic .get (
        "paddle_gpu_speed_backward","--")
        op_record .tf_gpu_perf_backwards =dic .get (
        "tensorflow_gpu_speed_backward","--")
        op_record .config =dic .get ("parameters","--")
        op_record .gpu_time_backward =dic .get ("gpu_time_backward","--")
        op_record .gpu_time =dic .get ("gpu_time","--")
        op_record .tf_gpu_time_backward =dic .get ("tf_gpu_time_backward","--")
        op_record .tf_gpu_time =dic .get ("tf_gpu_time","--")
        op_record .version =version
        op_record .save ()
        check_results (op_record ,alarm_results )
    if construct_email :
        construct_alarm_email (timestamp ,alarm_results )
def parse_logs (op_result_dir ,specified_op_list ):
    filenames =os .listdir (op_result_dir )
    if "api_info.txt"in filenames :
        filenames .remove ('api_info.txt')
    assert len (filenames )>0 ,"Каталог %s пуст."%op_result_dir
    fp32_res ={}
    fp16_res ={}
    compare_framework =None
    for filename in sorted (filenames ):
        res =fp16_res if "_fp16"in filename else fp32_res
        framework =get_job_res (res ,
        os .path .join (op_result_dir ,filename ),
        specified_op_list )
        if framework is not None and framework !="paddle":
            if compare_framework :
                assert framework ==compare_framework ,"Имя платформы анализируется из имени файла результата \
                    ожидается %s, но получено %s."%(
                compare_framework ,framework )
            else :
                compare_framework =framework
    fp32_benchmark_result_list =[]
    fp16_benchmark_result_list =[]
    for precision in ["fp32","fp16"]:
        res =fp32_res if precision =="fp32"else fp16_res
        for key ,value in sorted (
        res .items (),
        key =lambda t :op_benchmark_unit .unify_case_name (t [0 ])):
            case_detail =value .copy ()
            case_detail ['name']=key
            op_unit =op_benchmark_unit .OpBenchmarkUnit (case_detail ,
            compare_framework )
            if precision =="fp32":
                fp32_benchmark_result_list .append (op_unit )
            else :
                fp16_benchmark_result_list .append (op_unit )
    return fp32_benchmark_result_list ,fp16_benchmark_result_list ,compare_framework
if __name__ =='__main__':
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument (
    'op_result_dir',
    type =str ,
    default =None ,
    help ='Укажите каталог результатов теста оператора.')
    parser .add_argument (
    '--specified_op_list',
    type =str ,
    default =None ,
    help ='Укажите список операторов.')
    parser .add_argument (
    '--op_frequency_path',
    type =str ,
    default =None ,
    help ='Укажите путь к данным частоты оператора.')
    parser .add_argument (
    '--dump_to_text',
    type =system .str2bool ,
    default =False ,
    help ='Сбрасывать ли сводные данные в текстовый файл [True|False]')
    parser .add_argument (
    '--dump_to_excel',
    type =system .str2bool ,
    default =False ,
    help ='Сбрасывать ли сводные данные в Excel [True|False]')
    parser .add_argument (
    '--dump_with_parameters',
    type =system .str2bool ,
    default =False ,
    help ='Сбрасывать ли сводные данные в текст [True|False]')
    parser .add_argument (
    '--url_prefix',
    type =str ,
    default =None ,
    help ='Укажите префикс URL-адресов выходных журналов.')
    parser .add_argument (
    '--output_path',
    type =str ,
    default =None ,
    help ='Укажите путь вывода.')
    parser .add_argument (
    '--dump_to_mysql',
    type =system .str2bool ,
    default =False ,
    help ='Сбрасывать ли сводные данные в базу данных MySQL [True|False]')
    parser .add_argument (
    '--dump_to_json',
    type =system .str2bool ,
    default =False ,
    help ='Сбрасывать ли сводные данные в файл json [True|False]')
    parser .add_argument (
    '--version',
    type =str ,
    default ='1.8',
    help ='Укажите версию весла.')
    parser .add_argument (
    '--construct_email',
    type =system .str2bool ,
    default =True ,
    help ='Создавать ли тревожное электронное письмо [True|False]')
    parser .add_argument (
    "--card",type =str ,default =None ,help ="Номер карты, по умолчанию 0.")
    parser .add_argument (
    "--host",type =str ,default =None ,help ="Адрес хоста базы данных MySQL")
    parser .add_argument (
    "--user",type =str ,default =None ,help ="Имя пользователя базы данных MySQL")
    parser .add_argument (
    "--password",type =str ,default =None ,help ="Пароль базы данных MySQL")
    parser .add_argument (
    "--database",type =str ,default =None ,help ="Имя выбранной базы данных")
    args =parser .parse_args ()
    op_result_dir =os .path .abspath (args .op_result_dir )
    assert os .path .exists (
    op_result_dir ),"Каталог %s не существует."%op_result_dir
    specified_op_list =None
    if args .specified_op_list :
        specified_op_list =args .specified_op_list .split ()
    fp32_benchmark_result_list ,fp16_benchmark_result_list ,compare_framework =parse_logs (
    op_result_dir ,specified_op_list )
    op_frequency_dict =None
    if args .op_frequency_path :
        import read_frequency
        op_frequency_dict =read_frequency .read_frequency_from_text (
        args .op_frequency_path )
    if args .dump_to_text :
        import write_text
        write_text .dump_text (fp32_benchmark_result_list ,args .output_path ,
        compare_framework ,args .dump_with_parameters )
    if args .dump_to_excel :
        import write_excel
        write_excel .dump_excel (fp32_benchmark_result_list ,
        fp16_benchmark_result_list ,op_result_dir ,
        args .url_prefix ,args .output_path ,
        compare_framework ,op_frequency_dict )
    if args .dump_to_json :
        import write_json
        write_json .dump_json (fp32_benchmark_result_list ,args .output_path ,
        compare_framework ,args .dump_with_parameters )
    if args .dump_to_mysql :
        if args .host is None or args .user is None or args .password is None or args .database is None :
            print ("Пожалуйста, заполните информацию о базе данных!")
        else :
            from write_mysql import DB
            db =DB (args .host ,args .user ,args .password ,args .database )
            try :
                db .write_database (fp32_benchmark_result_list ,
                compare_framework ,args .card )
            except Exception as e :
                print ("Не удалось сохранить данные в MySQL, проверьте причину!")
                print (e )