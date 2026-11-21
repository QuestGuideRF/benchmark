from __future__ import print_function
import argparse
import json
import os
import sys
def parse_args ():
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument (
    "--filename",type =str ,help ="Имя журнала, который необходимо проанализировать.")
    parser .add_argument (
    "--jsonname",type =str ,help ="Имя сброшенного json, куда выводить.")
    parser .add_argument (
    "--keyword",type =str ,default ="time:",help ="Ключевое слово для указания данных анализа")
    parser .add_argument (
    '--model_name',type =str ,default ="fairmot",help ='имя_обучающей модели, Transformer_base')
    parser .add_argument (
    '--mission_name',type =str ,default ="目标检测",help ='название учебной миссии')
    parser .add_argument (
    '--direction_id',type =int ,default =0 ,help ='направление обучения_id')
    parser .add_argument (
    '--run_mode',type =str ,default ="sp",help ='многопроцессный или одиночный процесс')
    parser .add_argument (
    '--index',type =int ,default =1 ,help ='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    parser .add_argument (
    '--gpu_num',type =int ,default =1 ,help ='количество обучающих графических процессоров')
    parser .add_argument (
    '--batch_size',type =int ,default =1 ,help ='размер партии обучающих выборок')
    args =parser .parse_args ()
    return args
def parse_text_from_file (file_path :str ):
    with open (file_path ,"r")as f :
        lines =f .read ().splitlines ()
    return lines
def parse_avg_from_text (text :list ,keyword :str ,skip_line =4 ):
    count_list =[]
    for i ,line in enumerate (text ):
        if keyword in line :
            words =line .split (" ")
            for j ,word in enumerate (words ):
                if word ==keyword :
                    count_list .append (float (words [j +1 ][:-1 ]))
                    break
    count_list =count_list [skip_line :]
    if count_list :
        return round (sum (count_list )/len (count_list ),3 )
    else :
        return 0.0
if __name__ =='__main__':
    args =parse_args ()
    run_info =dict ()
    run_info ["log_file"]=args .filename
    res_log_file =args .jsonname
    run_info ["model_name"]=args .model_name
    run_info ["mission_name"]=args .mission_name
    run_info ["direction_id"]=args .direction_id
    run_info ["run_mode"]=args .run_mode
    run_info ["UNIT"]="images/s"
    run_info ["index"]=args .index
    run_info ["gpu_num"]=args .gpu_num
    run_info ["FINAL_RESULT"]=0
    run_info ["JOB_FAIL_FLAG"]=0
    text =parse_text_from_file (args .filename )
    avg_time =parse_avg_from_text (text ,args .keyword )
    if avg_time ==0.0 :
        run_info ["JOB_FAIL_FLAG"]=1
        print ("Не удалось получить информацию из журнала результатов обучения, проверьте.")
        sys .exit ()
    run_info ["FINAL_RESULT"]=round (args .batch_size /avg_time *args .gpu_num ,3 )
    json_info =json .dumps (run_info )
    with open (res_log_file +'.json',"w")as of :
        of .write (json_info )