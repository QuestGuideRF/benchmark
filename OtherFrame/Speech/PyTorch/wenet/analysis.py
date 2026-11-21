from __future__ import print_function
import argparse
import json
import os
import re
import traceback
def parse_args ():
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument (
    "--filename",type =str ,help ="Имя журнала, который необходимо проанализировать.")
    parser .add_argument (
    "--log_with_profiler",type =str ,help ="Путь журнала поездов с помощью профилировщика")
    parser .add_argument (
    "--profiler_path",type =str ,help ="Путь к журналу временной шкалы профилировщика.")
    parser .add_argument (
    "--keyword",type =str ,help ="Ключевое слово для указания данных анализа")
    parser .add_argument (
    "--separator",type =str ,default =None ,help ="Разделитель разных полей в журнале")
    parser .add_argument (
    '--position',type =int ,default =None ,help ='Положение поля данных')
    parser .add_argument (
    '--range',type =str ,default ="",help ='Диапазон поля данных для перехвата')
    parser .add_argument (
    '--base_batch_size',type =int ,help ='размер base_batch на графическом процессоре')
    parser .add_argument (
    '--skip_steps',type =int ,default =0 ,help ='Количество шагов, которые нужно пропустить')
    parser .add_argument (
    '--model_mode',type =int ,default =-1 ,help ='Режим анализа, значение по умолчанию — -1.')
    parser .add_argument (
    '--ips_unit',type =str ,default =None ,help ='блок IPS')
    parser .add_argument (
    '--model_name',type =str ,default =0 ,help ='имя_обучающей модели, Transformer_base')
    parser .add_argument (
    '--mission_name',type =str ,default =0 ,help ='название учебной миссии')
    parser .add_argument (
    '--direction_id',type =int ,default =0 ,help ='направление обучения_id')
    parser .add_argument (
    '--run_mode',type =str ,default ="sp",help ='многопроцессный или одиночный процесс')
    parser .add_argument (
    '--index',type =int ,default =1 ,help ='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    parser .add_argument (
    '--gpu_num',type =int ,default =1 ,help ='количество обучающих графических процессоров')
    args =parser .parse_args ()
    args .separator =None if args .separator =="None"else args .separator
    return args
def _is_number (num ):
    pattern =re .compile (r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result =pattern .match (num )
    if result :
        return True
    else :
        return False
class TimeAnalyzer (object ):
    def __init__ (self ,filename ,keyword =None ,separator =None ,position =None ,range ="-1"):
        if filename is None :
            raise Exception ("Пожалуйста, укажите имя файла!")
        if keyword is None :
            raise Exception ("Пожалуйста, укажите ключевое слово!")
        self .filename =filename
        self .keyword =keyword
        self .separator =separator
        self .position =position
        self .range =range
        self .records =None
        self ._distil ()
    def _distil (self ):
        self .records =[]
        with open (self .filename ,"r")as f_object :
            lines =f_object .readlines ()
            for line in lines :
                if self .keyword not in line :
                    continue
                try :
                    result =None
                    line =line .strip ()
                    line_words =line .split (self .separator )if self .separator else line .split ()
                    print ("line_words",line_words )
                    if args .position :
                        result =line_words [self .position ]
                    else :
                        for i in range (len (line_words )-1 ):
                            if line_words [i ]==self .keyword :
                                result =line_words [i +1 ]
                                break
                    if not self .range :
                        result =result [0 :]
                    elif _is_number (self .range ):
                        result =result [0 :int (self .range )]
                    else :
                        result =result [int (self .range .split (":")[0 ]):int (self .range .split (":")[1 ])]
                    self .records .append (float (result ))
                except Exception as exc :
                    pass
        print ("Извлечь записи {}: separator={}; позиция={}".format (len (self .records ),self .separator ,self .position ))
    def _get_fps (self ,mode ,batch_size ,gpu_num ,avg_of_records ,run_mode ,unit =None ):
        if mode ==-1 and run_mode =='sp':
            assert unit ,"Пожалуйста, установите единицу измерения, когда режим -1."
            fps =gpu_num *avg_of_records
        elif mode ==-1 and run_mode =='mp':
            assert unit ,"Пожалуйста, установите единицу измерения, когда режим -1."
            fps =gpu_num *avg_of_records
            print ("------------это МП")
        elif mode ==0 :
            fps =(batch_size *gpu_num )/avg_of_records
            unit ="samples/s"
        elif mode ==1 :
            fps =avg_of_records
            unit ="steps/s"
        elif mode ==2 :
            fps =1 /avg_of_records
            unit ="steps/s"
        elif mode ==3 :
            fps =batch_size *gpu_num *avg_of_records
            unit ="samples/s"
        elif mode ==4 :
            fps =avg_of_records
            unit ="s/epoch"
        else :
            ValueError ("Неподдерживаемый режим анализа.")
        return fps ,unit
    def analysis (self ,batch_size ,gpu_num =1 ,skip_steps =0 ,mode =-1 ,run_mode ='sp',unit =None ):
        if batch_size <=0 :
            print ("base_batch_size должно быть больше 0.")
            return 0 ,''
        if len (self .records )<=skip_steps :
            print ("нет записей")
            return 0 ,''
        sum_of_records =0
        sum_of_records_skipped =0
        skip_min =self .records [skip_steps ]
        skip_max =self .records [skip_steps ]
        count =len (self .records )
        for i in range (count ):
            sum_of_records +=self .records [i ]
            if i >=skip_steps :
                sum_of_records_skipped +=self .records [i ]
                if self .records [i ]<skip_min :
                    skip_min =self .records [i ]
                if self .records [i ]>skip_max :
                    skip_max =self .records [i ]
        avg_of_records =sum_of_records /float (count )
        avg_of_records_skipped =sum_of_records_skipped /float (count -skip_steps )
        fps ,fps_unit =self ._get_fps (mode ,batch_size ,gpu_num ,avg_of_records ,run_mode ,unit )
        fps_skipped ,_ =self ._get_fps (mode ,batch_size ,gpu_num ,avg_of_records_skipped ,run_mode ,unit )
        if mode ==-1 :
            print ("средний IP-адрес для %d шагов, пропустить 0 шагов:"%count )
            print ("\tСр.: %.3f %s"%(avg_of_records ,fps_unit ))
            print ("\tFPS: %.3f %s"%(fps ,fps_unit ))
            if skip_steps >0 :
                print ("средний IP-адрес для %d шагов, пропустить %d шагов:"%(count ,skip_steps ))
                print ("\tСр.: %.3f %s"%(avg_of_records_skipped ,fps_unit ))
                print ("\tМин: %.3f %s"%(skip_min ,fps_unit ))
                print ("\tМакс: %.3f %s"%(skip_max ,fps_unit ))
                print ("\tFPS: %.3f %s"%(fps_skipped ,fps_unit ))
        elif mode ==1 or mode ==3 :
            print ("средняя задержка %d шагов, пропустить 0 шагов:"%count )
            print ("\tAvg: %.3f шагов/с"%avg_of_records )
            print ("\tFPS: %.3f %s"%(fps ,fps_unit ))
            if skip_steps >0 :
                print ("средняя задержка: %d шагов, пропустить %d шагов:"%(count ,skip_steps ))
                print ("\tAvg: %.3f шагов/с"%avg_of_records_skipped )
                print ("\tMin: %.3f шагов/с"%skip_min )
                print ("\tМакс.: %.3f шагов/с"%skip_max )
                print ("\tFPS: %.3f %s"%(fps_skipped ,fps_unit ))
        elif mode ==0 or mode ==2 :
            print ("средняя задержка %d шагов, пропустить 0 шагов:"%count )
            print ("\tAvg: %.3f с/шаг"%avg_of_records )
            print ("\tFPS: %.3f %s"%(fps ,fps_unit ))
            if skip_steps >0 :
                print ("средняя задержка: %d шагов, пропустить %d шагов:"%(count ,skip_steps ))
                print ("\tAvg: %.3f с/шаг"%avg_of_records_skipped )
                print ("\tMin: %.3f с/шаг"%skip_min )
                print ("\tМакс: %.3f с/шаг"%skip_max )
                print ("\tFPS: %.3f %s"%(fps_skipped ,fps_unit ))
        return round (fps_skipped ,3 ),fps_unit
if __name__ =="__main__":
    args =parse_args ()
    run_info =dict ()
    run_info ["log_file"]=args .filename
    run_info ["model_name"]=args .model_name
    run_info ["mission_name"]=args .mission_name
    run_info ["direction_id"]=args .direction_id
    run_info ["run_mode"]=args .run_mode
    run_info ["index"]=args .index
    run_info ["gpu_num"]=args .gpu_num
    run_info ["FINAL_RESULT"]=0
    run_info ["JOB_FAIL_FLAG"]=0
    try :
        if args .index ==1 :
            if args .gpu_num ==1 :
                run_info ["log_with_profiler"]=args .log_with_profiler
                run_info ["profiler_path"]=args .profiler_path
            analyzer =TimeAnalyzer (args .filename ,args .keyword ,args .separator ,args .position ,args .range )
            run_info ["FINAL_RESULT"],run_info ["UNIT"]=analyzer .analysis (
            batch_size =args .base_batch_size ,
            gpu_num =args .gpu_num ,
            skip_steps =args .skip_steps ,
            mode =args .model_mode ,
            run_mode =args .run_mode ,
            unit =args .ips_unit )
        elif args .index ==3 :
            run_info ["FINAL_RESULT"]={}
            records_fo_total =TimeAnalyzer (args .filename ,'Накладные расходы на структуру',None ,3 ,'').records
            records_fo_ratio =TimeAnalyzer (args .filename ,'Накладные расходы на структуру',None ,5 ).records
            records_ct_total =TimeAnalyzer (args .filename ,'Время расчета',None ,3 ,'').records
            records_gm_total =TimeAnalyzer (args .filename ,'Вызовы GpuMemcpy',None ,4 ,'').records
            records_gm_ratio =TimeAnalyzer (args .filename ,'Вызовы GpuMemcpy',None ,6 ).records
            records_gmas_total =TimeAnalyzer (args .filename ,'GpuMemcpyAsync-вызовы',None ,4 ,'').records
            records_gms_total =TimeAnalyzer (args .filename ,'Вызовы GpuMemcpySync',None ,4 ,'').records
            run_info ["FINAL_RESULT"]["Framework_Total"]=records_fo_total [0 ]if records_fo_total else 0
            run_info ["FINAL_RESULT"]["Framework_Ratio"]=records_fo_ratio [0 ]if records_fo_ratio else 0
            run_info ["FINAL_RESULT"]["ComputationTime_Total"]=records_ct_total [0 ]if records_ct_total else 0
            run_info ["FINAL_RESULT"]["GpuMemcpy_Total"]=records_gm_total [0 ]if records_gm_total else 0
            run_info ["FINAL_RESULT"]["GpuMemcpy_Ratio"]=records_gm_ratio [0 ]if records_gm_ratio else 0
            run_info ["FINAL_RESULT"]["GpuMemcpyAsync_Total"]=records_gmas_total [0 ]if records_gmas_total else 0
            run_info ["FINAL_RESULT"]["GpuMemcpySync_Total"]=records_gms_total [0 ]if records_gms_total else 0
        else :
            print ("Не поддерживаю!")
    except Exception :
            traceback .print_exc ()
    print ("{}".format (json .dumps (run_info )))