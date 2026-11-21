import argparse
import ast
import json
import os
from analysis import traverse_logs ,analysis
def parse_args ():
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    '--log_path',
    type =str ,
    default ='../../logs/static',
    help ='путь к журналам тестов')
    parser .add_argument (
    '--standard_path',
    type =str ,
    default ='../../scripts/benchmark_ci/standard_value/static',
    help ='путь к стандартному_значению')
    args =parser .parse_args ()
    return args
def modify_standard_value ():
    file_list =traverse_logs (args .log_path )
    for file in file_list :
        model ,fail_flag ,result ,loss_result =analysis (file )
        if int (fail_flag )==1 :
            print ("{} Не удалось выполнить запуск весла!".format (model ))
        else :
            print ("результат:{}".format (result ))
            print ("модель:{}".format (model ))
            standard_record =os .path .join (args .standard_path ,model +'.txt')
            with open (standard_record ,'r')as f :
                for line in f :
                    standard_result =line .strip ('\n')
                    print ("При установке лопатки результат будет соответствовать стандартному значению.")
                    command ='sed -i "s/{}/{}/g" {}'.format (standard_result ,result ,standard_record )
                    os .system (command )
if __name__ =='__main__':
    args =parse_args ()
    modify_standard_value ()