import os
import ast
import sys
import argparse
import subprocess
import numpy as np
def parse_args ():
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    '--start_commit',
    type =str ,
    default ='8ec4af278d59d58ed9acc4657c7b6126fcfd8992',
    help ='начало пр поиска, это правильный пр.'
    )
    parser .add_argument (
    '--end_commit',
    type =str ,
    default ='4adac0e30978f3a28d10adf30f319e96ee3e03ed',
    help ='Конечный пр поиска, это пр с проблемой.'
    )
    parser .add_argument (
    '--is_perf',
    type =ast .literal_eval ,
    default =True ,
    help ='найдите pr с проблемой производительности или функцией, по умолчанию — проблема с производительностью.'
    )
    parser .add_argument (
    '--command',
    type =str ,
    default ='CUDA_VISIBLE_DEVICES=7 bash run_benchmark.sh 1 sp 300',
    help ='команда запуска скрипта.'
    )
    parser .add_argument (
    '--compile_WITH_DISTRIBUTE',
    type =str ,
    default ='ON',
    help ='параметр компиляции: WITH_DISTRIBUTE'
    )
    parser .add_argument (
    '--compile_ON_INFER',
    type =str ,
    default ='OFF',
    help ='параметр компиляции: ON_INFER'
    )
    parser .add_argument (
    '--compile_PY_VERSION',
    type =str ,
    default ='3.7',
    help ='параметр компиляции: PY_VERSION'
    )
    parser .add_argument (
    '--standard_value',
    type =float ,
    default =65.906 ,
    help ='стандартное значение, необходимое только в том случае, если is_perf=True.'
    )
    parser .add_argument (
    '--threshold',
    type =float ,
    default =0.05 ,
    help ='порог тревоги, по умолчанию 0,05.'
    )
    parser .add_argument (
    '--key_word',
    type =str ,
    default ='FINAL_RESULT',
    help ='ключевое слово для поиска данных о производительности в журнале.'
    )
    parser .add_argument (
    '--model_name',
    type =str ,
    default ='STGAN',
    help ='название модели'
    )
    args =parser .parse_args ()
    return args
def compile (commit_id ):
    uninstall_command ='pip удалить Paddlepaddle-GPU -y'
    os .system (uninstall_command )
    reset_command ='git сброс --hard %s'%commit_id
    os .system (reset_command )
    command ='rm -rf построить && построить mkdir'
    os .system (command )
    build_path =os .path .join (paddle_path ,'build')
    os .chdir (build_path )
    print ('зафиксировать {} начало сборки'.format (commit_id ))
    cmake_command ='cmake .. -DPY_VERSION={} -DWITH_GPU=ON -DWITH_DISTRIBUTE={} -DWITH_AMD_GPU=OFF -DWITH_MKL=ON -DWITH_NGRAPH=OFF -DWITH_AVX=ON -DWITH_GOLANG=OFF -DWITH_CONTRIB=ON -DWITH_INFERENCE_API_TEST=ON -DWITH_HIGH_LEVEL_API_TEST=OFF -DWITH_GRPC=ON -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF -DON_INFER={} -DCMAKE_BUILD_TYPE=Выпуск'.format (args .compile_PY_VERSION ,args .compile_WITH_DISTRIBUTE ,args .compile_ON_INFER )
    build_command ='сделать -j12'
    os .system (cmake_command )
    build_res =os .system (build_command )
    if build_res ==0 :
        print ('зафиксировать {} сборка завершена'.format (commit_id ))
    else :
        print ('зафиксировать {} сборка не удалась'.format (commit_id ))
        return -1
    install_command ='pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl'
    os .system (install_command )
    print ('зафиксировать {} установка завершена'.format (commit_id ))
    return 0
def save (commit_id ):
    mkdir_cmd ='мкдир {}'.format (commit_id )
    os .system (mkdir_cmd )
    mv_log_cmd ='mv {}* {}'.format (args .model_name ,commit_id )
    mv_wheel_cmd ='МВ весло/сборка/python/dist/paddlepaddle* {}'.format (commit_id )
    os .system (mv_log_cmd )
    os .system (mv_wheel_cmd )
def check_success (commit_id ):
    os .chdir (paddle_path )
    compile (commit_id )
    os .chdir (base_path )
    print ('базовый_путь:{}'.format (base_path ))
    if not args .is_perf :
        cmd =args .command
        cmd_res =os .system (cmd )
        save (commit_id )
        if cmd_res ==0 :
            return True
        else :
            return False
    else :
        cmd =args .command
        log =subprocess .getstatusoutput (cmd )
        save (commit_id )
        print ('бревно:{}'.format (log [1 ]))
        f =open ('log.txt','w')
        f .writelines (log [1 ])
        f .close ()
        cmd ='кот log.txt | тр "," "\n" | греп {}'.format (args .key_word )
        log_result =subprocess .getstatusoutput (cmd )
        f =open ('log_result.txt','w')
        f .writelines (log_result [1 ])
        f .close ()
        with open ('log_result.txt','r')as f :
            for line in f :
                sub_str =line .split (':')
                result =float (sub_str [1 ].strip ('\n'))
                print ('результат: {}'.format (result ))
                standard_value =args .standard_value
                print ('стандартное_значение — {}'.format (standard_value ))
                if abs ((result -standard_value )/standard_value )<=args .threshold :
                    return True
                else :
                    return False
def get_commits (start ,end ):
    print ('начинать:{}'.format (start ))
    print ('конец:{}'.format (end ))
    cmd ='git log {}..{} --pretty=oneline'.format (start ,end )
    log =subprocess .getstatusoutput (cmd )
    print (log [1 ])
    commit_list =[]
    candidate_commit =log [1 ].split ('\n')
    print (candidate_commit )
    for commit in candidate_commit :
        commit =commit .split (' ')[0 ]
        print ('совершить:{}'.format (commit ))
        commit_list .append (commit )
    return commit_list
def binary_search (commits ):
    if len (commits )==2 :
        print ('в бинарном_поиске осталось только два коммита-кандидата, последний коммит — {}'.format (commits [0 ]))
        return commits [0 ]
    left ,right =0 ,len (commits )-1
    if left <=right :
        mid =left +(right -left )//2
        commit =commits [mid ]
        if check_success (commit ):
            print ('фиксация {} выполнена успешно'.format (commit ))
            print ('среднее значение: {}'.format (mid ))
            selected_commits =commits [:mid +1 ]
            res =binary_search (selected_commits )
        else :
            print ('фиксация {} не удалась'.format (commit ))
            selected_commits =commits [mid :]
            res =binary_search (selected_commits )
    return res
if __name__ =='__main__':
    args =parse_args ()
    base_path =os .getcwd ()
    paddle_path =os .path .join (base_path ,'paddle')
    cmd ='rm -rf paddle && git clone http://github.com/paddlepaddle/paddle.git'
    os .system (cmd )
    os .chdir (paddle_path )
    commits =get_commits (start =args .start_commit ,end =args .end_commit )
    print ('кандидат фиксирует {}'.format (commits ))
    final_commit =binary_search (commits )
    print ('PR с проблемой: {}'.format (final_commit ))
    f =open ('final_commit.txt','w')
    f .writelines ('окончательный коммит: {}'.format (final_commit ))
    f .close ()