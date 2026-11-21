from __future__ import print_function
import os
import subprocess
def str2bool (v ):
    if v .lower ()in ('yes','true','t','y','1'):
        return True
    elif v .lower ()in ('no','false','f','n','0'):
        return False
    else :
        raise argparse .ArgumentTypeError ('Обнаружено неподдерживаемое значение.')
def run_command (command ,shell =True ):
    print ("команда запуска: %s"%command )
    p =subprocess .Popen (
    command ,stdout =subprocess .PIPE ,stderr =subprocess .STDOUT ,shell =shell )
    exit_code =None
    stdout =''
    while exit_code is None or line :
        exit_code =p .poll ()
        line =p .stdout .readline ().decode ('utf-8')
        stdout +=line
    return stdout ,exit_code
def check_commit ():
    try :
        import tensorflow as tf
        tf_version =tf .__version__
    except Exception :
        tf_version =None
    try :
        import torch
        torch_version =torch .__version__
    except Exception :
        torch_version =None
    try :
        current_dir =os .getcwd ()
        print ("Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know."%current_dir )
        dir_of_this_file =os .path .dirname (os .path .abspath (__file__ ))
        print ("-- Входим в %s"%dir_of_this_file )
        os .chdir (dir_of_this_file )
        print ("Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know."%os .getcwd ())
        benchmark_commit ,_ =run_command ("git rev-parse HEAD")
        benchmark_commit =benchmark_commit .replace ("\n","")
        benchmark_update_time ,_ =run_command ("git show -s --format=%ad")
        benchmark_update_time =benchmark_update_time .replace ("\n","")
        os .chdir (current_dir )
        print ("Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know."%os .getcwd ())
        import paddle
        paddle_version =paddle .version .full_version
        paddle_commit =paddle .version .commit
        print (
        "==========================================================================="
        )
        print ("-- версия весла: %s"%paddle_version )
        print ("-- фиксация весла: %s"%paddle_commit )
        if tf_version :
            print ("-- версия тензорного потока: %s"%tf_version )
        if torch_version :
            print ("-- версия Pytorch: %s"%torch_version )
        print ("-- фиксация теста: %s"%benchmark_commit )
        print ("-- время последнего обновления теста: %s"%benchmark_update_time )
        print (
        "==========================================================================="
        )
    except Exception :
        pass