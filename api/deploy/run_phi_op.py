import datetime
import os
import argparse
import yaml
import wget
def get_phi_op_list ():
    DATA_URL ="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/python/paddle/utils/code_gen/api.yaml"
    file_name ="phi_api_list.yaml"
    try :
        if os .path .exists (file_name ):
            os .remove (file_name )
            print ("Старый список фи уже существует в каталоге и удаляется.")
        print ("开始下载新phi列表")
        wget .download (DATA_URL ,out =file_name )
        print ("下载完成")
    except Exception as e :
        print ("下载yaml失败")
        print (e )
        exit (1 )
    try :
        with open (file_name )as f :
            yml =yaml .load (f ,Loader =yaml .FullLoader )
    except Exception as e :
        print ("загрузка yaml не удалась")
        print (e )
        exit (1 )
    phi_op_list =[]
    for i in yml :
        phi_op_list .append (i ["api"])
    phi_op_string =",".join (i for i in phi_op_list )
    return phi_op_string
def main ():
    pass
if __name__ =='__main__':
    parser =argparse .ArgumentParser (description =__doc__ )
    parser .add_argument (
    '--base_dir',type =str ,help ='введите каталог тестового кода')
    parser .add_argument ('--log_dir',type =str ,help ='введите каталог журнала')
    parser .add_argument (
    '--op_list',type =str ,default =None ,help ='укажите список операторов.')
    args =parser .parse_args ()
    base_dir =args .base_dir
    main_control =base_dir +"/deploy/main_control.sh"
    module_name =base_dir +"/tests"
    configs =base_dir +"/tests_v2/configs"
    logdir =args .log_dir
    oplist =args .op_list
    if oplist is None :
        oplist =get_phi_op_list ()
    cmd ="bash {} {} {} {} 0 скорость графического процессора нет оба динамические {}".format (
    main_control ,module_name ,configs ,logdir ,oplist )
    print ("[коммд] {}".format (cmd ))
    os .system (cmd )