from __future__ import print_function
import os
import json
import inspect
import importlib
import paddle .fluid as fluid
import sys
sys .path .append ("..")
from common import special_op_list
def import_layers (api_name ):
    def _import_func (module_name ,api_name ):
        try :
            module =importlib .import_module (module_name )
            func =getattr (module ,api_name )
            return func
        except Exception :
            return None
    if api_name in special_op_list .ALIAS_OP_MAP .keys ():
        api_name =special_op_list .ALIAS_OP_MAP [api_name ]
    all_module_names =[
    "paddle.fluid","paddle.fluid.layers","paddle.fluid.contrib.layers",
    "paddle"
    ]
    for module_name in all_module_names :
        func =_import_func (module_name ,api_name )
        if func is not None :
            break
    return func
def check_removable (api_name ,params ):
    if not isinstance (params ,dict )or not params :
        return True
    if api_name in special_op_list .CONTROL_FLOW_OPS :
        return True
    if api_name in special_op_list .EXCLUDE_OPS :
        return True
    return False
def check_frequency (api_name ,params ):
    if api_name in special_op_list .CONTROL_FLOW_OPS :
        return True
    if not isinstance (params ,dict )or not params :
        return False
    if api_name in special_op_list .EXCLUDE_OPS :
        return False
    return True
def check_and_clear_params (api_name ,params ,print_detail =False ):
    func =import_layers (api_name )
    assert func is not None ,"Невозможно импортировать %s из Paddle.fluid.layers и Paddle."%api_name
    if func is not None :
        argspec =inspect .getargspec (func )
        if print_detail :
            print ("   OP:",api_name ,",",argspec )
        no_need_args =[]
        if api_name in special_op_list .NO_NEED_ARGS .keys ():
            no_need_args =special_op_list .NO_NEED_ARGS [api_name ]
            if print_detail :
                print (no_need_args )
                print (type (no_need_args ))
        no_need_args .append ("name")
        for arg_name in argspec .args :
            if arg_name not in no_need_args and arg_name not in params .keys ():
                if print_detail :
                    print ("   Аргумент %s не установлен."%(arg_name ))
        non_params =[]
        for name ,content in params .items ():
            if name not in argspec .args or name in no_need_args :
                if print_detail :
                    if content ["type"]=="Variable":
                        print ("   Удалите %s (тип: %s, dtype: %s, форма: %s)."
                        %(name ,content ["type"],content ["dtype"],
                        content ["shape"]))
                    else :
                        print ("   Удалите %s (тип: %s, значение: %s)."%
                        (name ,content ["type"],content ["value"]))
                non_params .append (name )
        for name in non_params :
            params .pop (name )
        no_need_args .remove ("name")
def get_json_filenames (config_path ):
    abs_path =os .path .abspath (config_path )
    if os .path .isdir (abs_path ):
        filenames =[]
        files =os .listdir (abs_path )
        print ("В разделе %s находятся конфигурации %d."%(len (files ),abs_path ))
        for f in files :
            filenames .append (os .path .join (abs_path ,f ))
    else :
        filenames =[config_path ]
    return filenames
if __name__ =='__main__':
    config_path ="results"
    output_dir =os .path .abspath ("results_cleared")
    if not os .path .exists (output_dir ):
        os .makedirs (output_dir )
    filenames =get_json_filenames (config_path )
    for filename in filenames :
        try :
            data =[]
            with open (filename ,'r')as f :
                data =json .load (f )
                print ("-- Обработка %s: включая конфигурации %d."%
                (filename ,len (data )))
                remove_list =[]
                for i in range (0 ,len (data )):
                    op =data [i ]["op"]
                    params =data [i ]["param_info"]
                    if not check_removable (op ,params ):
                        check_and_clear_params (op ,params ,print_detail =True )
                    else :
                        remove_list .append (data [i ])
                for item in remove_list :
                    data .remove (item )
            if data :
                cleared_filename =os .path .join (output_dir ,
                os .path .basename (filename ))
                print ("-- Запись очищенных конфигураций %d обратно в %s.\n"%
                (len (data ),cleared_filename ))
                with open (cleared_filename ,'w')as f :
                    f .write (json .dumps (data ,sort_keys =True ,indent =4 ))
        except ValueError :
            print ("Невозможно декодировать как объект JSON в %s."%filename )
            break