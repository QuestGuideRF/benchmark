import logging
import time
import json
import sys
import re
import os
def load_dic (path ):
    dic ={}
    with open (path )as f :
        for id ,line in enumerate (f ):
            w =line .strip ()
            dic [w ]=id
    return dic
def get_result_file (conf_dict ,samples_file_path ,predictions_file_path ):
    with open (conf_dict ["result_file_path"],"w")as result_file :
        with open (samples_file_path )as samples_file :
            with open (predictions_file_path )as predictions_file :
                samples =[line .strip ("\n").split ("\t")
                for line in samples_file ]
                predictions =[line .strip ("\n")for line in predictions_file ]
                if len (samples )!=len (predictions ):
                    logging .error ("Номер выборки не равен номеру прогноза!")
                    return
                if conf_dict ["task_mode"]=="pairwise":
                    for ind ,prediction in enumerate (predictions ):
                        result_file .write (
                        samples [ind ][0 ]+"\t"+samples [ind ][1 ]+"\t"+prediction +"\n")
                else :
                    for ind ,prediction in enumerate (predictions ):
                        result_file .write (samples [ind ][2 ]+"\t"+prediction +"\n")
    os .remove (samples_file_path )
    os .remove (predictions_file_path )
def parse_json (json_path ):
    try :
        with open (json_path )as json_file :
            json_dict =json .load (json_file )
    except Exception :
        logging .error ("Ошибка загрузки файла json!")
    else :
        return json_dict
def pattern_match (pattern ,line ):
    if re .match (pattern ,line ):
        return True
    else :
        return False
def print_progress (task_name ,percentage ,style =0 ):
    styles =['#','█']
    mark =styles [style ]*percentage
    mark +=' '*(100 -percentage )
    status ='%d%%'%percentage if percentage <100 else 'Finished'
    sys .stdout .write ('%+20s [%s] %s\r'%(task_name ,mark ,status ))
    sys .stdout .flush ()
    time .sleep (0.002 )
def display_args (name ,args ):
    logger =logging .getLogger (name )
    logger .info ("Аргументы, передаваемые по командной строке:")
    for k ,v in sorted (v for v in vars (args ).items ()):
        logger .info ("{}:\t{}".format (k ,v ))
def import_class (module_path ,module_name ,class_name ):
    if module_path :
        sys .path .append (module_path )
    module =__import__ (module_name )
    return getattr (module ,class_name )