import logging
import sys
import os
_WORK_DIR =os .path .split (os .path .realpath (__file__ ))[0 ]
sys .path .append (os .path .join (_WORK_DIR ,'../../../../common'))
from static_graph .MMDNN .paddle import utils
def get_reader (conf_dict ,is_infer ,samples_file ):
    def reader_with_pairwise ():
        if is_infer :
            with open (conf_dict ["test_file_path"])as file :
                for line in file :
                    if not utils .pattern_match (r"(\d+)\t(\d+)\t((\d+ )*\d+)\t((\d+ )*\d+)\n",line ):
                        logging .warning ("строка не соответствует формату в тестовом файле")
                        continue
                    items =line .strip ("\n").split ("\t")
                    query =[int (id )for id in items [2 ].split (" ")]
                    title =[int (id )for id in items [3 ].split (" ")]
                    if samples_file :
                        samples_file .write (line )
                    yield [query ,title ]
        else :
            with open (conf_dict ["train_file_path"])as file :
                for line in file :
                    if not utils .pattern_match (r"((\d+ )*\d+)\t((\d+ )*\d+)\t((\d+ )*\d+)\n",line ):
                        logging .warning ("строка не соответствует формату в файле поезда")
                        continue
                    items =line .strip ("\n").split ("\t")
                    query =[int (id )for id in items [0 ].split (" ")]
                    pos_title =[int (id )for id in items [1 ].split (" ")]
                    neg_title =[int (id )for id in items [2 ].split (" ")]
                    if samples_file :
                        samples_file .write (line )
                    yield [query ,pos_title ,neg_title ]
    def reader_with_pointwise ():
        if is_infer :
            with open (conf_dict ["test_file_path"])as file :
                for line in file :
                    if not utils .pattern_match (r"((\d+ )*\d+)\t((\d+ )*\d+)\t(\d+)\n",line ):
                        logging .warning ("строка не соответствует формату в тестовом файле")
                        continue
                    items =line .strip ("\n").split ("\t")
                    query =[int (id )for id in items [0 ].split (" ")]
                    title =[int (id )for id in items [1 ].split (" ")]
                    if samples_file :
                        samples_file .write (line )
                    yield [query ,title ]
        else :
            with open (conf_dict ["train_file_path"])as file :
                for line in file :
                    if not utils .pattern_match (r"((\d+ )*\d+)\t((\d+ )*\d+)\t(\d+)\n",line ):
                        logging .warning ("строка не соответствует формату в файле поезда: %s"%line )
                        continue
                    items =line .strip ("\n").split ("\t")
                    query =[int (id )for id in items [0 ].split (" ")]
                    title =[int (id )for id in items [1 ].split (" ")]
                    label =int (items [2 ])
                    if samples_file :
                        samples_file .write (line )
                    yield [query ,title ,label ]
    if conf_dict ["task_mode"]=="pairwise":
        return reader_with_pairwise
    else :
        return reader_with_pointwise