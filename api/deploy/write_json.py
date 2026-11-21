import sys
import os
import time
import json
import op_benchmark_unit
COMPARE_RESULT_SHOWS ={
"Better":"优于",
"Equal":"打平",
"Less":"差于",
"Unknown":"未知",
"Unsupport":"不支持",
"Others":"其他",
"Total":"汇总"
}
def _create_summary_json (compare_result ,category ):
    summary_json_result =list ()
    compare_result_keys =compare_result .compare_result_keys
    titles ={"title":1 ,"row_0":category }
    for (i ,compare_result_key )in enumerate (compare_result_keys ,1 ):
        titles ["строка_%i"%i ]=COMPARE_RESULT_SHOWS [compare_result_key ]
    summary_json_result .append (titles )
    for device in ["gpu","cpu"]:
        for direction in ["forward","backward"]:
            method_set =["total"]if device =="cpu"else ["total","kernel"]
            for method in method_set :
                data ={
                "title":0 ,
                "row_0":"{} {} ({})".format (device .upper (),
                direction .capitalize (),
                method )
                }
                value =compare_result .get (device ,direction ,method )
                num_total_cases =value ["Total"]
                for (i ,compare_result_key )in enumerate (compare_result_keys ,
                1 ):
                    num_cases =value [compare_result_key ]
                    if num_cases >0 :
                        ratio =float (num_cases )/float (num_total_cases )
                        this_str ="{} ({:.2f}%)".format (num_cases ,
                        ratio *100 )
                    else :
                        this_str ="--"
                    data ["строка_%i"%i ]=this_str
                summary_json_result .append (data )
    return summary_json_result
def dump_json (benchmark_result_list ,
output_path =None ,
compare_framework =None ,
dump_with_parameters =None ):
    if output_path is None :
        timestamp =time .strftime ('%Y-%m-%d',time .localtime (int (time .time ())))
        output_path ="op_benchmark_summary-%s-json"%timestamp
        print ("Путь вывода не указан, используйте %s."%output_path )
    print ("-- Запись файлов JSON в %s."%output_path )
    pwd =os .getcwd ()
    save_path =os .path .join (pwd ,output_path )
    if not os .path .exists (save_path ):
        os .mkdir (save_path )
    compare_result_case_level =op_benchmark_unit .summary_compare_result (
    benchmark_result_list )
    summary_case_json =_create_summary_json (compare_result_case_level ,
    "case_level")
    with open (save_path +"/"+"case_level.json",'w')as f :
        f .write (json .dumps (summary_case_json ,ensure_ascii =False ))
    compare_result_op_level ,compare_result_dict_ops_detail =op_benchmark_unit .summary_compare_result_op_level (
    benchmark_result_list ,return_op_detail =True )
    summary_op_json =_create_summary_json (compare_result_op_level ,"op_level")
    with open (save_path +"/"+"op_level.json",'w')as f :
        f .write (json .dumps (summary_op_json ,ensure_ascii =False ))
    for op_type ,op_compare_result in sorted (
    compare_result_dict_ops_detail .items ()):
        summary_op_result =_create_summary_json (op_compare_result ,op_type )
        with open (save_path +"/"+op_type +".json",'w')as f :
            f .write (json .dumps (summary_op_result ,ensure_ascii =False ))
    json_result ={"paddle":dict (),"pytorch":dict (),"compare":dict ()}
    for device in ["cpu","gpu"]:
        json_result ["device"]=device
        for case_id in range (len (benchmark_result_list )):
            op_unit =benchmark_result_list [case_id ]
            for direction in ["forward","backward","backward_forward"]:
                result =op_unit .get (device ,direction )
                time_set =["total",
                "gpu_time"]if device =="gpu"else ["total"]
                for key in time_set :
                    compare_result =COMPARE_RESULT_SHOWS .get (
                    result ["compare"][key ],"--")
                    json_result ["compare"][direction +key ]=compare_result
                    for framework in ["paddle",compare_framework ]:
                        json_result [framework ]["case_name"]=op_unit .case_name
                        json_result [framework ][direction +key ]=result [
                        framework ][key ]
            with open (save_path +"/"+json_result [framework ]["case_name"]+
            "_"+device +".json",'w')as f :
                f .write (json .dumps (json_result ,ensure_ascii =False ))
                f .write ("\n")