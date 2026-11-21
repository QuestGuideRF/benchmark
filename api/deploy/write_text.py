import time
import op_benchmark_unit
import string
COMPARE_RESULT_SHOWS ={
"Better":"优于",
"Equal":"打平",
"Less":"差于",
"Unknown":"未知",
"Unsupport":"不支持",
"Others":"其他",
"Total":"汇总"
}
def _write_summary (compare_result ,category ,f ):
    compare_result_keys =compare_result .compare_result_keys
    f .write (category .ljust (25 ))
    for col in range (len (compare_result_keys )):
        title =COMPARE_RESULT_SHOWS [compare_result_keys [col ]]
        f .write (title .ljust (25 ))
    f .write ("\n")
    for device in ["gpu","cpu"]:
        for direction in ["forward","backward"]:
            method_set =["total"]if device =="cpu"else ["total","kernel"]
            for method in method_set :
                category =device .upper ()+" "+string .capwords (
                direction )+" ("+method +")"
                f .write (category .ljust (25 ))
                value =compare_result .get (device ,direction ,method )
                num_total_cases =value ["Total"]
                for col in range (len (compare_result_keys )):
                    compare_result_key =compare_result_keys [col ]
                    num_cases =value [compare_result_key ]
                    if num_cases >0 :
                        ratio =float (num_cases )/float (num_total_cases )
                        ratio_str ="%.2f"%(ratio *100 )
                        this_str ="{} ({}%)".format (num_cases ,ratio_str )
                    else :
                        this_str ="--"
                    f .write (this_str .ljust (25 ))
                f .write ("\n")
def dump_text (benchmark_result_list ,
output_path =None ,
compare_framework =None ,
dump_with_parameters =None ):
    if output_path is None :
        timestamp =time .strftime ('%Y-%m-%d',time .localtime (int (time .time ())))
        output_path ="op_benchmark_summary-%s.txt"%timestamp
        print ("Путь вывода не указан, используйте %s."%output_path )
    print ("-- Напишите %s."%output_path )
    title_total ="%s%s%s"%("весло (всего)".ljust (20 ),
    (compare_framework +"(общий)").ljust (20 ),
    "status".ljust (10 ))
    title_kernel ="%s%s%s"%("весло (ядро)".ljust (20 ),
    compare_framework +"(ядро)".ljust (20 ),
    "status".ljust (10 ))
    title_else ="%s%s"%("accuracy".ljust (10 ),"paramaters")
    gpu_title ="%s%s%s%s%s\n"%("case_id".ljust (12 ),"case_name".ljust (40 ),
    title_total ,title_kernel ,title_else )
    cpu_title ="%s%s%s%s\n"%("case_id".ljust (12 ),"case_name".ljust (40 ),
    title_total ,title_else )
    output_str_list ={
    "gpu_forward":"",
    "gpu_backward":"",
    "cpu_forward":"",
    "cpu_backward":""
    }
    for case_id in range (len (benchmark_result_list )):
        op_unit =benchmark_result_list [case_id ]
        for device in ["gpu","cpu"]:
            for direction in ["forward","backward"]:
                key =device +"_"+direction
                case_line ="%s%s"%(
                str (case_id +1 ).ljust (8 ),
                op_unit .to_string (device ,direction ,dump_with_parameters ))
                output_str_list [key ]+=case_line +"\n"
    with open (output_path ,'w')as f :
        compare_result_case_level =op_benchmark_unit .summary_compare_result (
        benchmark_result_list )
        f .writelines ("="*75 +" Сводная информация по уровню дела "+"="*84 +"\n")
        _write_summary (compare_result_case_level ,"case_level",f )
        compare_result_op_level ,compare_result_dict_ops_detail =op_benchmark_unit .summary_compare_result_op_level (
        benchmark_result_list ,return_op_detail =True )
        f .writelines ("="*75 +" Сводка уровня ОП "+"="*86 +"\n")
        _write_summary (compare_result_op_level ,"op_level",f )
        f .writelines ("="*75 +" Сводная информация для каждого ОП "+"="*75 +
        "\n")
        for op_type ,op_compare_result in sorted (
        compare_result_dict_ops_detail .items ()):
            _write_summary (op_compare_result ,op_type ,f )
        for direction in ["Forward","Backward"]:
            f .writelines ("="*75 +" %s Работает на графическом процессоре "%direction +"="*
            80 +"\n")
            f .writelines (gpu_title )
            f .writelines (output_str_list ["gpu_"+direction .lower ()])
            f .writelines ("\n")
        for direction in ["Forward","Backward"]:
            f .writelines ("="*75 +" %s Работает на процессоре "%direction +"="*
            80 +"\n")
            f .writelines (cpu_title )
            f .writelines (output_str_list ["cpu_"+direction .lower ()])
            f .writelines ("\n")