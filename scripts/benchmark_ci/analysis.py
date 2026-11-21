import argparse
import ast
import json
import os
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
    parser .add_argument (
    '--loss_threshold',
    type =float ,
    default =0.3 ,
    help ='порог потерь')
    parser .add_argument (
    '--paddle_dev',
    type =ast .literal_eval ,
    default =False ,
    help ='генерируется ли стандартное значение с помощью весла')
    args =parser .parse_args ()
    return args
def traverse_logs (log_path ):
    file_list =[]
    for dirpath ,dirnames ,filenames in os .walk (log_path ):
        for filename in filenames :
            file_path =os .path .join (dirpath ,filename )
            file_list .append (file_path )
    print ('file_list:{}'.format (file_list ))
    return file_list
def analysis (file_path ):
    with open (file_path ,'r')as f :
        lines =f .readlines ()
        try :
            job_info =json .loads (lines [-1 ])
        except Exception as e :
            print ("ошибка анализа файла {}.".format (file ))
        model =json .dumps (job_info ["model_name"])
        model =model .strip ('"')
        fail_flag =json .dumps (job_info ["JOB_FAIL_FLAG"])
        result =json .dumps (job_info ["FINAL_RESULT"])
        loss_result =json .dumps (job_info ["LOSS_RESULT"])
        loss_result =loss_result .strip ('"')
        loss_result =loss_result .strip (',')
    return model ,fail_flag ,result ,loss_result
def compare ():
    file_list =traverse_logs (args .log_path )
    errorcode =0
    has_file =os .path .exists ('errorcode.txt')
    if has_file :
        with open ('errorcode.txt','r')as f :
            for line in f :
                errorcode =int (line .strip ('\n'))
                print ('код ошибки: {}'.format (errorcode ))
    for file in file_list :
        model ,fail_flag ,result ,loss_result =analysis (file )
        if int (fail_flag )==1 :
            command ='sed -i "s/success/fail/g" log.txt'
            errorcode =errorcode |4
            os .system (command )
            print ("{} не удалось выполнить запуск!".format (model ))
        else :
            print ("ips_result:{}".format (result ))
            print ("loss_result:{}".format (loss_result ))
            print ("модель:{}".format (model ))
            standard_record =os .path .join (args .standard_path ,model +'.txt')
            loss_standard_record =os .path .join (args .standard_path ,model +'_loss'+'.txt')
            threshold_record =os .path .join (args .standard_path ,model +'_threshold'+'.txt')
            with open (threshold_record ,'r')as f :
                for line in f :
                    threshold =float (line .strip ('\n'))
                    print ("ips_threshold:{}".format (threshold ))
            with open (standard_record ,'r')as f :
                for line in f :
                    standard_result =float (line .strip ('\n'))
                    print ("ips_standard_result:{}".format (standard_result ))
                    ranges =round ((float (result )-standard_result )/standard_result ,4 )
                    if ranges >=threshold :
                        if args .paddle_dev :
                            command ='sed -i "s/success/fail/g" log.txt'
                            errorcode =errorcode |2
                            os .system (command )
                            print ("{}, НЕУДАЧА".format (model ))
                            print (
                            "Производительность модели {} увеличена с {} до {},"
                            "({}-{})/{} равно {},"
                            "что больше порога {},"
                            "пожалуйста, свяжитесь с xiege01 или heya02, чтобы изменить стандартное значение"
                            .format (model ,standard_result ,result ,result ,standard_result ,standard_result ,ranges ,threshold ))
                        else :
                            print ("Производительность модели {} увеличена с {} до {},"
                            "повторный запуск в весле".format (model ,standard_result ,result ))
                            f =open ('rerun_model.txt','a')
                            f .writelines (model +'\n')
                    elif ranges <=-threshold :
                        if args .paddle_dev :
                            command ='sed -i "s/success/fail/g" log.txt'
                            errorcode =errorcode |2
                            os .system (command )
                            print ("{}, НЕУДАЧА".format (model ))
                            print ("Производительность модели {} снижена с {} до {},"
                            "({}-{})/{} равно {},"
                            "что больше порога {}."
                            .format (model ,standard_result ,result ,standard_result ,result ,standard_result ,-ranges ,threshold ))
                        else :
                            print ("Производительность модели {} снижена с {} до {},"
                            "повторный запуск в весле".format (model ,standard_result ,result ))
                            f =open ('rerun_model.txt','a')
                            f .writelines (model +'\n')
                    else :
                        print ("{}, Performance_test, УСПЕХ".format (model ))
            with open (loss_standard_record ,'r')as f :
                for line in f :
                    loss_standard_result =float (line .strip ('\n'))
                    loss_ranges =round ((float (loss_result )-loss_standard_result )/loss_standard_result ,4 )
                    print ("loss_result:{}".format (loss_result ))
                    print ("loss_standard_result:{}".format (loss_standard_result ))
                    if loss_ranges >=args .loss_threshold :
                        if args .paddle_dev :
                            pass
                        else :
                            command ='sed -i "s/success/fail/g" log.txt'
                            errorcode =errorcode |1
                            os .system (command )
                            print ("{}, НЕУДАЧА".format (model ))
                            print ("Окончательный урон модели {} увеличен с {} до {},"
                            " что больше порога"
                            .format (model ,loss_standard_result ,loss_result ))
                    else :
                        print ("{}, Precision_test, УСПЕХ".format (model ))
    f =open ('errorcode.txt','w')
    f .writelines (str (errorcode ))
    f .close ()
if __name__ =='__main__':
    args =parse_args ()
    compare ()