import sys
import json
import time
import json
import op_benchmark_unit
import pymysql
import paddle
import torch
import platform
import socket
from datetime import datetime
COMPARE_RESULT_SHOWS ={
"Better":"优于",
"Equal":"打平",
"Less":"差于",
"Unknown":"未知",
"Unsupport":"不支持",
"Others":"其他",
"Total":"汇总"
}
class DB (object ):
    def __init__ (self ,address ,usr ,pwd ,database ):
        self .db =pymysql .connect (
        host =address ,
        user =usr ,
        password =pwd ,
        db =database ,
        port =3306 ,
        use_unicode =True ,
        charset ="utf8"
        )
        self .cursor =self .db .cursor ()
    def timestamp (self ):
        return datetime .now ().strftime ("%Y-%m-%d %H:%M:%S")
    def init_mission (self ,card ):
        sql =(
        "insert into `jobs` (`create_time`, `update_time`, `status`, `paddle_commit`, "
        "`paddle_version`, `torch_version`, `mode`, `hostname`, `place`, `card`, `system`) "
        "values ('{}','{}','{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}' );".
        format (
        self .timestamp (),
        self .timestamp (),
        "running",
        paddle .__git_commit__ ,
        paddle .__version__ ,
        torch .__version__ ,
        "op-benchmark",
        socket .gethostname (),
        "процессор + графический процессор",
        card ,
        platform .platform (),))
        try :
            self .cursor .execute (sql )
            self .job_id =self .db .insert_id ()
            self .db .commit ()
        except Exception as e :
            print (e )
    def save (self ,benchmark_result_list ,compare_framework =None ):
        retry =3
        for i in range (retry ):
            sql ="update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format (
            self .timestamp (),"saving",self .job_id )
            try :
                self .cursor .execute (sql )
                self .db .commit ()
                break
            except Exception :
                self .db .ping (True )
                continue
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
                        json_result ["compare"][direction +
                        key ]=compare_result
                        for framework in ["paddle",compare_framework ]:
                            json_result [framework ][
                            "case_name"]=op_unit .case_name
                            json_result [framework ][direction +key ]=result [
                            framework ][key ]
                sql =(
                "insert into `cases`(`jid`, `case_name`, `result`, `create_time`) "
                "values ('{}', '{}', '{}', '{}')".format (
                self .job_id ,
                json_result [framework ]["case_name"],
                json .dumps (
                json_result ,ensure_ascii =False ),
                self .timestamp ()))
                try :
                    self .cursor .execute (sql )
                    self .db .commit ()
                except Exception as e :
                    print (e )
        sql ="update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format (
        self .timestamp (),"done",self .job_id )
        try :
            self .cursor .execute (sql )
            self .db .commit ()
        except Exception as e :
            print (e )
    def set_place (self ,card =None ):
        if paddle .is_compiled_with_cuda ()and torch .cuda .is_available (
        )is True :
            if card is None :
                paddle .set_device ("gpu:0")
                torch .device (0 )
                return '0'
            else :
                paddle .set_device ("графический процессор: {}".format (card ))
                torch .device (card )
                return card
        else :
            raise EnvironmentError
    def error (self ):
        retry =3
        for i in range (retry ):
            sql ="update `jobs` set `update_time`='{}', `status`='{}' where id='{}';".format (
            self .timestamp (),"error",self .job_id )
            try :
                self .cursor .execute (sql )
                self .db .commit ()
                break
            except Exception as e :
                self .db .ping (True )
                print (e )
                continue
    def write_database (self ,
    benchmark_result_list ,
    compare_framework =None ,
    card =None ):
        try :
            card =self .set_place (card )
            self .init_mission (card )
            self .save (benchmark_result_list ,compare_framework )
        except Exception as e :
            print (e )
            self .error ()