import os
import json
import copy
import numpy as np
from operator import attrgetter
def is_string (value ):
    return isinstance (value ,str )
def parse_string (value ):
    import six
    if six .PY3 :
        return value
    else :
        return value .encode ("utf-8")if isinstance (value ,unicode )else value
def parse_float (value ):
    if isinstance (value ,float ):
        return value
    else :
        return float (parse_string (value ))
def parse_int (value ):
    if isinstance (value ,int ):
        return value
    else :
        return int (parse_string (value ))
def parse_list (value_str ,sub_dtype ="int"):
    value_str =parse_string (value_str )
    if sub_dtype in ["int","int64"]:
        try :
            if value_str !="[]":
                value_str_list =value_str .replace ("L","").replace (
                "[","").replace ("]","").split (',')
                value_list =[]
                for item in value_str_list :
                    value_list .append (int (item ))
                return value_list
            else :
                return []
        except Exception as e :
            assert False ,"Анализ {} не удался: {}".format (value_str ,e )
    else :
        raise ValueError ("Не поддерживать список синтаксического анализа типов данных, отличных от int.")
def parse_tuple (value_str ,sub_dtype ="int"):
    value_str =parse_string (value_str )
    if sub_dtype in ["int","int64"]:
        try :
            if value_str !="()":
                value_str_list =value_str .replace ("L","").replace (
                "(","").replace (")","").split (',')
                value_list =[]
                for item in value_str_list :
                    value_list .append (int (item ))
                return value_list
            else :
                return []
        except Exception as e :
            assert False ,"Анализ {} не удался: {}".format (value_str ,e )
    else :
        raise ValueError ("Не поддерживать список синтаксического анализа типов данных, отличных от int.")
class BaseParamInfo (object ):
    def __init__ (self ,name ,type ,value ):
        self .name =self ._encode_item (name )
        self .type =self ._encode_item (type )
        self .value =self ._translate_value (self ._encode_item (value ))
    def _encode_item (self ,item ):
        if isinstance (item ,list ):
            item_str =[]
            for ele in item :
                item_str .append (parse_string (ele ))
            return item_str
        else :
            return parse_string (item )
    def to_string (self ):
        return self .name +' ('+self .type +'): '+str (self .value )
    def _translate_value (self ,value_str ):
        if self .type in ["float","float32","float64"]:
            return float (value_str )
        elif self .type in ["int","int32","int64","long"]:
            return int (value_str )
        elif self .type =="bool":
            return eval (value_str )
        elif self .type in ["string","str"]:
            return None if value_str =="None"else value_str
        elif self .type =="list":
            return parse_list (value_str ,sub_dtype ="int")
        elif self .type =="tuple":
            return parse_tuple (value_str ,sub_dtype ="int")
        else :
            raise ValueError ("Неподдерживаемый тип \"%s\" для %s, значение — %s."%
            (self .type ,self .name ,value_str ))
class VarParamInfo (BaseParamInfo ):
    def __init__ (self ,name ,type ,dtype ,shape ,lod_level =0 ):
        self .name =self ._encode_item (name )
        self .type =self ._encode_item (type )
        self .dtype =self ._encode_item (dtype )
        shape_str =self ._encode_item (shape )
        if is_string (shape_str ):
            self .shape =parse_list (shape_str )
        else :
            assert isinstance (shape_str ,list )
            self .shape =shape_str
        self .lod_level =self ._encode_item (lod_level )
    def _is_same (self ,values ):
        value_0 =values [0 ]
        for i in range (len (values )):
            if value_0 !=values [i ]:
                return False
        return True
    def to_string (self ):
        if self .type =="Variable":
            return self .name +" (Переменная) – тип d: "+str (
            self .dtype )+", форма: "+str (self .shape )
        elif self .type =="список<Переменная>":
            str_list ="%s (список<Переменная>[%d]) – "%(self .name ,
            len (self .dtype ))
            if self ._is_same (self .dtype )and self ._is_same (self .shape ):
                params_len =1
            else :
                params_len =len (self .dtype )
            for i in range (params_len ):
                str_list =str_list +"dtype: "+str (self .dtype [
                i ])+", форма: "+str (self .shape [i ])+"; "
            return str_list
class APIConfig (object ):
    def __init__ (self ,op_type ,params =None ):
        self .__name =op_type
        self .__framework ="paddle"
        self .__run_tf =True
        self .__run_torch =True
        self .api_name =self .name
        self .paddle_api =None
        self .torch_api =None
        self .params =params
        self .variable_list =None
        self .params_list =None
        self .backward =False
        self .feed_spec =None
        self .alias_name =None
    @classmethod
    def get_all_subclasses (self ):
        all_subclasses =[]
        for subclass in self .__subclasses__ ():
            all_subclasses .append (subclass )
            all_subclasses .extend (subclass .get_all_subclasses ())
        return all_subclasses
    def alias_filename (self ,filename ):
        if hasattr (self ,"alias_name")and self .alias_name is not None :
            dirname =os .path .dirname (filename )
            basename =self .alias_name +os .path .splitext (filename )[-1 ]
            return os .path .join (dirname ,basename )
        return filename
    @property
    def name (self ):
        return self .__name
    @property
    def framework (self ):
        return self .__framework
    @property
    def run_tf (self ):
        return self .__run_tf
    @run_tf .setter
    def run_tf (self ,value ):
        self .__run_tf =value
    @property
    def run_torch (self ):
        return self .__run_torch
    @run_torch .setter
    def run_torch (self ,value ):
        self .__run_torch =value
    def compute_dtype (self ):
        dtype =None
        for name ,value in vars (self ).items ():
            if name .endswith ("_dtype"):
                if value =="float16":
                    dtype ="float16"
                elif dtype is not None :
                    dtype =value
        return dtype
    def disabled (self ):
        use_gpu =os .environ .get ("CUDA_VISIBLE_DEVICES",None )!=""
        if not use_gpu and self .compute_dtype ()=="float16":
            print (
            "Warning:\n"
            "  1. Эта конфигурация отключена, поскольку float16 не поддерживается для %s на ЦП.\n"
            %(self .api_name ))
            return True
        return False
    def convert_to_fp16 (self ):
        def _is_floating_point_type (dtype ):
            if dtype in ["float","float32","double","float64"]:
                return True
            return False
        for var in self .variable_list :
            if var .type =="Variable"and _is_floating_point_type (var .dtype ):
                var .dtype ="float16"
                setattr (self ,var .name +"_dtype",var .dtype )
            elif var .type =="список<Переменная>":
                for i in range (len (var .dtype )):
                    if _is_floating_point_type (var .dtype [i ]):
                        var .dtype [i ]="float16"
                setattr (self ,var .name +"_dtype",var .dtype )
    def init_from_json (self ,filename ,config_id =0 ,unknown_dim =16 ):
        filename =self .alias_filename (filename )
        print ("---- Инициализируйте APIConfig из %s, config_id = %d.\n"%
        (filename ,config_id ))
        with open (filename ,'r')as f :
            data =json .load (f )
            op =data [config_id ]["op"]
            assert op ==self .name or op ==self .alias_name ,"The op type (%s) in json file is different from the name (%s) and the alias name (%s). "  "Имя файла: %s, config_id: %d."%(
            op ,self .name ,self .alias_name ,filename ,config_id )
            if data [config_id ].get ("paddle_api",None )is not None :
                self .paddle_api =parse_string (data [config_id ]["paddle_api"])
            if data [config_id ].get ("torch_api",None )is not None :
                self .torch_api =parse_string (data [config_id ]["torch_api"])
            self .params =data [config_id ]["param_info"]
            if data [config_id ].get ("atol",None )is not None :
                self .atol =parse_float (data [config_id ]["atol"])
            if data [config_id ].get ("repeat",None )is not None :
                self .repeat =parse_int (data [config_id ]["repeat"])
        self ._parse_params ()
        for param in self .params_list :
            setattr (self ,param .name ,param .value )
        for var in self .variable_list :
            for i in range (len (var .shape )):
                if var .shape [i ]==-1 :
                    var .shape [i ]=unknown_dim
                var_temp =var .shape [i ]
                if type (var_temp )==list :
                    for j in range (len (var_temp )):
                        if var_temp [j ]==-1 :
                            var_temp [j ]=unknown_dim
            setattr (self ,var .name +'_shape',var .shape )
            setattr (self ,var .name +'_dtype',var .dtype )
        if not hasattr (self ,"atol"):
            self .atol =1e-3 if self .compute_dtype ()=="float16"else 1e-6
        return self
    def to_tensorflow (self ):
        assert self .__framework =="paddle"
        tf_config =copy .deepcopy (self )
        tf_config .__framework ="tensorflow"
        if hasattr (self ,"api_list"):
            tf_config .api_name =self .api_list [self .api_name ]
        return tf_config
    def to_pytorch (self ):
        assert self .__framework =="paddle"
        torch_config =copy .deepcopy (self )
        torch_config .__framework ="pytorch"
        if hasattr (self ,"api_list"):
            torch_config .api_name =self .api_list [self .api_name ]
        return torch_config
    def to_string (self ):
        if self .params_list is None and self .variable_list is None :
            self ._parse_params ()
        if self .variable_list is None and self .params_list is None :
            return "None"
        params_str =""
        self .variable_list =sorted (self .variable_list ,key =attrgetter ('name'))
        self .params_list =sorted (self .params_list ,key =attrgetter ('name'))
        for var in self .variable_list :
            params_str =params_str +var .to_string ()+"\n"
        for attr in self .params_list :
            params_str =params_str +attr .to_string ()+"\n"
        return params_str
    def __str__ (self ):
        exclude_attrs =[
        '_APIConfig__name','_APIConfig__framework','params','api_name',
        'api_list','variable_list','params_list','backward',
        'feed_spec','alias_name'
        ]
        if self .framework !="paddle":
            exclude_attrs .append ("_APIConfig__run_torch")
            exclude_attrs .append ("_APIConfig__run_tf")
        prefix =""
        debug_str =('[%s][%s] %s {\n')%(self .framework ,self .name ,
        self .api_name )
        for name ,value in vars (self ).items ():
            if name not in exclude_attrs :
                if isinstance (value ,np .ndarray ):
                    debug_str =debug_str +(
                    '  %s: np.ndarray(shape=%s, dtype=%s)\n')%(
                    name ,str (value .shape ),value .dtype )
                else :
                    debug_str =debug_str +('  %s%s: %s\n')%(
                    prefix ,name .replace ("_APIConfig__",""),value )
        debug_str =debug_str +prefix +'}'
        return debug_str
    def _parse_params (self ):
        self .variable_list =[]
        self .params_list =[]
        if self .params is None :
            self .variable_list =None
            self .params_list =None
        else :
            for name ,value in self .params .items ():
                assert value .get ("type",None )is not None
                if value ["type"]=="Variable":
                    info =VarParamInfo (name ,value ["type"],value ["dtype"],
                    value ["shape"])
                    self .variable_list .append (info )
                elif value ["type"]=="список<Переменная>":
                    dtype_list =[]
                    shape_list =[]
                    for key ,var in value .items ():
                        if key !="type":
                            assert var ["type"]=="Variable"
                            dtype_list .append (var ["dtype"])
                            shape_list .append (parse_list (var ["shape"]))
                    info =VarParamInfo (name ,value ["type"],dtype_list ,
                    shape_list )
                    self .variable_list .append (info )
                else :
                    info =BaseParamInfo (name ,value ["type"],value ["value"])
                    self .params_list .append (info )