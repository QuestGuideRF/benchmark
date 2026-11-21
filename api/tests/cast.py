from common_import import *
@benchmark_registry .register ("cast")
class CastConfig (APIConfig ):
    def __init__ (self ):
        super (CastConfig ,self ).__init__ ('cast')
        self .feed_spec ={"range":[-10 ,10 ]}
@benchmark_registry .register ("cast")
class PaddleCast (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .cast (x =x ,dtype =config .dtype )
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("cast")
class TorchCast (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        dtype_map ={
        "float16":torch .float16 ,
        "float32":torch .float32 ,
        "float64":torch .float64 ,
        "int32":torch .int32 ,
        "int64":torch .int64 ,
        "bool":torch .bool
        }
        if config .dtype in dtype_map .keys ():
            result =x .to (dtype_map [config .dtype ])
        else :
            assert False ,"Пока не поддерживается!"
        self .feed_list =[x ]
        self .fetch_list =[result ]
@benchmark_registry .register ("cast")
class TFCast (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .dtypes .cast (x =x ,dtype =config .dtype )
        self .feed_list =[x ]
        self .fetch_list =[result ]