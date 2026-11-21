from common_import import *
@benchmark_registry .register ("set_value")
class PaddleSetValue (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',
        shape =config .x_shape ,
        dtype =config .x_dtype ,
        stop_gradient =True )
        if config .is_tensor_value :
            value =self .variable (
            name ='value',
            shape =config .value_shape ,
            dtype =config .value_dtype ,
            stop_gradient =True )
            x [:,10 :500 :2 ]=value
            self .feed_list =[x ,value ]
            self .fetch_list =[x ]
        else :
            x [:,0 :20 ,::2 ]=10000
            self .feed_list =[x ]
            self .fetch_list =[x ]
@benchmark_registry .register ("set_value")
class TorchSetValue (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (
        name ='x',
        shape =config .x_shape ,
        dtype =config .x_dtype ,
        stop_gradient =True )
        if config .is_tensor_value :
            value =self .variable (
            name ='value',
            shape =config .value_shape ,
            dtype =config .value_dtype ,
            stop_gradient =True )
            x [:,10 :500 :2 ]=value
            self .feed_list =[x ,value ]
            self .fetch_list =[x .contiguous ()]
        else :
            x [:,0 :20 ,::2 ]=10000
            self .feed_list =[x ]
            self .fetch_list =[x .contiguous ()]