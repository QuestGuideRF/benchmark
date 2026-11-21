from __future__ import division
from common_import import *
@benchmark_registry .register ("expand_as")
class PaddleExpandAs (PaddleOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        y .stop_gradient =True
        result =paddle .expand_as (x =x ,y =y )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("expand_as")
class TorchExpandAs (PytorchOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        y .requires_grad =False
        result =x .expand_as (y ).contiguous ()
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
@benchmark_registry .register ("expand_as")
class TFExpandAs (TensorflowOpBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        assert len (config .x_shape )==len (config .y_shape )
        expand_times =[]
        for i in range (len (config .x_shape )):
            if config .x_shape [i ]==-1 :
                expand_times .append (1 )
            else :
                expand_times .append (config .y_shape [i ]//config .x_shape [i ])
        result =tf .tile (input =x ,multiples =expand_times )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])