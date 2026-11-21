from __future__ import division
from common_import import *
class PDExpandAs (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =paddle .full (
        shape =config .y_shape ,dtype =config .y_dtype ,fill_value =0.0 )
        y .stop_gradient =True
        result =paddle .expand_as (x =x ,y =y )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFExpandAs (TensorflowAPIBenchmarkBase ):
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
if __name__ =='__main__':
    test_main (PDExpandAs (),TFExpandAs (),config =APIConfig ("expand_as"))