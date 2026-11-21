from common_import import *
class ReduceConfig (APIConfig ):
    def __init__ (self ):
        super (ReduceConfig ,self ).__init__ ('reduce')
        self .feed_spec ={"range":[-1 ,1 ]}
        self .api_name ='mean'
        self .api_list ={
        'max':'reduce_max',
        'mean':'reduce_mean',
        'min':'reduce_min',
        'sum':'reduce_sum',
        'prod':'reduce_prod'
        }
class PDReduce (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,x =x ,axis =config .axis ,keepdim =config .keepdim )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFReduce (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (
        config .api_name ,
        input_tensor =x ,
        axis =config .axis ,
        keepdims =config .keepdim )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDReduce (),TFReduce (),config =ReduceConfig ())