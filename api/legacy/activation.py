from common_import import *
class ActivationConfig (APIConfig ):
    def __init__ (self ):
        super (ActivationConfig ,self ).__init__ ('activation')
        self .api_name ='cos'
        self .api_list ={
        'cos':'cos',
        'exp':'exp',
        'floor':'floor',
        'sin':'sin',
        'square':'square',
        'tanh':'tanh',
        'sigmoid':'sigmoid',
        'sqrt':'sqrt'
        }
class PDActivation (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .fluid_layers (config .api_name ,x =x )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
class TFActivation (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =self .layers (config .api_name ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ])
if __name__ =='__main__':
    test_main (PDActivation (),TFActivation (),config =ActivationConfig ())