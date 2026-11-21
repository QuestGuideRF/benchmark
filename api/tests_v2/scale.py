from common_import import *
class PDScale (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .scale (
        x =x ,scale =config .scale ,bias =0.0 ,bias_after_scale =True ,act =None )
        self .feed_vars =[x ]
        self .fetch_vars =[result ]
class TFScale (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .scalar_mul (scalar =config .scale ,x =x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDScale (),TFScale (),config =APIConfig ("scale"))