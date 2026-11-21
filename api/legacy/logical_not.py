from common_import import *
class PDLogicalNot (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =fluid .layers .logical_not (x =data )
        self .feed_vars =[data ]
        self .fetch_vars =[result ]
class TFLogicalNot (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='data',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .math .logical_not (x =data )
        self .feed_list =[data ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDLogicalNot (),TFLogicalNot (),config =APIConfig ("logical_not"))