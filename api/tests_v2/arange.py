from common_import import *
class PDArange (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        result =paddle .arange (
        start =config .start ,
        end =config .end ,
        step =config .step ,
        dtype =config .dtype )
        self .feed_vars =[]
        self .fetch_vars =[result ]
class TFArange (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .range (
        start =config .start ,
        limit =config .end ,
        delta =config .step ,
        dtype =config .dtype )
        self .feed_list =[]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDArange (),TFArange (),config =APIConfig ("arange"))