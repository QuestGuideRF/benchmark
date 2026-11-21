from common_import import *
class PDLinspace (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        result =paddle .linspace (start =config .start ,stop =config .stop ,num =config .num ,dtype =config .dtype )
        self .feed_vars =[]
        self .fetch_vars =[result ]
class TFLinspace (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .linspace (start =config .start ,stop =config .stop ,num =config .num )
        self .feed_list =[]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDLinspace (),TFLinspace (),config =APIConfig ('linspace'))