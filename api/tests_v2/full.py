from common_import import *
class PDFull (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        result =paddle .full (
        shape =config .shape ,
        dtype =config .dtype ,
        fill_value =config .fill_value )
        self .feed_vars =[]
        self .fetch_vars =[result ]
class TFFull (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .constant (
        shape =config .shape ,
        dtype =tf .as_dtype (config .dtype ),
        value =config .fill_value )
        self .feed_list =[]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (PDFull (),TFFull (),config =APIConfig ("full"))