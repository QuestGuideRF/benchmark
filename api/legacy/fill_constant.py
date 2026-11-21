from common_import import *
class PDFillConstant (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        result =fluid .layers .fill_constant (
        shape =config .shape ,dtype =config .dtype ,value =config .value )
        self .feed_vars =[]
        self .fetch_vars =[result ]
class TFFillConstant (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        result =tf .constant (
        shape =config .shape ,
        dtype =tf .as_dtype (config .dtype ),
        value =config .value )
        self .feed_list =[]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main (
    PDFillConstant (),TFFillConstant (),config =APIConfig ("fill_constant"))