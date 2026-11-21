from common_import import *
class PDFeed (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =[1600000 ],dtype ="float32")
        self .feed_vars =[x ]
        self .fetch_vars =None
class TFFeed (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =[1600000 ],dtype ="float32")
        result =tf .identity (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main_without_json (PDFeed (),TFFeed (),config =APIConfig ("feed"))