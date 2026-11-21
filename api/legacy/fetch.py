from common_import import *
class PDFetch (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ="x",shape =[1600000 ],dtype ="float32")
        self .feed_vars =[x ]
        self .fetch_vars =[x ]
class TFFetch (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ="x",shape =[1600000 ],dtype ="float32")
        self .feed_list =[x ]
        self .fetch_list =[x ]
if __name__ =='__main__':
    test_main_without_json (PDFetch (),TFFetch (),config =APIConfig ("fetch"))