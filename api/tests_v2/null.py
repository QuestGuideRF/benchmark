from common_import import *
class PDNull (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =[1 ],dtype ="float32")
        self .feed_vars =[x ]
        self .fetch_vars =None
class TFNull (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =[1 ],dtype ="float32")
        result =tf .identity (x )
        self .feed_list =[x ]
        self .fetch_list =[result ]
if __name__ =='__main__':
    test_main_without_json (PDNull (),TFNull (),config =APIConfig ("null"))