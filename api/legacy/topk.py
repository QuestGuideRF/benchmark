from common_import import *
class PDTopK (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        value ,indices =fluid .layers .topk (input =data ,k =config .k )
        self .feed_vars =[data ]
        self .fetch_vars =[value ,indices ]
        if config .backward :
            self .append_gradients ([value ],[data ])
class TFTopK (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        value ,indices =tf .math .top_k (input =data ,k =config .k )
        self .feed_list =[data ]
        self .fetch_list =[value ,indices ]
        if config .backward :
            self .append_gradients ([value ],[data ])
if __name__ =='__main__':
    test_main (PDTopK (),TFTopK (),config =APIConfig ("topk"))