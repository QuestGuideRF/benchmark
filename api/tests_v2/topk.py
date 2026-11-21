from common_import import *
class PDTopK (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =paddle .topk (x =data ,k =config .k )
        self .feed_vars =[data ]
        self .fetch_vars =[value ,indices ]
        if config .backward :
            self .append_gradients ([value ],[data ])
        print (self .fetch_vars )
class TFTopK (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        data =self .variable (
        name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        value ,indices =tf .math .top_k (input =data ,k =config .k )
        self .feed_list =[data ]
        self .fetch_list =[value ,indices ]
        if config .backward :
            self .append_gradients ([value ],[data ])
        print (self .fetch_list )
if __name__ =='__main__':
    test_main (PDTopK (),TFTopK (),config =APIConfig ("topk"))