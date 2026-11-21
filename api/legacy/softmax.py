from common_import import *
class PDSoftmax (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =fluid .layers .softmax (
        input =input ,use_cudnn =config .use_cudnn ,axis =config .axis )
        self .feed_vars =[input ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
class TFSoftmax (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .input_shape ,dtype =config .input_dtype )
        result =tf .nn .softmax (logits =input ,axis =config .axis )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
if __name__ =='__main__':
    test_main (PDSoftmax (),TFSoftmax (),config =APIConfig ("softmax"))