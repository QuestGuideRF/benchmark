from common_import import *
class SoftmaxConfig (APIConfig ):
    def __init__ (self ):
        super (SoftmaxConfig ,self ).__init__ ("softmax")
        self .feed_spec ={"range":[-1 ,1 ]}
class PDSoftmax (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =paddle .nn .functional .softmax (x =input ,axis =config .axis )
        self .feed_vars =[input ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
class TFSoftmax (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ='input',shape =config .x_shape ,dtype =config .x_dtype )
        result =tf .nn .softmax (logits =input ,axis =config .axis )
        self .feed_list =[input ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[input ])
if __name__ =='__main__':
    test_main (PDSoftmax (),TFSoftmax (),config =SoftmaxConfig ())