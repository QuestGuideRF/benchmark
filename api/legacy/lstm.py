from common_import import *
class LstmConfig (APIConfig ):
    def __init__ (self ):
        super (LstmConfig ,self ).__init__ ("lstm")
        self .run_tf =False
    def disabled (self ):
        if not use_gpu ():
            print (
            "Warning:\n"
            "  1. lstm отключен, поскольку поддерживается только запуск на графическом процессоре, теперь на процессоре.\n"
            )
            return True
        return super (LstmConfig ,self ).disabled ()
class PDLstm (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        input =self .variable (
        name ="input",shape =config .input_shape ,dtype =config .input_dtype )
        init_h =fluid .layers .fill_constant (
        shape =config .init_h_shape ,dtype =config .init_h_dtype ,value =0.0 )
        init_c =fluid .layers .fill_constant (
        shape =config .init_c_shape ,dtype =config .init_c_dtype ,value =0.0 )
        rnn_out ,last_h ,last_c =fluid .layers .lstm (
        input =input ,
        init_h =init_h ,
        init_c =init_c ,
        max_len =config .max_len ,
        hidden_size =config .hidden_size ,
        num_layers =config .num_layers ,
        dropout_prob =0.2 ,
        is_bidirec =config .is_bidirec )
        self .feed_vars =[input ]
        self .fetch_vars =[rnn_out ]
        if config .backward :
            self .append_gradients (rnn_out ,[input ])
class TFLstm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ="input",shape =config .input_shape ,dtype =config .input_dtype )
        init_h =tf .constant (
        shape =config .init_h_shape ,
        dtype =tf .as_dtype (config .init_h_dtype ),
        value =0.0 )
        init_c =tf .constant (
        shape =config .init_c_shape ,
        dtype =tf .as_dtype (config .init_c_dtype ),
        value =0.0 )
if __name__ =='__main__':
    test_main (PDLstm (),TFLstm (),config =LstmConfig ())