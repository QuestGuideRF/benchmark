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
        name ="input",shape =config .inputs_shape ,dtype =config .inputs_dtype )
        init_h =paddle .full (
        shape =config .inital_states_shape ,
        dtype =config .inital_states_dtype ,
        fill_value =0.0 )
        init_c =paddle .full (
        shape =config .inital_states_shape ,
        dtype =config .inital_states_dtype ,
        fill_value =0.0 )
        rnn =paddle .nn .LSTM (
        input_size =config .inputs_shape [-1 ],
        hidden_size =config .hidden_size ,
        num_layers =config .num_layers ,
        dropout =0.0 ,
        direction =config .direction )
        rnn_out ,(last_h ,last_c )=rnn (input ,(init_h ,init_c ))
        self .feed_vars =[input ]
        self .fetch_vars =[rnn_out ]
class TFLstm (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        input =self .variable (
        name ="input",shape =config .inputs_shape ,dtype =config .inputs_dtype )
        init_h =tf .constant (
        shape =config .inital_states_shape ,
        dtype =tf .as_dtype (config .inital_states_dtype ),
        value =0.0 )
        init_c =tf .constant (
        shape =config .inital_states_shape ,
        dtype =tf .as_dtype (config .inital_states_dtype ),
        value =0.0 )
if __name__ =='__main__':
    test_main (PDLstm (),TFLstm (),config =LstmConfig ())