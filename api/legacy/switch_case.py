from common_import import *
class PDSwitchCase (PaddleAPIBenchmarkBase ):
    def build_program (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        zero =fluid .layers .zeros (
        shape =config .input_shape ,dtype =config .input_dtype )
        def f1 ():
            return fluid .layers .elementwise_add (x =x ,y =y )
        def f2 ():
            return fluid .layers .elementwise_sub (x =x ,y =y )
        def f3 ():
            return fluid .layers .elementwise_mul (x =x ,y =y )
        result =fluid .layers .switch_case (
        branch_index =zero ,branch_fns ={0 :f1 ,
        1 :f2 },default =f3 )
        self .feed_vars =[x ,y ]
        self .fetch_vars =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
class TFSwitchCase (TensorflowAPIBenchmarkBase ):
    def build_graph (self ,config ):
        x =self .variable (name ='x',shape =config .x_shape ,dtype =config .x_dtype )
        y =self .variable (name ='y',shape =config .y_shape ,dtype =config .y_dtype )
        zero =tf .zeros (shape =config .input_shape ,dtype =config .input_dtype )
        def f1 ():
            return tf .add (x ,y )
        def f2 ():
            return tf .subtract (x ,y )
        def f3 ():
            return tf .multiply (x ,y )
        result =tf .switch_case (
        branch_index =tf .reshape (zero ,[]),
        branch_fns ={0 :f1 ,
        1 :f2 },
        default =f3 )
        self .feed_list =[x ,y ]
        self .fetch_list =[result ]
        if config .backward :
            self .append_gradients (result ,[x ,y ])
if __name__ =='__main__':
    test_main (PDSwitchCase (),TFSwitchCase (),config =APIConfig ('switch_case'))