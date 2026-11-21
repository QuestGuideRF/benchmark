import os ,sys
import numpy as np
try :
    import paddle
except ImportError :
    sys .stderr .write (
    "Невозможно импортировать весло или Paddle.fluid, возможно, весло не установлено.\n"
    )
try :
    import tensorflow as tf
except ImportError :
    sys .stderr .write (
    "Невозможно импортировать tensorflow, возможно, tensorflow не установлен.\n")
package_path =os .path .dirname (os .path .dirname (os .path .abspath (__file__ )))
sys .path .append (package_path )
from common .paddle_op_benchmark import PaddleAPIBenchmarkBase
from common .tensorflow_op_benchmark import TensorflowAPIBenchmarkBase
from common .api_param import APIConfig
from common .main import test_main ,test_main_without_json
def use_gpu ():
    return os .environ .get ("CUDA_VISIBLE_DEVICES",None )!=""
def unsqueeze_short (short ,long ):
    short_extend_zeros =np .zeros ([len (long )],dtype =np .int32 ).tolist ()
    start =0
    for value in short :
        for i in range (start ,len (long )):
            if long [i ]==value :
                short_extend_zeros [i ]=value
                start =i
                break
    short_extend =[]
    first_nonzero_idx =-1
    for i in range (len (short_extend_zeros )):
        if first_nonzero_idx ==-1 and short_extend_zeros [i ]!=0 :
            first_nonzero_idx =i
        if first_nonzero_idx >-1 :
            if short_extend_zeros [i ]==0 :
                short_extend .append (1 )
            else :
                short_extend .append (short_extend_zeros [i ])
    return short_extend