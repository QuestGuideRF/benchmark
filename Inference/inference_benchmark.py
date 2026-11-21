import argparse
import logging
import model
logging .basicConfig (
format ="%(asctime)s - %(имя)s - %(имя уровня)s - %(сообщение)s")
logger =logging .getLogger (__name__ )
logger .setLevel (logging .DEBUG )
def main ():
    models ={"resnet":model .ResNet50Benchmark ,
    "mobilenet":model .ResNet50Benchmark ,
    "vgg":model .ResNet50Benchmark ,
    "googlenet":model .ResNet50Benchmark ,
    "shufflenet":model .ResNet50Benchmark ,
    "MobileNet_SSD":model .ResNet50Benchmark ,
    "deeplab":model .ResNet50Benchmark ,
    "rcnn":model .RcnnBenchmark ,
    "yolo":model .YoloBenchmark ,
    "transformer":model .TransformerBenchmark ,
    "bert":model .BertBenchmark }
    args =parse_args ()
    model =models .get (args .model )()
    model .set_config (
    use_gpu =args .device =='gpu',
    model_dir =args .model_dir ,
    model_filename =args .model_filename ,
    params_filename =args .params_filename ,
    use_tensorrt =args .use_tensorrt ,
    use_anakin =args .use_anakin ,
    model_precision =args .model_precision )
    tensor =model .load_data (args .filename )
    warmup =args .warmup
    repeat =args .repeat
    model .run (tensor ,warmup ,repeat )
def parse_args (prog =None ):
    parser =argparse .ArgumentParser (prog =prog )
    parser .add_argument ("--model",type =str )
    parser .add_argument ("--model_dir",type =str ,help ="Модельный директор")
    parser .add_argument ("--model_filename",type =str ,help ="имя файла модели")
    parser .add_argument ("--params_filename",type =str ,help ="имя файла параметров")
    parser .add_argument ("--device",choices =["cpu","gpu"])
    parser .add_argument ("--use_tensorrt",action ='store_true',help ='Если установлено, запустите модель в tensorrt.')
    parser .add_argument ("--use_anakin",action ='store_true',help ='Если установлено, запустите модель с использованием Анакина.')
    parser .add_argument ("--model_precision",type =str ,default ="float",choices =["float","int8"],help ='Если установлено, запустите модель с использованием Анакина.')
    parser .add_argument ("--filename",type =str ,default ="data/image.jpg",help ="путь к данным")
    parser .add_argument ("--warmup",type =int ,default =10 ,help ="warmup")
    parser .add_argument ("--repeat",type =int ,default =1000 ,help ="повторить раз")
    return parser .parse_args ()
if __name__ =="__main__":
    main ()