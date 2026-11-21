import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv .data .transforms import presets
from matplotlib import pyplot as plt
def parse_args ():
    parser =argparse .ArgumentParser (description ='Протестируйте сети YOLO.')
    parser .add_argument ('--network',type =str ,default ='yolo3_darknet53_coco',
    help ="Имя базовой сети")
    parser .add_argument ('--images',type =str ,default ='',
    help ='Тестовые изображения. Для разделения нескольких изображений используйте запятую.')
    parser .add_argument ('--gpus',type =str ,default ='',
    help ='Для обучения с использованием графических процессоров вы можете указать, например, 1,3.')
    parser .add_argument ('--pretrained',type =str ,default ='True',
    help ='Загрузите веса из ранее сохраненных параметров.')
    parser .add_argument ('--thresh',type =float ,default =0.5 ,
    help ='Порог оценки объекта при визуализации bbox.')
    args =parser .parse_args ()
    return args
if __name__ =='__main__':
    args =parse_args ()
    ctx =[mx .gpu (int (i ))for i in args .gpus .split (',')if i .strip ()]
    ctx =[mx .cpu ()]if not ctx else ctx
    if not args .images .strip ():
        gcv .utils .download ("https://cloud.githubusercontent.com/assets/3307514/"+
        "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg",'street.jpg')
        image_list =['street.jpg']
    else :
        image_list =[x .strip ()for x in args .images .split (',')if x .strip ()]
    if args .pretrained .lower ()in ['true','1','yes','t']:
        net =gcv .model_zoo .get_model (args .network ,pretrained =True )
    else :
        net =gcv .model_zoo .get_model (args .network ,pretrained =False ,pretrained_base =False )
        net .load_parameters (args .pretrained )
    net .set_nms (0.45 ,200 )
    net .collect_params ().reset_ctx (ctx =ctx )
    for image in image_list :
        ax =None
        x ,img =presets .yolo .load_test (image ,short =512 )
        x =x .as_in_context (ctx [0 ])
        ids ,scores ,bboxes =[xx [0 ].asnumpy ()for xx in net (x )]
        ax =gcv .utils .viz .plot_bbox (img ,bboxes ,scores ,ids ,thresh =args .thresh ,
        class_names =net .classes ,ax =ax )
        plt .show ()