import os
import caffe
import argparse
import numpy as np
import scipy .misc
from PIL import Image
from util import segrun ,fast_hist ,get_scores
from cityscapes import cityscapes
parser =argparse .ArgumentParser ()
parser .add_argument ("--cityscapes_dir",type =str ,required =True ,help ="Путь к исходному набору данных городских пейзажей")
parser .add_argument ("--result_dir",type =str ,required =True ,help ="Путь к сгенерированным изображениям для оценки")
parser .add_argument ("--output_dir",type =str ,required =True ,help ="Где сохранить результаты оценки")
parser .add_argument ("--caffemodel_dir",type =str ,default ='./scripts/eval_cityscapes/caffemodel/',help ="Где хранится кофемодель FCN-8s")
parser .add_argument ("--gpu_id",type =int ,default =0 ,help ="Какой идентификатор графического процессора использовать")
parser .add_argument ("--split",type =str ,default ='val',help ="Разделение данных для оценки")
parser .add_argument ("--save_output_images",type =int ,default =0 ,help ="Сохранять ли выходные изображения FCN")
args =parser .parse_args ()
def main ():
    if not os .path .isdir (args .output_dir ):
        os .makedirs (args .output_dir )
    if args .save_output_images >0 :
        output_image_dir =args .output_dir +'image_outputs/'
        if not os .path .isdir (output_image_dir ):
            os .makedirs (output_image_dir )
    CS =cityscapes (args .cityscapes_dir )
    n_cl =len (CS .classes )
    label_frames =CS .list_label_frames (args .split )
    caffe .set_device (args .gpu_id )
    caffe .set_mode_gpu ()
    net =caffe .Net (args .caffemodel_dir +'/deploy.prototxt',
    args .caffemodel_dir +'fcn-8s-cityscapes.caffemodel',
    caffe .TEST )
    hist_perframe =np .zeros ((n_cl ,n_cl ))
    for i ,idx in enumerate (label_frames ):
        if i %10 ==0 :
            print ('Оценка: %d/%d'%(i ,len (label_frames )))
        city =idx .split ('_')[0 ]
        label =CS .load_label (args .split ,city ,idx )
        im_file =args .result_dir +'/'+idx +'_leftImg8bit.png'
        im =np .array (Image .open (im_file ))
        im =scipy .misc .imresize (im ,(label .shape [1 ],label .shape [2 ]))
        out =segrun (net ,CS .preprocess (im ))
        hist_perframe +=fast_hist (label .flatten (),out .flatten (),n_cl )
        if args .save_output_images >0 :
            label_im =CS .palette (label )
            pred_im =CS .palette (out )
            scipy .misc .imsave (output_image_dir +'/'+str (i )+'_pred.jpg',pred_im )
            scipy .misc .imsave (output_image_dir +'/'+str (i )+'_gt.jpg',label_im )
            scipy .misc .imsave (output_image_dir +'/'+str (i )+'_input.jpg',im )
    mean_pixel_acc ,mean_class_acc ,mean_class_iou ,per_class_acc ,per_class_iou =get_scores (hist_perframe )
    with open (args .output_dir +'/evaluation_results.txt','w')as f :
        f .write ('Средняя точность пикселей: %f\n'%mean_pixel_acc )
        f .write ('Средняя точность класса: %f\n'%mean_class_acc )
        f .write ('Средний класс IoU: %f\n'%mean_class_iou )
        f .write ('************ Номера классов ниже ************\n')
        for i ,cl in enumerate (CS .classes ):
            while len (cl )<15 :
                cl =cl +' '
            f .write ('%s: acc = %f, iou = %f\n'%(cl ,per_class_acc [i ],per_class_iou [i ]))
main ()