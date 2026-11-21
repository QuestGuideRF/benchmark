import argparse
import os
from static_graph .MMDNN .paddle .util import util
import torch
import models
import data
class BaseOptions ():
    def __init__ (self ):
        self .initialized =False
    def initialize (self ,parser ):
        parser .add_argument ('--dataroot',required =True ,help ='путь к изображениям (должен иметь подпапки trainA, trainB, valA, valB и т.д.)')
        parser .add_argument ('--name',type =str ,default ='experiment_name',help ='название эксперимента. Он решает, где хранить образцы и модели.')
        parser .add_argument ('--gpu_ids',type =str ,default ='0',help ='идентификаторы графического процессора: например. 0 0,1,2, 0,2. используйте -1 для процессора')
        parser .add_argument ('--checkpoints_dir',type =str ,default ='./checkpoints',help ='модели сохраняются здесь')
        parser .add_argument ('--model',type =str ,default ='cycle_gan',help ='выбирает, какую модель использовать. [цикл_ган | пикс2пикс | тест | раскрашивание]')
        parser .add_argument ('--input_nc',type =int ,default =3 ,help ='Количество каналов входного изображения: 3 для RGB и 1 для оттенков серого.')
        parser .add_argument ('--output_nc',type =int ,default =3 ,help ='Количество каналов выходного изображения: 3 для RGB и 1 для оттенков серого.')
        parser .add_argument ('--ngf',type =int ,default =64 ,help ='Количество фильтров генерации на последнем слое конвекторов')
        parser .add_argument ('--ndf',type =int ,default =64 ,help ='Количество дискриминационных фильтров в первом слое конвекторов')
        parser .add_argument ('--netD',type =str ,default ='basic',help ='указать архитектуру дискриминатора [базовый | n_слоев | пиксель]. Базовая модель — PatchGAN 70x70. n_layers позволяет указать слои в дискриминаторе')
        parser .add_argument ('--netG',type =str ,default ='resnet_9blocks',help ='укажите архитектуру генератора [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser .add_argument ('--n_layers_D',type =int ,default =3 ,help ='используется только если netD==n_layers')
        parser .add_argument ('--norm',type =str ,default ='instance',help ='нормализация экземпляра или пакетная нормализация [экземпляр | партия | никто]')
        parser .add_argument ('--init_type',type =str ,default ='normal',help ='инициализация сети [обычная | Ксавье | кайминг | ортогональный]')
        parser .add_argument ('--init_gain',type =float ,default =0.02 ,help ='коэффициент масштабирования для нормального, Ксавьера и ортогонального.')
        parser .add_argument ('--no_dropout',action ='store_true',help ='нет отключения генератора')
        parser .add_argument ('--dataset_mode',type =str ,default ='unaligned',help ='выбирает способ загрузки наборов данных. [без выравнивания | выровнено | одинокий | раскрашивание]')
        parser .add_argument ('--direction',type =str ,default ='AtoB',help ='АтоБ или БтоА')
        parser .add_argument ('--serial_batches',action ='store_true',help ='if true, takes images in order to make batches, otherwise takes them randomly')
        parser .add_argument ('--num_threads',default =4 ,type =int ,help ='# потоков для загрузки данных')
        parser .add_argument ('--batch_size',type =int ,default =1 ,help ='размер входного пакета')
        parser .add_argument ('--load_size',type =int ,default =286 ,help ='масштабировать изображения до этого размера')
        parser .add_argument ('--crop_size',type =int ,default =256 ,help ='затем обрезать до этого размера')
        parser .add_argument ('--max_dataset_size',type =int ,default =float ("inf"),help ='Максимальное количество выборок, разрешенное для каждого набора данных. Если каталог набора данных содержит больше max_dataset_size, загружается только подмножество.')
        parser .add_argument ('--preprocess',type =str ,default ='resize_and_crop',help ='масштабирование и обрезка изображений во время загрузки [resize_and_crop | урожай | масштаб_ширина | масштаб_ширина_и_обрезка | никто]')
        parser .add_argument ('--no_flip',action ='store_true',help ='if specified, do not flip the images for data augmentation')
        parser .add_argument ('--display_winsize',type =int ,default =256 ,help ='размер окна отображения как для Visdom, так и для HTML')
        parser .add_argument ('--epoch',type =str ,default ='latest',help ='какую эпоху загружать? установите последнюю версию, чтобы использовать последнюю кэшированную модель')
        parser .add_argument ('--load_iter',type =int ,default ='0',help ='какую итерацию загрузить? если load_iter > 0, код будет загружать модели с помощью iter_[load_iter]; в противном случае код загрузит модели к [эпохе]')
        parser .add_argument ('--verbose',action ='store_true',help ='if specified, print more debugging information')
        parser .add_argument ('--suffix',default ='',type =str ,help ='настраиваемый суффикс: opt.name = opt.name + суффикс: например, {model}_{netG}_size{load_size}')
        self .initialized =True
        return parser
    def gather_options (self ):
        if not self .initialized :
            parser =argparse .ArgumentParser (formatter_class =argparse .ArgumentDefaultsHelpFormatter )
            parser =self .initialize (parser )
        opt ,_ =parser .parse_known_args ()
        model_name =opt .model
        model_option_setter =models .get_option_setter (model_name )
        parser =model_option_setter (parser ,self .isTrain )
        opt ,_ =parser .parse_known_args ()
        dataset_name =opt .dataset_mode
        dataset_option_setter =data .get_option_setter (dataset_name )
        parser =dataset_option_setter (parser ,self .isTrain )
        self .parser =parser
        return parser .parse_args ()
    def print_options (self ,opt ):
        message =''
        message +='----------------- Options ---------------\n'
        for k ,v in sorted (vars (opt ).items ()):
            comment =''
            default =self .parser .get_default (k )
            if v !=default :
                comment ='\t[по умолчанию: %s]'%str (default )
            message +='{:>25}: {:<30}{}\n'.format (str (k ),str (v ),comment )
        message +='----------------- End -------------------'
        print (message )
        expr_dir =os .path .join (opt .checkpoints_dir ,opt .name )
        util .mkdirs (expr_dir )
        file_name =os .path .join (expr_dir ,'opt.txt')
        with open (file_name ,'wt')as opt_file :
            opt_file .write (message )
            opt_file .write ('\n')
    def parse (self ):
        opt =self .gather_options ()
        opt .isTrain =self .isTrain
        if opt .suffix :
            suffix =('_'+opt .suffix .format (**vars (opt )))if opt .suffix !=''else ''
            opt .name =opt .name +suffix
        self .print_options (opt )
        str_ids =opt .gpu_ids .split (',')
        opt .gpu_ids =[]
        for str_id in str_ids :
            id =int (str_id )
            if id >=0 :
                opt .gpu_ids .append (id )
        if len (opt .gpu_ids )>0 :
            torch .cuda .set_device (opt .gpu_ids [0 ])
        self .opt =opt
        return self .opt