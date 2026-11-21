from .base_options import BaseOptions
class TrainOptions (BaseOptions ):
    def initialize (self ,parser ):
        parser =BaseOptions .initialize (self ,parser )
        parser .add_argument ('--display_freq',type =int ,default =400 ,help ='частота отображения результатов тренировок на экране')
        parser .add_argument ('--display_ncols',type =int ,default =4 ,help ='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser .add_argument ('--display_id',type =int ,default =1 ,help ='идентификатор окна веб-дисплея')
        parser .add_argument ('--display_server',type =str ,default ="http://localhost",help ='visdom-сервер веб-дисплея')
        parser .add_argument ('--display_env',type =str ,default ='main',help ='имя среды отображения visdom (по умолчанию «основное»)')
        parser .add_argument ('--display_port',type =int ,default =8097 ,help ='порт Visdom веб-дисплея')
        parser .add_argument ('--update_html_freq',type =int ,default =1000 ,help ='частота сохранения результатов обучения в html')
        parser .add_argument ('--print_freq',type =int ,default =100 ,help ='частота отображения результатов обучения на консоли')
        parser .add_argument ('--no_html',action ='store_true',help ='не сохранять промежуточные результаты обучения в [opt.checkpoints_dir]/[opt.name]/web/')
        parser .add_argument ('--save_latest_freq',type =int ,default =5000 ,help ='частота сохранения последних результатов')
        parser .add_argument ('--save_epoch_freq',type =int ,default =5 ,help ='частота сохранения контрольных точек в конце эпох')
        parser .add_argument ('--save_by_iter',action ='store_true',help ='сохраняет ли модель путем итерации')
        parser .add_argument ('--continue_train',action ='store_true',help ='продолжить обучение: загрузить последнюю модель')
        parser .add_argument ('--epoch_count',type =int ,default =1 ,help ='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser .add_argument ('--phase',type =str ,default ='train',help ='поезд, вал, тест и т. д.')
        parser .add_argument ('--niter',type =int ,default =100 ,help ='# итера при начальной скорости обучения')
        parser .add_argument ('--niter_decay',type =int ,default =100 ,help ='# итера для линейного снижения скорости обучения до нуля')
        parser .add_argument ('--beta1',type =float ,default =0.5 ,help ='Импульсный срок Адама')
        parser .add_argument ('--lr',type =float ,default =0.0002 ,help ='начальная скорость обучения Адама')
        parser .add_argument ('--gan_mode',type =str ,default ='lsgan',help ='тип цели GAN. [ваниль| лсган | вгангп]. Потеря ванильного GAN — это цель перекрестной энтропии, использованная в оригинальной статье GAN.')
        parser .add_argument ('--pool_size',type =int ,default =50 ,help ='размер буфера изображений, в котором хранятся ранее сгенерированные изображения')
        parser .add_argument ('--lr_policy',type =str ,default ='linear',help ='политика скорости обучения. [линейный | шаг | плато | косинус]')
        parser .add_argument ('--lr_decay_iters',type =int ,default =50 ,help ='умножать на гамму каждую итерацию lr_decay_iters')
        self .isTrain =True
        return parser