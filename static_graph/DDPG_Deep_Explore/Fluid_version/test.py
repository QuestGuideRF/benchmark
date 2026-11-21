import sys
sys .path .append ('../PARL')
import time
from DDPG_algorithm import DDPGAlgorithm
from DDPG_agent import DDPGAgent
import paddle .fluid .profiler as profiler
import pickle
ACTION_DIM =19
VEL_OBS_DIM =4
OBS_DIM =185 +VEL_OBS_DIM
GAMMA =0.96
TAU =0.001
actor_lr =3e-5
critic_lr =3e-5
train_times =100
BATCH_SIZE =128
ensemble_num =12
alg =DDPGAlgorithm (OBS_DIM ,VEL_OBS_DIM ,ACTION_DIM ,GAMMA ,TAU ,gpu_id =0 ,ensemble_num =1 )
agent =DDPGAgent (alg )
with open ('batch_data.pickle','rb')as f :
    batch_data =pickle .load (f )
def test ():
    global alg ,agent ,batch_data
    [states ,actions ,rewards ,dones ,new_states ]=batch_data
    start =time .time ()
    for _ in range (100 ):
        agent .learn (states ,actions ,rewards ,new_states ,dones ,actor_lr ,critic_lr ,0 )
    print ('время: {}'.format (time .time ()-start ))
if __name__ =='__main__':
    for i in range (100 ):
        print ('Тестирование: {}'.format (i ))
        test ()