######adding random filp and random crop when training, center crop when testing
import argparse
import os
import numpy as np 
#import scipy.misc
import tensorflow as tf
import random
import time
import makedata_digit_gray as makedata
#from skimage import transform
from six.moves import xrange
from GAN_dann_gray import GAN_Net
import logging
import tensorflow.examples.tutorials.mnist.input_data as input_data
#from tqdm import tqdm
#from DQN import DQN_Net
#from glob import glob
#from scipy.misc import imread
#from collections import deque 

parser = argparse.ArgumentParser(description='')

parser.add_argument('--Dataset_name_source', dest='Dataset_name_source', default='U10')
parser.add_argument('--Dataset_name_target', dest='Dataset_name_target', default='M5')
parser.add_argument('--source_list', dest='source_list', default='./data_list/U10_list.txt')
#parser.add_argument('--source_test_list', dest='source_test_list', default='./data_list_crop/Ar_65_list.txt')
parser.add_argument('--target_list', dest='target_list', default='./data_list/M5_list.txt')
#parser.add_argument('--target_test_list', dest='target_test_list', default='./data_list_crop/Pr_25_list.txt')
#parser.add_argument('--Class_name', dest='Class_name', default=\
#['motorbike','monitor','horse','dog','car','bottle','boat','bird','bike','airplane','people','bus'])
#parser.add_argument('--Class_name', dest='Class_name', default=['projector','mug','mouse','monitor','laptop_computer','keyboard','headphones','calculator','bike','back_pack'])
parser.add_argument('--category_num', dest='category_num', type=int, default=10)
#parser.add_argument('--resnet_depth', dest='resnet_depth', type=int, default=50)
#parser.add_argument('--total_feature_path', dest='total_feature_path', default=\
#    '/media/mcislab3d/Elements/chenjin/IJCV/data/crop_224')
#parser.add_argument('--total_feature_path', dest='total_feature_path', default=\
#    '/home/mcislab/chenjin/data')
parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# batchsize of GAN')
parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=84)
#parser.add_argument('--save_iter_gan', dest='save_iter_gan', type=int, default=50,help='save model')
parser.add_argument('--test_iter', dest='test_iter', type=int, default=100)
parser.add_argument('--test_iter_s', dest='test_iter_s', type=int, default=100)

parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for GDC')
parser.add_argument('--decay_steps', dest='decay_steps', type=int, default=1000, help='momentum term of adam')
parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--LambdaT', dest='LambdaT', type=float, default=0, help='momentum term of adam')
parser.add_argument('--BetaGD', dest='BetaGD', type=float, default=1.0, help='initial learning rate for adam')
parser.add_argument('--checkpoint_dir_pretrain', dest='checkpoint_dir_pretrain', default='./pretrain-Lenet-gray-DANN', help='models are saved here')
#parser.add_argument('--alexnet_model_path', dest='alexnet_model_path', default='./ResNet-L50.npy')
#parser.add_argument('--keepPro', dest='keepPro',type=int, default=1.0)
#parser.add_argument('--skip', dest='skip', default=['fc8'])
parser.add_argument('--feature_layers', dest='feature_layers', default=['feature_extractor'])
parser.add_argument('--classifier_layer', dest='classifier_layer', default=['label_predictor'])
parser.add_argument('--accuracy_path', dest='accuracy_path', default='./accuracy_finetune_gray_dann')
parser.add_argument('--log_path', dest='log_path', default='./log_finetune_gray_dann')
#parser.add_argument('--train_layers', dest='train_layers', default=['scale5','fc256'])
#parser.add_argument('--bottleneck_layers', dest='bottleneck_layers', default=['fc256'])
args = parser.parse_args()
print(args)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess_DQN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess_GAN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess_Alexnet = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def main():
    if not os.path.exists(args.checkpoint_dir_pretrain):
        os.makedirs(args.checkpoint_dir_pretrain)
    if not os.path.exists(args.accuracy_path):
        os.makedirs(args.accuracy_path)        
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path) 
    pretrain(args) 

def pretrain(args):
    max_acc = 0
    max_acc_s = 0
    #save_Flag = True
    print(args.source_list.split('/')[-1].split('_')[0])

    filename = args.accuracy_path+os.sep+args.Dataset_name_source+'_to_'+args.Dataset_name_target+'_'+str(args.lr)+'_Lenet_dann_gray.txt'
    logfile = args.log_path+os.sep+args.Dataset_name_source+'_to_'+args.Dataset_name_target+'_'+str(args.lr)+'_Lenet_dann_gray.log'
    handler = logging.FileHandler(logfile, mode='a')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    f = open(filename,"a")
    f.write("begin training lr\n")
    for i in range(len(args.feature_layers)):
        f.write('feature_layers:'+args.feature_layers[i]+'\n')
    f.write('classifier_layer:'+args.classifier_layer[0]+'\n')
    f.write('lr:'+str(args.lr)+'\n')
    #f.write('decay:'+str(args.decay_rate)+'\n')
    #f.write('decay_steps:'+str(args.decay_steps)+'\n')
    f.write('source_list:'+str(args.source_list)+'\n')
    f.write('target_list:'+str(args.target_list)+'\n')
    f.write('batch_size:'+str(args.batch_size)+'\n')
    f.close()
    GANetwork = GAN_Net(sess_GAN,args)
    var_list_feature_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.feature_layers]
    var_list_calssifier_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.classifier_layer]
    print("var_list_feature_layers:",var_list_feature_layers)
    print("var_list_calssifier_layers:",var_list_calssifier_layers)
    #is_training = tf.placeholder('bool', [])

    #var_list_feature_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.train_layers]
    #print("var_list_feature_layers:",var_list_feature_layers)
    #var_list_bottleneck_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.bottleneck_layers]
    #C_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
    #            .minimize(GANetwork.C_loss_S)
    #global_step = tf.Variable(0,name='global_step',trainable=False)
    #lr=tf.train.exponential_decay(\
    #   learning_rate = args.lr,global_step=global_step,decay_steps=args.decay_steps,decay_rate=args.decay_rate,staircase=False)
    #C_optim_SF = tf.train.AdamOptimizer(learning_rate=lr)\
    #            .minimize(GANetwork.C_loss_S, var_list=[var_list_feature_layers])
    C_optim_SB = tf.train.AdamOptimizer(learning_rate=args.lr)\
                .minimize(GANetwork.C_loss_S, var_list=[var_list_feature_layers,var_list_calssifier_layers])
    #C_optim_SB = tf.train.AdamOptimizer(learning_rate=args.lr,beta1=0.5,beta2=0.999).minimize(GANetwork.C_loss_S)
    #C_optim = tf.group(C_optim_SF,C_optim_SB)
    #C_optim_T = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
    #            .minimize(GANetwork.C_loss_T, var_list=[GANetwork.theta_C,var_list_alexnet])

    sess_GAN.run(tf.global_variables_initializer())
    GANetwork.load(args.checkpoint_dir_pretrain)
    #GANetwork.loadModel(sess_GAN)
    #D_sum = tf.summary.merge([GANetwork.D_loss_S_sum,GANetwork.D_loss_T_sum,GANetwork.D_loss_sum,GANetwork.D_S_sum,GANetwork.D_T_sum])
    #G_sum = tf.summary.merge([GANetwork.G_loss_sum, GANetwork.G_loss_S_sum,GANetwork.G_loss_T_sum])
    C_sum_S = GANetwork.C_loss_S_sum
    #C_sum_T = GANetwork.C_loss_T_sum
    #Q_sum = DQNetwork.Q_loss_sum
    writer = tf.summary.FileWriter("./logs-finetune-Lenet-1e4-MU-gray",sess_GAN.graph)
    #writer_dqn = tf.summary.FileWriter("./logs-DQN",sess_DQN.graph)
    data_source,data_target  = makedata.get_data_list(args)
    #data_source_test,data_target_test = makedata.get_test_data_list(args)
    #target_list,target_label = zip(*data_target)
    #target_images_total = makedata.process_imgs(target_list)    
    #source_list, source_label = zip(*data_source)
    #source_images_total = makedata.process_imgs(source_list)
    print(len(data_target))
    #add_iter = global_step.assign_add(1)
    for i in xrange(args.epoch):        
        #random.shuffle(data_source)
        data_batch_source = random.sample(data_source,args.batch_size)
        batch_source_list,batch_label = zip(*data_batch_source)
        #data_batch_target = random.sample(data_target,args.candidate_num)
        #batch_target_list,batch_target_label = zip(*data_batch_target)
        batch_label = list(batch_label)
        #print(len(batch_label))
        start_time =time.time()
        batch_source_images = makedata.process_imgs(list(batch_source_list))
        #batch_source_images = sess_data.run(images_batch) 
        #print("batch_source_images.shape:",batch_source_images.shape)
        #logger.info("batch time of processing image:[%f]"%(time.time()-start_time))
        #batch_target_images = makedata.process_imgs(list(batch_target_list))

        summary_str,_,errC_S = sess_GAN.run([C_sum_S,C_optim_SB,GANetwork.C_loss_S],\
        feed_dict={GANetwork.source_images: batch_source_images,\
                   GANetwork.y_label_S: batch_label,\
                   GANetwork.is_training: True})                
        writer.add_summary(summary_str, i)
        logger.info("Epoch: [%2d] C_loss_S: %.8f" % (i,errC_S))  
        #print(sess_GAN.run('layer-conv1/bias:0'))                     

#print('______________________________________________________')
        #test_image = np.zeros([1,224,224,3])

        if np.mod((i+1),args.test_iter) == 0 or i==0:
            correct_num = 0
            data_test_lists,test_labels = zip(*data_target)  
            start_time =time.time()        
            for test_it in xrange(len(data_test_lists)):
                test_image = makedata.process_img(data_test_lists[test_it])
                class_pred_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                    feed_dict={GANetwork.target_images:test_image,GANetwork.is_training: False})
                label_pred = np.argmax(class_pred_T_softmax[0],axis=1)
                if label_pred[0]==int(test_labels[test_it]):
                    correct_num=correct_num + 1
            target_accuracy = (correct_num/float(len(data_test_lists)))*100       
            #logger.info("testing time:[%f]"%(time.time()-start_time))
            logger.info("Epoch: [%2d] target accuracy: %.8f " % (i,target_accuracy))
            #g_temp,lr_temp=sess_GAN.run([global_step,lr])
            f = open(filename,"a")
            #f.write(str('step: %d,lr:%.8f'%(g_temp,lr_temp)))
            #f.write('\n')
            f.write(str('Epoch: [%2d] target accuracy: %.2f '%(i,target_accuracy)))
            f.write('\n')
            f.close()
            if target_accuracy > max_acc:
                max_acc = target_accuracy
                logger.info("saving model")
                GANetwork.save(args.checkpoint_dir_pretrain,i) 
                #DQNetwork.save(args.checkpoint_dir_dqn, DQNetwork.timeStep)  

        if np.mod((i+1),args.test_iter_s) == 0:
            correct_num_s = 0
            source_lists,source_labels = zip(*data_source)
            print(len(source_lists))
            for test_it_s in xrange(len(source_lists)):
                test_image_s = makedata.process_img(source_lists[test_it_s])
                class_pred_S_softmax = sess_GAN.run([GANetwork.class_pred_S_softmax],\
                    feed_dict={GANetwork.source_images: test_image_s,GANetwork.is_training: False})                  
                label_pred_s = np.argmax(class_pred_S_softmax[0],axis=1)
                if label_pred_s[0]== source_labels[test_it_s]:
                    correct_num_s=correct_num_s + 1                
            source_accuracy = (correct_num_s/float(len(source_lists)))*100     
            logger.info("Epoch: [%2d] source accuracy: %.8f " % (i,source_accuracy))    
            temp = "source accuracy="+str("Epoch: [%2d] source accuracy: %.2f" % (i,source_accuracy))+'\n'
            #if source_accuracy > max_acc_s:
            #    max_acc_s = source_accuracy
            #    print("saving model source")
            #    GANetwork.save(args.checkpoint_dir_pretrain_s,i)

            f = open(filename,"a")
            f.write(temp)
            f.close()
    f = open(filename,"a")
    f.write("max_acc=") 
    temp = str(float('%.2f'%(max_acc)))+'\n'
    f.write(temp)    
    f.close()   
if __name__ == '__main__':
    main()

