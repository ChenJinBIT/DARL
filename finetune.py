import argparse
import os
import numpy as np 
import tensorflow as tf
import random
import time
import makedata_digit_gray as makedata
from six.moves import xrange
from GAN_dann_gray import GAN_Net
import logging
import tensorflow.examples.tutorials.mnist.input_data as input_data

parser = argparse.ArgumentParser(description='')

parser.add_argument('--Dataset_name_source', dest='Dataset_name_source', default='S10')
parser.add_argument('--Dataset_name_target', dest='Dataset_name_target', default='M5')
parser.add_argument('--source_list', dest='source_list', default='./data_list/S10_list.txt')
parser.add_argument('--target_list', dest='target_list', default='./data_list/M5_list.txt')
parser.add_argument('--category_num', dest='category_num', type=int, default=10)
parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# batchsize of GAN')
parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=84)
parser.add_argument('--test_iter', dest='test_iter', type=int, default=100)
parser.add_argument('--test_iter_s', dest='test_iter_s', type=int, default=100)

parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for GDC')
parser.add_argument('--decay_steps', dest='decay_steps', type=int, default=1000, help='momentum term of adam')
parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--LambdaT', dest='LambdaT', type=float, default=0, help='momentum term of adam')
parser.add_argument('--BetaGD', dest='BetaGD', type=float, default=1.0, help='initial learning rate for adam')
parser.add_argument('--checkpoint_dir_pretrain', dest='checkpoint_dir_pretrain', default='./pretrain-Lenet-gray', help='models are saved here')
parser.add_argument('--feature_layers', dest='feature_layers', default=['feature_extractor'])
parser.add_argument('--classifier_layer', dest='classifier_layer', default=['label_predictor'])
parser.add_argument('--accuracy_path', dest='accuracy_path', default='./accuracy_finetune_gray')
parser.add_argument('--log_path', dest='log_path', default='./log_finetune_gray')
args = parser.parse_args()
print(args)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu_options = tf.GPUOptions(allow_growth=True)
sess_GAN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
    print(args.source_list.split('/')[-1].split('_')[0])

    filename = args.accuracy_path+os.sep+args.Dataset_name_source+'_to_'+args.Dataset_name_target+'_'+str(args.lr)+'_Lenet_gray.txt'
    logfile = args.log_path+os.sep+args.Dataset_name_source+'_to_'+args.Dataset_name_target+'_'+str(args.lr)+'_Lenet_gray.log'
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
    f.write('source_list:'+str(args.source_list)+'\n')
    f.write('target_list:'+str(args.target_list)+'\n')
    f.write('batch_size:'+str(args.batch_size)+'\n')
    f.close()
    GANetwork = GAN_Net(sess_GAN,args)
    var_list_feature_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.feature_layers]
    var_list_calssifier_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.classifier_layer]
    print("var_list_feature_layers:",var_list_feature_layers)
    print("var_list_calssifier_layers:",var_list_calssifier_layers)
    C_optim_SB = tf.train.AdamOptimizer(learning_rate=args.lr)\
                .minimize(GANetwork.C_loss_S, var_list=[var_list_feature_layers,var_list_calssifier_layers])
    sess_GAN.run(tf.global_variables_initializer())
    GANetwork.load(args.checkpoint_dir_pretrain)

    C_sum_S = GANetwork.C_loss_S_sum
    writer = tf.summary.FileWriter("./logs-finetune-Lenet-1e4-SM-gray",sess_GAN.graph)
    data_source,data_target  = makedata.get_data_list(args)
    print(len(data_target))
    for i in xrange(args.epoch):        
        data_batch_source = random.sample(data_source,args.batch_size)
        batch_source_list,batch_label = zip(*data_batch_source)
        batch_label = list(batch_label)
        #print(len(batch_label))
        start_time =time.time()
        batch_source_images = makedata.process_imgs(list(batch_source_list))
        summary_str,_,errC_S = sess_GAN.run([C_sum_S,C_optim_SB,GANetwork.C_loss_S],\
        feed_dict={GANetwork.source_images: batch_source_images,\
                   GANetwork.y_label_S: batch_label,\
                   GANetwork.is_training: True})                
        writer.add_summary(summary_str, i)
        logger.info("Epoch: [%2d] C_loss_S: %.8f" % (i,errC_S))                      

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
            if source_accuracy > max_acc_s:
                max_acc_s = source_accuracy
                print("saving model source")
                GANetwork.save(args.checkpoint_dir_pretrain,i)

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

