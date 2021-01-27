import argparse
import os
import numpy as np 
#import scipy.misc
import tensorflow as tf 
import random
import time
import makedata_digit_gray as makedata
from six.moves import xrange
from GAN_dann_gray import GAN_Net
from DQN import DQN_Net
import logging
from collections import deque 
from tqdm import tqdm

parser = argparse.ArgumentParser(description='')

parser.add_argument('--Dataset_name_source', dest='Dataset_name_source', default='S10')
parser.add_argument('--Dataset_name_target', dest='Dataset_name_target', default='M5')
parser.add_argument('--source_list', dest='source_list', default='./data_list/S10_list.txt')
parser.add_argument('--target_list', dest='target_list', default='./data_list/M5_list.txt')
parser.add_argument('--category_num', dest='category_num', type=int, default=10)

parser.add_argument('--epoch', dest='epoch', type=int, default=10000, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# batchsize of GAN')
parser.add_argument('--sample_dqn', dest='sample_dqn', type=int, default=32, help='# batchsize of RL')
parser.add_argument('--state_size', dest='state_size', default=64*16, help='scale state to this size')
parser.add_argument('--candidate_num', dest='candidate_num', type=int, default=16, help='candidate number')
parser.add_argument('--feature_dim', dest='feature_dim', type=int, default=64)
parser.add_argument('--test_iter', dest='test_iter', type=int, default=10)
parser.add_argument('--test_iter_s', dest='test_iter_s', type=int, default=100)

parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for GDC')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--BetaGD', dest='BetaGD', type=float, default=1.0, help='initial learning rate for adam')
parser.add_argument('--LambdaT', dest='LambdaT', type=float, default=0, help='initial learning rate for adam')
parser.add_argument('--lr_Q', dest='lr_Q', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--OBSERVE', dest='OBSERVE', type=int, default=100, help='# OBSERVE for the agent')
parser.add_argument('--EXPLORE', dest='EXPLORE', type=int, default=10000, help='# EXPLORE for the agent')
parser.add_argument('--INITIAL_EPSILON', dest='INITIAL_EPSILON', type=float, default=1.0, help='# INITIAL_EPSILON for the agent,1,0.9')
parser.add_argument('--FINAL_EPSILON', dest='FINAL_EPSILON', type=float, default=0, help='# FINAL_EPSILON for the agent,0,0.001,0.1')
parser.add_argument('--replace_target_iter', dest='replace_target_iter', type=int, default=20, help='# train for GAN')
parser.add_argument('--REPLAY_MEMORY', dest='REPLAY_MEMORY', type=int, default=2000, help='# replay memory for agent 100,500,1000,2000')
parser.add_argument('--REPLAY_MEMORY_GAN', dest='REPLAY_MEMORY_GAN', type=int, default=20, help='# EXPLORE for GAN')
parser.add_argument('--GAMMA', dest='GAMMA', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--checkpoint_dir_gan', dest='checkpoint_dir_gan', default='./checkpoint_gan', help='models are saved here')
parser.add_argument('--checkpoint_dir_dqn', dest='checkpoint_dir_dqn', default='./checkpoint_dqn', help='models are saved here')
parser.add_argument('--checkpoint_pretrain', dest='checkpoint_pretrain', default='./pretrain-Lenet-gray', help='models are saved here')
parser.add_argument('--feature_layers', dest='feature_layers', default=['feature_extractor'])
parser.add_argument('--classifier_layer', dest='classifier_layer', default=['label_predictor'])
parser.add_argument('--d_layer', dest='d_layer', default=['domain_predictor'])
parser.add_argument('--r_threshold', type= float, dest='r_threshold', default=0.3)
parser.add_argument('--r', type=int,dest='r', default=1)
parser.add_argument('--accuracy_path', dest='accuracy_path', default='./accuracy')
parser.add_argument('--log_path', dest='log_path', default='./log')
args = parser.parse_args()



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0' 

gpu_options = tf.GPUOptions(allow_growth=True)
sess_DQN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess_GAN = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def main():
    if not os.path.exists(args.checkpoint_dir_gan):
        os.makedirs(args.checkpoint_dir_gan)
    if not os.path.exists(args.checkpoint_dir_dqn):
        os.makedirs(args.checkpoint_dir_dqn)        
    if not os.path.exists(args.accuracy_path):
        os.makedirs(args.accuracy_path)    
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path) 
    train(args) 

def train(args):
    max_acc = 0
    max_acc_s = 0
    #np.set_printoptions(threshold='nan')
    print("D_score_w>t: r=+1------------------------------------------------------------------------------------")
    np.set_printoptions(threshold=np.inf)  
    filename = args.accuracy_path+os.sep+args.Dataset_name_source+'_to_'+args.Dataset_name_target+'_'+str(args.lr)+'.txt'

    f = open(filename,"w")
    f.write("begin training pretrain\n")
    for i in range(len(args.feature_layers)):
        f.write('feature_layers:'+args.feature_layers[i]+'\n')
    f.write('classifier_layer:'+args.classifier_layer[0]+'\n')
    f.write('lr:'+str(args.lr)+'\n')
    f.write('lr_Q:'+str(args.lr_Q)+'\n')
    f.write('OBSERVE:'+str(args.OBSERVE)+'\n')
    f.write('EXPLORE:'+str(args.EXPLORE)+'\n')
    f.write('source_list:'+str(args.source_list)+'\n')
    f.write('target_list:'+str(args.target_list)+'\n')
    f.write('batch_size:'+str(args.batch_size)+'\n')
    f.close()
    max_acc = 0
    GANetwork = GAN_Net(sess_GAN,args)
    DQNetwork = DQN_Net(sess_DQN,args)
    select_set = deque(maxlen=args.REPLAY_MEMORY_GAN)
    select_set_update = deque(maxlen=args.REPLAY_MEMORY_GAN)

    var_list_feature_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.feature_layers]
    var_list_calssifier_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.classifier_layer]
    var_list_discriminator_layers = [v for v in tf.trainable_variables() if v.name.split('/')[0] in args.d_layer]
    print("var_list_feature_layers:",var_list_feature_layers)
    print("var_list_calssifier_layers:",var_list_calssifier_layers)
    print("var_list_discriminator_layers:",var_list_discriminator_layers)
    print(args)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    D_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(GANetwork.D_loss, var_list=var_list_discriminator_layers)
    G_optim = tf.train.AdamOptimizer(learning_rate, beta1=args.beta1).minimize(GANetwork.G_loss, var_list=var_list_feature_layers)
    C_optim_S = tf.train.AdamOptimizer(args.lr, beta1=args.beta1)\
                .minimize(GANetwork.C_loss_S, var_list=[var_list_calssifier_layers,var_list_feature_layers])

    sess_DQN.run(tf.global_variables_initializer())
    sess_GAN.run(tf.global_variables_initializer())
    GANetwork.loadmodel(args.checkpoint_pretrain)

    #GANetwork.loadModel(sess_GAN)
    D_sum = tf.summary.merge([GANetwork.D_loss_S_sum,GANetwork.D_loss_T_sum,GANetwork.D_loss_sum,GANetwork.D_S_sum,GANetwork.D_T_sum])
    G_sum = tf.summary.merge([GANetwork.G_loss_sum, GANetwork.G_loss_S_sum,GANetwork.G_loss_T_sum])
    C_sum_S = GANetwork.C_loss_S_sum

    writer = tf.summary.FileWriter("./logs-SM")
    data_source,data_target  = makedata.get_data_list(args)
    correct_num = 0
    data_test_lists,test_labels = zip(*data_target)
    class_weight = np.zeros((1,args.category_num))

    for test_it in xrange(len(data_test_lists)):
        test_image = makedata.process_img(data_test_lists[test_it])
        class_pred_T_softmax  = sess_GAN.run([GANetwork.class_pred_T_softmax],\
        feed_dict={GANetwork.target_images: test_image,GANetwork.is_training: False})
        class_weight = class_weight + class_pred_T_softmax[0]

    class_weight = class_weight/float(len(data_test_lists))
    class_weight = class_weight/np.max(class_weight) 
    class_weight = class_weight.tolist()[0]

    for i in tqdm(xrange(args.epoch),desc='training'):        
        D_score = np.zeros([args.candidate_num])
        data_batch_source = random.sample(data_source,args.candidate_num)
        batch_source_list,batch_label = zip(*data_batch_source)

        data_batch_target = random.sample(data_target,args.candidate_num)
        batch_target_list,batch_target_label = zip(*data_batch_target)
        batch_source_images = makedata.process_imgs(list(batch_source_list))
        batch_target_images = makedata.process_imgs(list(batch_target_list))

        state,D_sorce_c = sess_GAN.run([GANetwork.X_feature_S,GANetwork.D_S],\
            feed_dict={GANetwork.source_images:batch_source_images,\
            GANetwork.is_training: False})
        D_score = D_sorce_c[:,args.category_num].copy()
        D_score_w = []

        for k in xrange(args.candidate_num):
            aaa = D_score[k]*class_weight[batch_label[k]]
            D_score_w.append(aaa)

        DQNetwork.setState(state)

        it = 0 
        while batch_source_images.shape[0] > 0:
            terminal = 0
            action,action_index,Flag= DQNetwork.getAction(args,it)
            if D_score_w[action_index] > args.r_threshold:
                reward = args.r                              
            else:
                reward = -args.r 
            print("action random",Flag,"source_label:",batch_label[action_index],\
                "D_score_w:",D_score_w[action_index],"reward:",reward)
            if DQNetwork.timeStep < args.OBSERVE:
                print("OBSERVE-------------------------------------------------------------------")
                select_set.append((batch_source_images[action_index],batch_label[action_index]))
            else:
                if reward > 0:
                    select_set_update.append((batch_source_images[action_index],batch_label[action_index])) 
            D_score_w = np.delete(D_score_w,action_index,axis=0)
            batch_source_images = np.delete(batch_source_images,action_index,axis = 0)
            batch_label = np.delete(batch_label,action_index,axis = 0)
            state = np.delete(state,action_index,axis=0)

            it = it+1
            next_state = makedata.get_nextstate(args,state,it)

            if reward < 0:
                terminal = 1
            DQNetwork.setPerception(args,action,reward,next_state,it,terminal,i)

            if reward < 0:
                break

        if len(select_set) > args.batch_size and DQNetwork.timeStep < args.OBSERVE:
            print("update discriminator---------------------------------------------------------")
            minibatch = random.sample(select_set,args.batch_size)
            source_batch = [data[0] for data in minibatch]
            source_batch_label = [data[1] for data in minibatch]
            print("source_batch_label:",source_batch_label)
            target_data = random.sample(data_target,args.batch_size)
            target_batch,target_batch_label = zip(*target_data)
            batch_target_images = makedata.process_imgs(list(target_batch))

            D_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                feed_dict={GANetwork.target_images: batch_target_images,\
                GANetwork.is_training: False})

            D_T_softmax = D_T_softmax[0]
            label_t = np.argmax(D_T_softmax,axis = 1)

            source_label_d = makedata.one_hot_label(source_batch_label,0,True,args)
            target_label_d = makedata.one_hot_label(label_t,1,True,args)
            source_label_g = makedata.one_hot_label(source_batch_label,0,False,args)
            target_label_g = makedata.one_hot_label(label_t,1,False,args)

            _,errD,summary_str = sess_GAN.run([D_optim,GANetwork.D_loss,D_sum],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sd:source_label_d, \
                       GANetwork.label_td:target_label_d,\
                       GANetwork.is_training: True})
            writer.add_summary(summary_str, i)
            print("Epoch: [%2d] d_loss: %.8f "\
                % (i, errD))  
        if len(select_set_update) > args.batch_size:
            p = float(i) / args.epoch
            l = 2. / (1. + np.exp(-10. * p)) - 1
            print("l: %f learning_rate: %f:"%(l,l*args.lr))
            print("adding adversarial loss---------------------------------------------------------")
            minibatch = random.sample(select_set_update,args.batch_size)
            source_batch = [data[0] for data in minibatch]
            source_batch_label = [data[1] for data in minibatch]
            print("source_batch_label:",source_batch_label)
            source_data = random.sample(data_source,args.batch_size)
            source_batch_all,source_batch_label_all = zip(*source_data)
            target_data = random.sample(data_target,args.batch_size)
            target_batch,target_batch_label = zip(*target_data)
            batch_source_images = makedata.process_imgs(list(source_batch_all))
            batch_target_images = makedata.process_imgs(list(target_batch))

            D_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                feed_dict={GANetwork.target_images: batch_target_images,\
                GANetwork.is_training: False})

            D_T_softmax = D_T_softmax[0]
            label_t = np.argmax(D_T_softmax,axis = 1)

            source_label_d = makedata.one_hot_label(source_batch_label,0,True,args)
            target_label_d = makedata.one_hot_label(label_t,1,True,args)
            source_label_g = makedata.one_hot_label(source_batch_label,0,False,args)
            target_label_g = makedata.one_hot_label(label_t,1,False,args)
            #print("target_label_d:",target_label_d)

            _, summary_str,errD = sess_GAN.run([D_optim, D_sum,GANetwork.D_loss],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sd:source_label_d, \
                       GANetwork.label_td:target_label_d,\
                       GANetwork.is_training: True})
            writer.add_summary(summary_str, i)

    # Update G network
            _,errG,summary_str = sess_GAN.run([G_optim,GANetwork.G_loss,G_sum],\
            feed_dict={GANetwork.source_images: source_batch,\
                       GANetwork.target_images: batch_target_images,\
                       GANetwork.label_sg:source_label_g, \
                       GANetwork.label_tg:target_label_g,\
                       GANetwork.is_training: True,\
                       learning_rate: l*args.lr})
            writer.add_summary(summary_str, i)

            _,errC_S,summary_str = sess_GAN.run([C_optim_S,GANetwork.C_loss_S,C_sum_S],\
            feed_dict={GANetwork.source_images: batch_source_images,\
                       GANetwork.y_label_S: source_batch_label_all,\
                       GANetwork.is_training: True})                
            writer.add_summary(summary_str, i)

            print("Epoch: [%2d] d_loss: %.8f, g_loss: %.8f, C_loss_S: %.8f"\
                % (i, errD,errG,errC_S))
                       
        if np.mod((i+1),args.test_iter) == 0 or i==0:
            correct_num = 0
            class_weight = np.zeros((1,args.category_num))           
            for test_it in xrange(len(data_test_lists)):
                test_image = makedata.process_img(data_test_lists[test_it])
                class_pred_T_softmax = sess_GAN.run([GANetwork.class_pred_T_softmax],\
                    feed_dict={GANetwork.target_images: test_image,GANetwork.is_training: False})
                label_pred = np.argmax(class_pred_T_softmax[0],axis=1)
                class_weight = class_weight + class_pred_T_softmax[0]
                if label_pred[0]==test_labels[test_it]:
                    correct_num = correct_num + 1
            class_weight = class_weight/float(len(data_test_lists))
            class_weight = class_weight/np.max(class_weight) 
            class_weight = class_weight.tolist()[0]
            target_accuracy = (correct_num/float(len(data_test_lists)))*100       
            print("Epoch: [%2d] target accuracy: %.8f " % (i,target_accuracy))
            f = open(filename,"a")
            f.write(str('Epoch: [%2d] target accuracy: %.2f '%(i,target_accuracy)))
            f.write('\n')
            f.close() 
            if target_accuracy > max_acc:
                max_acc = target_accuracy
                print("update max_acc")
                GANetwork.save(args.checkpoint_dir_gan,i)  
                DQNetwork.save(args.checkpoint_dir_dqn,i)  

        if np.mod((i+1),args.test_iter_s) == 0 and i > 0:
            correct_num_s = 0
            source_list,source_label = zip(*data_source) 
            for test_it_s in xrange(len(source_list)):
                test_image = makedata.process_img(source_list[test_it_s]) 
                class_pred_S_softmax = sess_GAN.run([GANetwork.class_pred_S_softmax],\
                    feed_dict={GANetwork.source_images: test_image,GANetwork.is_training: False})                  
                label_pred_s = np.argmax(class_pred_S_softmax[0],axis=1)
                if label_pred_s[0]==source_label[test_it_s]:
                    correct_num_s = correct_num_s + 1                
            source_accuracy = (correct_num_s/float(len(source_list)))*100    
            print("Epoch: [%2d] source accuracy: %.8f " % (i,source_accuracy))  
            temp =str("****Epoch: [%2d] source accuracy: %.2f" % (i,source_accuracy))+'\n'
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

