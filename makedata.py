import numpy as np
import os
import random
import shutil
from glob import glob
from six.moves import xrange
import cv2
import tensorflow as tf
import time

class_name=[]
feature_path=''

def get_data_list(args):
	list_source=[]
	list_target=[]
	label_s = []
	label_t = []
	file_s = open(args.source_list,'r')
	file_data_s = file_s.readlines()
	for row in file_data_s:
		tmp_list=row.split(' ')
		tmp_list[-1]=tmp_list[-1].replace('\n','')
		list_source.append(tmp_list[0])
		label_s.append(int(tmp_list[1]))
	
	file_t = open(args.target_list,'r')
	file_data_t = file_t.readlines()
	for row in file_data_t:
		tmp_list=row.split(' ')
		tmp_list[-1]=tmp_list[-1].replace('\n','')
		list_target.append(tmp_list[0])
		label_t.append(int(tmp_list[1]))
	data_source = list(zip(list_source,label_s))
	data_target = list(zip(list_target,label_t))
	return data_source,data_target

def one_hot_label(label_batch,domain_code,FlagD,args):

	num = len(label_batch)
	label = np.zeros((num,args.category_num+1))
	if FlagD:
		#the label for D
		if domain_code == 0:
			#the label for source
			for i in xrange(num):
				label[i,label_batch[i]]=1
		else:
			label[:,args.category_num]=1
	else:
		if domain_code == 0:
			label[:,args.category_num] = 1
		else:
			for i in xrange(num):
				label[i,label_batch[i]]=1
	return label

def process_imgs(path_list):
	batch_img = np.zeros([len(path_list),32,32,1])
	for i in xrange(len(path_list)):
		#print(path_list[i])
		img = cv2.resize(cv2.imread(path_list[i],cv2.IMREAD_GRAYSCALE),(32,32))
		batch_img[i,:,:,:] = img[:,:,np.newaxis]
	return batch_img
def process_img(path):
	img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(32,32))
	img = img[np.newaxis,:,:,np.newaxis]
	return img

def get_nextstate(args,state,selected_num):
    temp = np.zeros([1,args.feature_dim])
    for i in xrange(selected_num):
        state=np.vstack((state,temp))
    next_state = state.reshape(1,-1)
    return next_state