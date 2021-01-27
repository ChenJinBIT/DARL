#coding:utf-8
"""
Resnet 输出维度为128，subset图片特征拼接起来，共128*10
目标域128*100
Generator 
Discriminator
condition :y = tf.placeholder(tf.float32, shape=[None, y_dim])噪声
"""
import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.slim as slim
import os
from six.moves import xrange
from utils import *


w = 32
h = 32
c = 1


class GAN_Net(object):
	def __init__(self,sess,args):
		self.source_images = tf.placeholder(tf.float32, [None, w,h,c])
		self.target_images = tf.placeholder(tf.float32, [None, w,h,c])
		self.is_training = tf.placeholder(tf.bool, shape=None,name="is_training")

		self.feature_dim = args.feature_dim
		self.category_num = args.category_num
		self.batch_size = args.batch_size

		self.y_label_S = tf.placeholder(tf.int32,[None])

		self.label_tg = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_td = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_sg = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.label_sd = tf.placeholder(tf.int32,[None,args.category_num+1])
		self.lambda_T = args.LambdaT
		self.BetaGD = args.BetaGD
		
		self.Dataset_name_source = args.Dataset_name_source
		self.Dataset_name_target = args.Dataset_name_target
		self.sess = sess
		self.build_model()
	def build_model(self):

		""" extract feature model """
		self.X_feature_S  = self.feature_extractor(self.source_images)
		self.X_feature_T = self.feature_extractor(self.target_images)

		self.class_pred_S  = self.classifier(self.X_feature_S)
		self.class_pred_T = self.classifier(self.X_feature_T)		

		self.D_S, self.D_logit_S = self.discriminator(self.X_feature_S)		
		self.D_T, self.D_logit_T = self.discriminator(self.X_feature_T)
		

		self.D_S_sum = tf.summary.histogram("D_S", self.D_S)
		self.D_T_sum = tf.summary.histogram("D_T", self.D_T)

		self.C_loss_S = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_label_S,logits=self.class_pred_S))
		
		self.class_pred_T_softmax = slim.softmax(self.class_pred_T)
		self.class_pred_S_softmax = slim.softmax(self.class_pred_S)
		self.C_T_softmax_sum_h = tf.summary.histogram("class_pred_T_softmax", self.class_pred_T_softmax)

		self.C_loss_T = -self.lambda_T *tf.reduce_mean(tf.reduce_sum(self.class_pred_T_softmax * tf.log(self.class_pred_T_softmax), axis=1))
		self.C_loss_S_sum = tf.summary.scalar("C_loss_S", self.C_loss_S)
		self.C_loss_T_sum = tf.summary.scalar("C_loss_T", self.C_loss_T)

		self.D_loss_S = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_S, labels=self.label_sd))
		self.D_loss_T = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_T, labels=self.label_td))		
		self.D_loss = self.BetaGD * (self.D_loss_S + self.D_loss_T)

		self.D_loss_S_sum = tf.summary.scalar("D_loss_S", self.D_loss_S)
		self.D_loss_T_sum = tf.summary.scalar("D_loss_T", self.D_loss_T)
		self.D_loss_sum = tf.summary.scalar("D_loss", self.D_loss)

		self.G_loss_S = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_S, labels=self.label_sg))
		self.G_loss_T = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logit_T, labels=self.label_tg))
		self.G_loss = self.BetaGD*(self.G_loss_T + self.G_loss_S)

		self.G_loss_S_sum = tf.summary.scalar("G_loss_S", self.G_loss_S)
		self.G_loss_T_sum = tf.summary.scalar("G_loss_T", self.G_loss_T)
		self.G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)
		self.saver = tf.train.Saver()
		
	def xavier_init(self,size):
		in_dim = size[0]
		xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
		return tf.random_normal(shape=size, stddev=xavier_stddev)

	def feature_extractor(self, image):
		with tf.variable_scope('feature_extractor',reuse=tf.AUTO_REUSE):
			with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
				with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
					net = slim.conv2d(image, 64, 5, scope='conv1')
					print("*********************************")
					print(net.get_shape())
					net = slim.max_pool2d(net, 2, stride=2, scope='pool1')
					print("*********************************")
					print(net.get_shape())
					net = slim.conv2d(net,128, 5, scope='conv2')
					print("*********************************")
					print(net.get_shape())
					net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
					print("*********************************")
					print(net.get_shape())
					net = tf.contrib.layers.flatten(net)
					print("###############################")
					net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc3')
					net = slim.dropout(net,0.5, is_training=self.is_training)
					feature = slim.fully_connected(net,64, activation_fn=tf.nn.relu,scope='fc4')
					print(tf.shape(feature))
		print(tf.shape(feature))
		return feature

	def classifier(self,feature):
		with tf.variable_scope('label_predictor',reuse=tf.AUTO_REUSE):
			logits = slim.fully_connected(feature,self.category_num, activation_fn=None, scope='fc5')
		return logits

		
	def discriminator(self,feature):
		with tf.variable_scope('domain_predictor',reuse=tf.AUTO_REUSE):
			net = slim.fully_connected(feature,500, activation_fn=tf.nn.relu, scope='dfc1')
			net = slim.fully_connected(feature,500, activation_fn=tf.nn.relu, scope='dfc2')
			d_logits = slim.fully_connected(net,self.category_num+1, activation_fn=None, scope='dfc3')
		return slim.softmax(d_logits), d_logits
		
	def save(self,checkpoint_dir,step):
		model_name = "GAN.model"
		model_dir = "%s_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target, self.feature_dim)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
			
		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		print("Reading checkpoint...")
		
		model_dir = "%s_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target,self.feature_dim)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False	


	def loadmodel(self, checkpoint_dir):
		print("Reading checkpoint...")
		
		model_dir = "%s_%s_%s" % (self.Dataset_name_source,self.Dataset_name_target,self.feature_dim)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
		
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			var = tf.contrib.framework.get_variables_to_restore()
			var = [v for v in var if not 'domain_predictor' in v.name]
			var = [v for v in var if not 'Adam' in v.name]
			var = [v for v in var if not 'eval_net' in v.name]
			var = [v for v in var if not 'power' in v.name]
			saver = tf.train.Saver(var)
			saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			print("successful")
			return True
		else:
			return False	