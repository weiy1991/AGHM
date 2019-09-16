####
#modified by Yuanwi 2018-12-19
####

import os
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model
#import VGG16_E2E as model
import numpy as np
#import densenet as model
#import ResNet50 as model
#import GoogleNet as model
#import vgg16 as model
#import BaiduE2E as model

# add by Yuanwei 2019-1-18 ref the id of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
# end by Yuanwei 2019-1-18

count_check_set = 8000

#LOGDIR = './'+driving_data.dataset_name+'BaiduE2E_save_focal_0.1_1.0_1.0'
#LOGDIR = './'+driving_data.dataset_name+'BaiduE2E_save_square'
#LOGDIR = './'+driving_data.dataset_name+'BaiduE2E_save_absolute'

# add by Wei Yuan 2019-07-02 test the SteeringLoss++
LOGDIR = './'+driving_data.dataset_name+'BaiduE2E_save_laplace' #+ '_constant_3.0'
def laplace_function(x, lambda_, mu_):
    return 1.0/(2.0*lambda_)-(1/((2*lambda_)) * np.e**(-1*(np.abs(x-mu_)/lambda_))) #  + 3.0


delta_ = float(driving_data.delta)
mu_cal_ = float(driving_data.mu)


#lambda_ = tf.placeholder_with_default(input=0.1511, shape=())
lambda_ = tf.placeholder_with_default(input=delta_, shape=())
mu_ = tf.placeholder_with_default(input=mu_cal_, shape=())
# end by Wei Yuan 2019-07-02

# add by Yuanwei loss parameters 2018-05-19
alpha_constant = tf.placeholder_with_default(input=1.0, shape=()) 
beta = tf.placeholder_with_default(input=1.0, shape=()) 
alpha = tf.placeholder_with_default(input=0.1, shape=())  
gamma = tf.placeholder_with_default(input=1.0, shape=())
# end loss parameters 2018-05-19


#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.InteractiveSession()
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 


L2NormConst = 0.001

train_vars = tf.trainable_variables()
# Square loss
#loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
# SteeringLoss
#loss = tf.reduce_mean(tf.multiply(tf.pow(tf.add(alpha_constant ,tf.multiply(alpha, tf.pow(tf.abs(model.y_), beta))), gamma), tf.square(tf.subtract(model.y_, model.y)))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
# Absolute loss
#loss = tf.reduce_mean(tf.abs(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
# end by Wei Yuan 2019-04-29

# add by Wei Yuan 2019-05 test SteeringLoss++
loss = tf.reduce_mean(laplace_function(model.y_, lambda_, mu_)*tf.square(tf.subtract(model.y_, model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
# end by Wei Yuan 2019-05

#end by Yuanwei 20171215
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())


# create a summary to monitor cost tensor

tf.summary.scalar("loss", loss)

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
#modified by Yuanwei 20171215
#logs_path = './'+driving_data.dataset_name+'BaiduE2E_logs_focal_0.1_1.0_1.0'
#logs_path = './'+driving_data.dataset_name+'BaiduE2E_logs_square'
#logs_path = './'+driving_data.dataset_name+'BaiduE2E_logs_absolute'

# add by Wei Yuan 2019-05 test SL++

logs_path = './'+driving_data.dataset_name+'_logs_laplace'# + '_constant_3.0'

# end by Wei Yuan 2019-05

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


epochs = 1000
batch_size = 100

# for DenseNet batch_size = 32

loss_value = 0.0
loss_his = 100.0

count_check = 0

loss_value_stop = []
def early_stop(num, loss_value):
  if len(loss_value_stop)<num:
    loss_value_stop.append(loss_value)
    return False
  old_mean_loss = sum(loss_value_stop)/len(loss_value_stop)

  del loss_value_stop[0]
  loss_value_stop.append(loss_value)
  new_mean_loss = sum(loss_value_stop)/len(loss_value_stop)

  print("get the loss_old %g, and the loss_new %g" % (new_mean_loss, old_mean_loss))

  if new_mean_loss>old_mean_loss:
    return True
  else:
    return False


# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)

    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    #train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8, model.if_training_placeholder: True})
    #train_step.run(feed_dict={model.x: xs, model.y_: ys})
    
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      #loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0, model.if_training_placeholder: True})
      #loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    #summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0, model.if_training_placeholder: True})
    #summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)


    if not os.path.exists(LOGDIR):
      os.makedirs(LOGDIR)
    if loss_value < loss_his:
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
      print("Best Model saved in file: %s with loss_value %g" % (filename, loss_value))
      loss_his = loss_value

      count_check = 0

    print("count_check:", count_check)

    if count_check>count_check_set:#1000:  #10000 and 1000
      print("early stop!")
      exit()

    count_check +=1


  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
          "--> tensorboard --logdir=./logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
