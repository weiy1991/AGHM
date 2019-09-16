####
#modified by Yuanwi 2018-12-19
####

import scipy.misc
import random
import numpy as np
import os
import h5py
import cv2
import csv

dataset_name = input("which dataset will be used to training or testing:")

# reference from Comma.ai
# def calc_curvature(v_ego, angle_steers, angle_offset=0):
#   deg_to_rad = np.pi/180.
#   slip_fator = 0.0014 # slip factor obtained from real data
#   steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
#   wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

#   angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
#   curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
#   return curvature

# define of the parameters:
# function: return the steering angle based on the inverse turning radius 
# u_t: the inverse turning radius
# d_w: the length between the front and rear wheels
# K_s: steering ratio
# K_slip: relative motion between a wheel and the surface of road
# v_t: the velocity of time t (m/s)

def AckermannSteering(u_t = 0.0, d_w = 2.67, K_s = 15.3, K_slip = 0.0014, v_t = 0.0):
    return u_t*d_w*K_s*(1.0 + K_slip*pow(v_t, 2))

def AckermannInverseRadius(theta_t = 0.0, d_w = 2.67, K_s = 15.3, K_slip = 0.0014, v_t = 0.0):
    return theta_t/(d_w*K_s*(1.0 + K_slip*pow(v_t, 2)))

# end by Wei Yuan 2019-06-25

# add by Wei Yuan 2019-08-18

import matplotlib.pyplot as plt
def showHist(data, bins):
    plt.hist(data,bins)
    plt.show()

# end by Wei Yuan 2019-08-18

# add by Wei Yuan 2019-09-10

def writeToDataset(data):
    for steering in data:
        with open("result/"+dataset_name+"_Dataset.csv", 'a' , newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([steering])
            csvFile.close()

# end by Wei Yuan 2019-09-10


xs = []
ys = []

xs_ = []
ys_ = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#modified by Yuanwei 20171224
def read_csv(filename):
    with open(filename, 'r') as f:
        lines_all = [ln.strip().split(",")[:] for ln in f.readlines()]
        #del(lines_all[0]) # remove the head of the csv
        if dataset_name=="DBNet" or dataset_name=="CARLA" or dataset_name=="Commaai":
            lines_all = map(lambda x: (x[0], np.float(x[1]), np.float(x[2])), lines_all)
        elif dataset_name=="Udacity":
            lines_all = map(lambda x: (x[5], np.float(x[6]), np.float(x[8])), lines_all)
        else: # Baidu
            lines_all = map(lambda x: (x[0], np.float(x[1])), lines_all)


        return lines_all

def getTrainingData(filename):
    lines_all = read_csv(filename)
    for ln in lines_all:
        #if ln[0].find('center')  != -1:
        if dataset_name=="DBNet":  
            xs_.append("/media/weiy/weiy/data/DBNet/"+ln[0])
            steering = 100.0*AckermannInverseRadius(theta_t = -ln[1]*np.pi/180.0, d_w = 3.088, K_s = 15.3, K_slip = 0.0014, v_t = (ln[2]*1000.0/3600.0))
            ys_.append(steering)
        elif dataset_name=="CARLA":
            xs_.append("/media/weiy/weiy/data/AgentHuman/imgcsv/img/"+ln[0])
            steering = 100.0*AckermannInverseRadius(theta_t = -ln[1]*70.0*np.pi/180.0, d_w = 2.72, K_s = 16.0, K_slip = 0.0014, v_t = (ln[2]*1000.0/3600.0))
            ys_.append(steering)
        elif dataset_name=="Udacity":
            if ln[0].find('center') == -1:
                continue
            xs_.append("/media/weiy/weiy/data/udacity-output/"+ln[0])
            steering = 100.0*AckermannInverseRadius(theta_t = ln[1], d_w = 2.69, K_s = 15.6, K_slip = 0.0014, v_t = (ln[2]*1000.0/3600.0))
            ys_.append(steering)
        elif dataset_name=="Commaai":
            xs_.append("/media/weiy/weiy/data/comma.ai/img/"+ln[0])
            steering = 100.0*AckermannInverseRadius(theta_t = ln[1]*np.pi/180.0, d_w = 2.67, K_s = 15.3, K_slip = 0.0014, v_t = (ln[2]*1000.0/3600.0))
            ys_.append(steering)
        else:
            xs_.append("/media/weiy/weiy/data/Baidu/img/"+ln[0])
            ys_.append(100.0*ln[1])


if dataset_name=="DBNet": #DBNet dataset
    getTrainingData("/media/weiy/weiy/data/DBNet/driving_log.csv")
elif dataset_name=="Commaai": # Baidu Dataset 
    getTrainingData("/media/weiy/weiy/data/comma.ai/driving_log.csv")
elif dataset_name=="CARLA": # CARLA dataset
    getTrainingData("/media/weiy/weiy/data/AgentHuman/imgcsv/driving_log.csv")
elif dataset_name=="Udacity": # Udacity dataset
    getTrainingData("/media/weiy/weiy/data/udacity-output/interpolated_train_shuffle.csv")
else: # Commaai dataset
    getTrainingData("/media/weiy/weiy/data/Baidu/driving_log.csv")


xs = xs_
ys = ys_

#end by Yuaniwei 20171224



#shuffle list of images
c = list(zip(xs, ys))
random.seed(0)
random.shuffle(c)
#xs, ys = zip(*c)

if dataset_name=="CARLA" or dataset_name=="Commaai" or dataset_name=="BDD":
    xs, ys = zip(*c[:30000])
else:
    xs, ys = zip(*c)


#get number of images
num_images = len(xs)
print("num_images:", num_images)

train_xs = xs[:int(len(xs) * 0.6)]
train_ys = ys[:int(len(xs) * 0.6)]

val_xs = xs[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]
val_ys = ys[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]

test_xs = xs[-int(len(xs) * 0.2):]
test_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

# get the parameter of laplace add by Wei Yuan 2019-07-25
np.seterr(invalid='ignore')

mu = np.median(ys)
delta = np.mean(abs(ys-mu))

print("the mu is:", mu)
print("tht delta is:", delta)

showHist(train_ys, 200)

#writeToDataset(ys)

#input("waiting")

# end the laplace


def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        #modified by Yuanwei 20171224
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]), [66, 200]) / 255.0)
        # for DenseNet
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]), [224, 224]) / 255.0)
        # end for DenseNet
        # for ReseNet
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]), [320, 320]) / 255.0)
        # end for ReseNet

        #scipy.misc.imshow(x_out[i])
        #x_out.append(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images]) / 255.0)
        #end by Yuanwei 20171224
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
        #print(y_out[i])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        #modified by Yuanwei 20171224
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]), [66, 200]) / 255.0)
        # for DenseNet
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]), [224, 224]) / 255.0)
        # end for DenseNet
        # for ReseNet
        #x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]), [320, 320]) / 255.0)
        # end for ReseNet
        #x_out.append(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images]) / 255.0)
        #end by Yuanwei 20171224
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out

