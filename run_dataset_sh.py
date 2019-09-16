import tensorflow as tf
import scipy.misc
#import model
import cv2
import csv
import numpy as np
from subprocess import call
import os
from math import sqrt
import driving_data

import model
#import VGG16_E2E as model
#import densenet as model
#import ResNet50 as model
#import GoogleNet as model
#import vgg16 as model
#import BaiduE2E as model

# add by Wei Yuan 2019-05-05
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# end by Wei Yuan 2019-05-05

# add by YuanWei 20181112

def getSaveFile(dir, tag):
    save_dir = []
    save_files = os.listdir(dir)
    for file in save_files:
        if os.path.isdir(file) and file.find(tag)!=-1:
            save_dir.append(file)
    return save_dir

save_dir = getSaveFile('.', driving_data.dataset_name+'save')

print(save_dir)
input("waiting")

# end by YuanWei 20181112

sess = tf.InteractiveSession()
saver = tf.train.Saver()

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#modified by Yuanwei 20171224
xs_ = []
ys_ = []

def getPerformance(target, prediction):

    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])


    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 
        absError.append(abs(val))#误差绝对值


    MSE = sum(squaredError) / len(squaredError)

    RMSE = sqrt(sum(squaredError) / len(squaredError))

    MAE = sum(absError) / len(absError)

    predictionDeviation = []
    predictionMean = sum(prediction) / len(prediction)#target平均值
    for val in prediction:
        predictionDeviation.append((val - predictionMean) * (val - predictionMean))

    VAR = sum(predictionDeviation) / len(predictionDeviation)

    SD = sqrt(sum(predictionDeviation) / len(predictionDeviation))

    return MSE, RMSE, MAE, VAR, SD

def writeToResult(name, MSE, RMSE, MAE, VAR, SD):
    with open("result/"+driving_data.dataset_name+".csv", 'a' , newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow([name,MSE, RMSE, MAE, VAR, SD])
        csvFile.close()

xs_ = driving_data.test_xs
ys_ = driving_data.test_ys

for save_file in save_dir:
    print("save_file:", save_dir)
    saver.restore(sess, save_file+"/model.ckpt")

    target = []
    prediction = []
    i = 0
    while(cv2.waitKey(10) != ord('q') and i<len(xs_)):
        #modified by Yuanwei 20171224
        full_image = scipy.misc.imread(xs_[i], mode="RGB")
        #image = scipy.misc.imresize(full_image, [66, 200]) / 255.0 #VGG
        #image = scipy.misc.imresize(full_image, [224, 224]) / 255.0 #GoogleNet DenseNet 
        image = scipy.misc.imresize(full_image, [320, 320]) / 255.0 #ReseNet

        degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]

        #degrees = model.y.eval(feed_dict={model.x:[image],  model.keep_prob: 1.0})[0][0]
        #degrees = model.y.eval(feed_dict={model.x:[image],  model.keep_prob: 1.0, model.if_training_placeholder: False})[0][0] #GoogleNet DenseNet 
        #degrees = model.y.eval(feed_dict={model.x:[image]})[0][0] # VGG ResNet


        with open("result/"+save_file+".csv", 'a' , newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow([degrees, ys_[i]])
                    #print("pre_steer:%f,  pre_speed:%f",pre_steer,  pre_speed)
                    csvFile.close()
        #end by Yuanwei 20171224

        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)

        target.append(ys_[i])
        prediction.append(degrees)

        i += 1


    MSE, RMSE, MAE, VAR, SD = getPerformance(target, prediction)
    writeToResult(save_file, MSE, RMSE, MAE, VAR, SD)

    print(MSE, RMSE, MAE, VAR, SD)

    cv2.destroyAllWindows()
