#! /usr/bin/python

import cv2
import numpy as np
import scipy.io as sio
import os
import os.path
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def DHW2HWD(img_DHW):
    depth  = img_DHW.shape[0]
    height = img_DHW.shape[1]
    width  = img_DHW.shape[2]
    img_HWD = \
        np.arange(height*width*depth,dtype = np.float64).reshape([height,width,depth])
    for k in range(depth):
        img_HWD[:,:,k] = img_DHW[k,:,:]
    return img_HWD

def HyPrepareData(rootdir, output_dir, output_train_filename, output_test_filename, output_final_filename):
    sub_image_height = 32
    sub_image_width  = 32
    hyper_image_height = 1024
    hyper_image_width  = 1024
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    train_filename = output_dir + "/" + output_train_filename + '.tfrecords'
    test_filename  = output_dir + "/" + output_test_filename + '.tfrecords'
    final_test_filename  = output_dir + "/" + output_final_filename + '.tfrecords'
    train_writer = tf.python_io.TFRecordWriter(train_filename)
    test_writer  = tf.python_io.TFRecordWriter(test_filename)
    final_test_writer  = tf.python_io.TFRecordWriter(final_test_filename)

    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:
            if dirname == 'train':
                for parent, dirnames, filenames in os.walk(rootdir + '/train'):
                    train_num = 1
                    for filename in filenames:
                        data =  sio.loadmat(rootdir + '/train' + '/' + filename)
                        image = DHW2HWD(data['data'])
                        image = image.astype(dtype = np.float32)
                        for h in range(hyper_image_height/sub_image_height):
                            for w in range(hyper_image_width/sub_image_width):
                                sub_image = image[h*sub_image_height:(h+1)*sub_image_height,
                                                w*sub_image_width:(w+1)*sub_image_width, :]
                                feature={
                                    'image_raw': _float_feature(sub_image.reshape((sub_image_height*sub_image_width*31))),
                                    'height': _int64_feature(sub_image.shape[0]),
                                    'width':  _int64_feature(sub_image.shape[1]),
                                    'depth':  _int64_feature(sub_image.shape[2]),
                                }
                                example = tf.train.Example(features=tf.train.Features(feature=feature))
                                train_writer.write(example.SerializeToString())
                                print ("adding training subimage N0.%d, height = %d, width = %d, depth = %d"
                                        % (train_num, sub_image.shape[0], sub_image.shape[1], sub_image.shape[2]))
                                train_num = train_num + 1
                    train_writer.close()
            if dirname == 'test':
                for parent, dirnames, filenames in os.walk(rootdir + '/test'):
                    final_num = 1
                    test_num = 1
                    for filename in filenames:
                        data =  sio.loadmat(rootdir + '/test' + '/' + filename)
                        image = DHW2HWD(data['data'])
                        image = image.astype(dtype = np.float32)

                        feature={
                            'image_raw': _float_feature(image.reshape((hyper_image_height*hyper_image_width*31))),
                            'height': _int64_feature(image.shape[0]),
                            'width':  _int64_feature(image.shape[1]),
                            'depth':  _int64_feature(image.shape[2]),
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        final_test_writer.write(example.SerializeToString())
                        print ("adding test image N0.%d, height = %d, width = %d, depth = %d"
                                % (final_num, image.shape[0], image.shape[1], image.shape[2]))
                        final_num = final_num + 1

                        for h in range(hyper_image_height/sub_image_height):
                            for w in range(hyper_image_width/sub_image_width):
                                sub_image = image[h*sub_image_height:(h+1)*sub_image_height,
                                                w*sub_image_width:(w+1)*sub_image_width, :]
                                feature={
                                    'image_raw': _float_feature(sub_image.reshape((sub_image_height*sub_image_width*31))),
                                    'height': _int64_feature(sub_image.shape[0]),
                                    'width':  _int64_feature(sub_image.shape[1]),
                                    'depth':  _int64_feature(sub_image.shape[2]),
                                }
                                example = tf.train.Example(features=tf.train.Features(feature=feature))
                                test_writer.write(example.SerializeToString())
                                print ("adding test subimage N0.%d, height = %d, width = %d, depth = %d"
                                        % (test_num, sub_image.shape[0], sub_image.shape[1], sub_image.shape[2]))
                                test_num = test_num + 1
                    test_writer.close()
                    final_test_writer.close()

    #print ("add %d training subimages" % (train_num-1))
    #print ("add %d test subimages" % (test_num-1))
    #print ("add %d test images" % (final_num-1))

    return train_num,test_num,final_num
