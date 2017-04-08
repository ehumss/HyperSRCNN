#! /usr/bin/python

import cv2
import numpy as np
import os
import os.path
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def HyPrepareData(rootdir, output_dir, output_train_filename, output_test_filename, output_final_filename):
    # merge various bands to form hyperspectral images
    train_dataset = []
    test_dataset  = []
    for parent, dirnames, filenames in os.walk(rootdir):
        for dirname in dirnames:
            if dirname == 'train':
                for parent, dirnames, filenames in os.walk(rootdir + '/' + 'train'):
                    num = 1
                    for dirname in dirnames:
                        for parent, dirnames, filenames in os.walk(rootdir + '/' + 'train/' + dirname):
                            tup = ()
                            for filename in filenames:
                                if filename[-3:-1]+filename[-1] == 'png':
                                    temp = cv2.imread(rootdir + '/' + 'train/' + dirname + '/' + filename)
                                    tup = tup + tuple([temp[:,:,0]])
                            img = np.stack(tup, axis = 2)
                        train_dataset.append(img)
                        print ("training image No.%d, height = %d, width = %d, depth = %d"
                                % (num, img.shape[0], img.shape[1], img.shape[2]))
                        num = num + 1
            if dirname == 'test':
                for parent, dirnames, filenames in os.walk(rootdir + '/' + 'test'):
                    num = 1
                    for dirname in dirnames:
                        for parent, dirnames, filenames in os.walk(rootdir + '/' + 'test/' + dirname):
                            tup = ()
                            for filename in filenames:
                                if filename[-3:-1]+filename[-1] == 'png':
                                    temp = cv2.imread(rootdir + '/' + 'test/' + dirname + '/' + filename)
                                    tup = tup + tuple([temp[:,:,0]])
                            img = np.stack(tup, axis = 2)
                        test_dataset.append(img)
                        print ("test image No.%d, height = %d, width = %d, depth = %d"
                                % (num, img.shape[0], img.shape[1], img.shape[2]))
                        num = num + 1

    # build TFRecords file
    sub_image_height = 32
    sub_image_width  = 32
    hyper_image_height = train_dataset[0].shape[0]
    hyper_image_width  = train_dataset[0].shape[1]
    if not os.path.exists(output_dir) or os.path.isfile(output_dir):
        os.makedirs(output_dir)
    train_filename = output_dir + "/" + output_train_filename + '.tfrecords'
    test_filename  = output_dir + "/" + output_test_filename + '.tfrecords'
    final_test_filename  = output_dir + "/" + output_final_filename + '.tfrecords'
    train_writer = tf.python_io.TFRecordWriter(train_filename)
    test_writer  = tf.python_io.TFRecordWriter(test_filename)
    final_test_writer  = tf.python_io.TFRecordWriter(final_test_filename)

    train_num = 0
    for image in train_dataset:
        for h in range(hyper_image_height/sub_image_height):
            for w in range(hyper_image_width/sub_image_width):
                sub_image = image[h*sub_image_height:(h+1)*sub_image_height,
                                w*sub_image_width:(w+1)*sub_image_width]
                image_raw = sub_image.tostring()
                feature={
                    'image_raw': _bytes_feature(image_raw),
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

    test_num = 0
    for image in test_dataset:
        for h in range(hyper_image_height/sub_image_height):
            for w in range(hyper_image_width/sub_image_width):
                sub_image = image[h*sub_image_height:(h+1)*sub_image_height,
                                w*sub_image_width:(w+1)*sub_image_width]
                image_raw = sub_image.tostring()
                feature={
                    'image_raw': _bytes_feature(image_raw),
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

    final_num = 0
    for image in test_dataset:
        image_raw = image.tostring()
        feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width':  _int64_feature(image.shape[1]),
            'depth':  _int64_feature(image.shape[2]),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        final_test_writer.write(example.SerializeToString())
        print ("adding test image N0.%d, height = %d, width = %d, depth = %d"
                % (final_num, image.shape[0], image.shape[1], image.shape[2]))
        final_num = final_num + 1

    print ("add %d training subimages" % train_num)
    print ("add %d test subimages" % test_num)
    print ("add %d test images" % final_num)

    return train_num,test_num,final_num
