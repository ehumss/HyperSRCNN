#! /usr/bin/python

import tensorflow as tf
import cv2
import numpy as np

def read_tfrecord(tf_filename, image_size):
    filename_queue = tf.train.string_input_producer([tf_filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature={
          'image_raw': tf.FixedLenFeature([image_size[0]*image_size[1]*image_size[2]], tf.float32),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(serialized_example, features=feature)
    image = features['image_raw']
    image = tf.reshape(image, image_size)
    return image

def build_img_pair(image, img_size):
    num_img = 1
    height  = img_size[0]
    width   = img_size[1]
    depth   = img_size[2]
    img_HR = \
        np.arange(num_img*height*width*depth,dtype = np.float32).reshape([num_img,height,width,depth])
    img_LR = \
        np.arange(num_img*height*width*depth,dtype = np.float32).reshape([num_img,height,width,depth])

    img_downscale = cv2.pyrDown(image)
    img_interpolation = \
        cv2.resize(img_downscale,None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
    img_HR[0,:,:,:] = image
    img_LR[0,:,:,:] = img_interpolation
    return img_HR[:,6:1018,6:1018,:], img_LR, img_LR[:,6:1018,6:1018,:]

def weight(shape,name):
	initial = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

if __name__ == '__main__':
    output_dir = './tfrecords'
    output_final_filename = 'final_test_data'

    image_height = 1024; image_width  = 1024; image_depth  = 31
    conv_height = 1012; conv_width  = 1012; conv_depth  = 31

    W1 = 9; H1 = 9; C1 = 32
    W2 = 1; H2 = 1; C2 = 32
    W3 = 5; H3 = 5; C3 = image_depth

    with tf.name_scope('input_data'):
    	img_LR = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth])
        img_LR_pad = tf.placeholder(tf.float32, [None, conv_height, conv_width, conv_depth])
        img_HR = tf.placeholder(tf.float32, [None, conv_height, conv_width, conv_depth])

    with tf.name_scope('conv_1_feature_extraction'):
    	W_conv1 = weight([W1,H1,image_depth,C1],name='weights')
    	h_conv1 = conv2d(img_LR, W_conv1)

    with tf.name_scope('conv_2_nonlinear_mapping'):
    	W_conv2 = weight([W2,H2,C1,C2],name='weights')
    	h_conv2 = conv2d(h_conv1, W_conv2)

    with tf.name_scope('conv_3_reconstruction'):
    	W_conv3 = weight([W3,H3,C2,C3],name='weights')
    	h_conv3 = conv2d(h_conv2, W_conv3)

    with tf.name_scope('evaluate'):
        MSE_loss = tf.reduce_mean(tf.square(img_HR-img_LR_pad-h_conv3))
        output_clip = tf.clip_by_value(img_LR_pad+h_conv3,0.,1.) * 255.
        input_clip  = tf.clip_by_value(img_HR,0.,1.) * 255.
        MSE = tf.reduce_mean(tf.square(output_clip-input_clip))
        PSNR = 10.*log10(255.*255./MSE)

    final_test_file  = output_dir +'/' + output_final_filename  + '.tfrecords'
    img_size = [image_height,image_width,image_depth]
    test_image = read_tfrecord(final_test_file, img_size)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './HySRCNN_logs/model.ckpt-8200')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        mean_PSNR = 0
        num = 1
        fobj = open('test_result.txt','w')
        for k in range(10):
            test_HR, test_LR, test_LR_pad = build_img_pair(sess.run(test_image),img_size)
            test_loss = MSE_loss.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
            test_PSNR = PSNR.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
            print('test image # %d: loss %f, PSNR %f' % (num,test_loss,test_PSNR))
            fobj.writelines(['test image # %d: loss %f, PSNR %f\n' % (num,test_loss,test_PSNR)])
            mean_PSNR = mean_PSNR + test_PSNR
            num = num + 1
        mean_PSNR = mean_PSNR / 10.
        print('average test PSNR: %f' % mean_PSNR)
        fobj.writelines(['average test PSNR: %f' % mean_PSNR])
        fobj.close()
        coord.request_stop()
        coord.join(threads)
