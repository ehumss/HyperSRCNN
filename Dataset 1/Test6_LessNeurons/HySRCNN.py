#! /usr/bin/python

import tensorflow as tf
import cv2
import numpy as np
from HyPrepareData import *

def read_tfrecord(tf_filename, image_size):
    filename_queue = tf.train.string_input_producer([tf_filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    feature={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
    }
    features = tf.parse_single_example(serialized_example, features=feature)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, image_size)
    return image

def build_img_pair(image, img_size):
    num_img = image.shape[0]
    height  = img_size[0]
    width   = img_size[1]
    depth   = img_size[2]
    img_norm = \
        np.arange(num_img*height*width*depth,dtype = np.float32).reshape([num_img,height,width,depth])
    img_inter_norm = \
        np.arange(num_img*height*width*depth,dtype = np.float32).reshape([num_img,height,width,depth])
    for i in range(num_img):
        temp = image[i,:,:,:]
        img_downscale = cv2.pyrDown(temp)
        img_interpolation = \
            cv2.resize(img_downscale,None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
        # normalize
        img_cast = temp.astype(dtype = np.float32)
        max_element = np.amax(img_cast)
        min_element = np.amin(img_cast)
        img_inter_cast = img_interpolation.astype(dtype = np.float32)
        max_element_inter = np.amax(img_inter_cast)
        min_element_inter = np.amin(img_inter_cast)

        img_norm[i,:,:,:] = (img_cast - min_element) / (max_element - min_element)
        img_inter_norm[i,:,:,:] = \
            (img_inter_cast - min_element_inter) / (max_element_inter - min_element_inter)
    return img_norm[:,6:26,6:26,:], img_inter_norm, img_inter_norm[:,6:26,6:26,:]

def weight(shape,name):
	initial = tf.random_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def bias(shape,name):
	initial = tf.constant(0, shape=shape, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

if __name__ == '__main__':
    rootdir = './complete_ms_data'
    output_dir = './tfrecords'
    output_train_filename = 'train_data'
    output_test_filename  = 'test_data'
    output_final_filename = 'final_test_data'
    train_num,test_num,final_num = HyPrepareData(rootdir, output_dir,
        output_train_filename, output_test_filename, output_final_filename)

    image_height = 32; image_width  = 32; image_depth  = 31
    conv_height = 20; conv_width  = 20; conv_depth  = 31

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
    	# summaries
    	tf.summary.histogram('histogram_W1', W_conv1)

    with tf.name_scope('conv_2_nonlinear_mapping'):
    	W_conv2 = weight([W2,H2,C1,C2],name='weights')
    	h_conv2 = conv2d(h_conv1, W_conv2)
    	# summaries
    	tf.summary.histogram('histogram_W2', W_conv2)

    with tf.name_scope('conv_3_reconstruction'):
    	W_conv3 = weight([W3,H3,C2,C3],name='weights')
    	h_conv3 = conv2d(h_conv2, W_conv3)
    	# summaries
    	tf.summary.histogram('histogram_W3', W_conv3)

    with tf.name_scope('evaluate'):
        MSE_loss = tf.reduce_mean(tf.square(img_HR-img_LR_pad-h_conv3))
        output_clip = tf.clip_by_value(img_LR_pad+h_conv3,0.,1.) * 255.
        input_clip  = tf.clip_by_value(img_HR,0.,1.) * 255.
        MSE = tf.reduce_mean(tf.square(output_clip-input_clip))
        PSNR = 10.*log10(255.*255./MSE)
    	# summary
    	tf.summary.scalar('loss', MSE_loss)
        tf.summary.scalar('MSE', MSE)
        tf.summary.scalar('PSNR', PSNR)

    with tf.name_scope('training_op'):
    	optimizer = tf.train.AdamOptimizer()
    	train_step = optimizer.minimize(MSE_loss)

    batch_size = 300
    maxiter = 10000
    train_file = output_dir +'/' + output_train_filename + '.tfrecords'
    test_file  = output_dir +'/' + output_test_filename  + '.tfrecords'
    img_size = [image_height,image_width,image_depth]
    train_image = read_tfrecord(train_file, img_size)
    train_batch = tf.train.shuffle_batch([train_image],
                batch_size = batch_size,
                capacity = 1000 + 3 * batch_size,
                num_threads = 2,
                min_after_dequeue = 1000)
    test_batch = read_tfrecord(test_file, img_size)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./HySRCNN_logs/train', sess.graph)
        test_writer  = tf.summary.FileWriter('./HySRCNN_logs/test', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tup = ()
        for i in range(test_num):
            temp = sess.run(test_batch)
            tup = tup + tuple([temp])
        test_HR, test_LR, test_LR_pad = build_img_pair(np.stack(tup, axis = 0),img_size)

        for iter in range(maxiter):
            batch_HR, batch_LR, batch_LR_pad = build_img_pair(sess.run(train_batch),img_size)
            if iter%10 == 0:
                summary_train = sess.run(summary, {img_LR:batch_LR, img_HR:batch_HR, img_LR_pad:batch_LR_pad})
                train_writer.add_summary(summary_train, iter)
                train_writer.flush()
                summary_test = sess.run(summary, {img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
                test_writer.add_summary(summary_test, iter)
                test_writer.flush()
            if iter%100 == 0:
                train_loss  = MSE_loss.eval({img_LR:batch_LR, img_HR:batch_HR, img_LR_pad:batch_LR_pad})
                test_loss   = MSE_loss.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
                train_PSNR = PSNR.eval({img_LR:batch_LR, img_HR:batch_HR, img_LR_pad:batch_LR_pad})
                test_PSNR = PSNR.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
                print("iter step %d trainning batch loss %f"%(iter, train_loss))
                print("iter step %d test loss %f"%(iter, test_loss))
                print("iter step %d train PSNR %f"%(iter, train_PSNR))
                print("iter step %d test PSNR %f \n"%(iter, test_PSNR))
            if iter%100 == 0:
                saver.save(sess, './HySRCNN_logs/model.ckpt', global_step=iter)
            train_step.run({img_LR:batch_LR, img_HR:batch_HR, img_LR_pad:batch_LR_pad})
        coord.request_stop()
        coord.join(threads)


        saver.save(sess, './HySRCNN_logs/model.ckpt')
        test_loss = MSE_loss.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
        test_MSE  = MSE.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
        test_PSNR = PSNR.eval({img_LR:test_LR, img_HR:test_HR, img_LR_pad:test_LR_pad})
        print("final test loss %f" % test_loss)
        print("final test MSE %f" % test_MSE)
        print("final test PSNR %f" % test_PSNR)
