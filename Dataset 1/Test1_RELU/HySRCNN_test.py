#! /usr/bin/python

import tensorflow as tf
import cv2
import numpy as np

def weight(shape,name):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def bias(shape,name):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial,name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'VALID')

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

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
    height  = img_size[0]
    width   = img_size[1]
    depth   = img_size[2]
    img_norm = np.arange(height*width*depth,dtype = np.float32).\
        reshape([1,height,width,depth])
    img_inter_norm = np.arange(height*width*depth,dtype = np.float32).\
        reshape([1,height,width,depth])
    img_downscale = cv2.pyrDown(image)
    img_interpolation = \
        cv2.resize(img_downscale,None, fx = 2, fy = 2, interpolation=cv2.INTER_CUBIC)
    # normalize
    img_cast = image.astype(dtype = np.float32)
    max_element = np.amax(img_cast)
    min_element = np.amin(img_cast)
    img_inter_cast = img_interpolation.astype(dtype = np.float32)
    max_element_inter = np.amax(img_inter_cast)
    min_element_inter = np.amin(img_inter_cast)

    img_norm[0,:,:,:] = (img_cast - min_element) / (max_element - min_element)
    img_inter_norm[0,:,:,:] = \
        (img_inter_cast - min_element_inter) / (max_element_inter - min_element_inter)
    return img_norm[:,6:506,6:506,:], img_inter_norm, \
        np.array([[min_element], [max_element], [min_element_inter], [max_element_inter]])

if __name__ == '__main__':
    image_height = 512; conv_height = 500
    image_width  = 512; conv_width  = 500
    image_depth  = 31

    rootdir = './complete_ms_data'
    output_dir = './tfrecords'
    fine_test_filename  = 'final_test_data'

    W1 = 9; H1 = 9; C1 = 64
    W2 = 1; H2 = 1; C2 = 32
    W3 = 5; H3 = 5; C3 = image_depth

    with tf.name_scope('input_data'):
    	img_LR = tf.placeholder(tf.float32, [None, image_height, image_width, image_depth])
    	img_HR = tf.placeholder(tf.float32, [None, conv_height, conv_width, image_depth])
    	min_SR_image = tf.placeholder(tf.float32, [1])
    	max_SR_image = tf.placeholder(tf.float32, [1])
    	min_HR_image = tf.placeholder(tf.float32, [1])
    	max_HR_image = tf.placeholder(tf.float32, [1])

    with tf.name_scope('conv_1_feature_extraction'):
    	W_conv1 = weight([W1,H1,image_depth,C1],name='weights')
        b_conv1 = bias([C1],name='bias')
    	h_conv1 = tf.nn.relu(conv2d(img_LR, W_conv1) + b_conv1)

    with tf.name_scope('conv_2_nonlinear_mapping'):
    	W_conv2 = weight([W2,H2,C1,C2],name='weights')
        b_conv2 = bias([C2],name='bias')
    	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    with tf.name_scope('conv_3_reconstruction'):
    	W_conv3 = weight([W3,H3,C2,C3],name='weights')
        b_conv3 = bias([C3],name='bias')
    	h_conv3 = conv2d(h_conv2, W_conv3) + b_conv3

    with tf.name_scope('evaluate'):
        clip_SR_image = tf.clip_by_value(h_conv3,0.,1.) * (max_HR_image-min_HR_image) \
            +  min_HR_image
        cast_SR_image = tf.cast(clip_SR_image, tf.uint8)
        clip_HR_image = tf.clip_by_value(img_HR,0.,1.) * (max_HR_image-min_HR_image) \
            +  min_HR_image
        #reshape_SR_image = tf.reshape(cast_SR_image,[image_depth, conv_height, conv_width, 1])
        MSE_loss = tf.reduce_mean(tf.square(img_HR-h_conv3))
        MSE = tf.reduce_mean(tf.square(clip_SR_image-clip_HR_image))
        PSNR = 10.*log10(255.*255./MSE)
        #tf.summary.image('reconstructed_image', reshape_SR_image, max_outputs=image_depth)

    test_file  = output_dir +'/' + fine_test_filename  + '.tfrecords'
    img_size = [image_height,image_width,image_depth]
    test_batch = read_tfrecord(test_file, img_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './HySRCNN_logs/model.ckpt')
        #summary = tf.summary.merge_all()
        #final_test_writer = tf.summary.FileWriter('./HySRCNN_logs/final_model_test', sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_HR, test_LR, Range = build_img_pair(sess.run(test_batch), img_size)
        coord.request_stop()
        coord.join(threads)

        I = sess.run(cast_SR_image, \
                {img_HR:test_HR, img_LR:test_LR, min_HR_image:Range[0], max_HR_image:Range[1], min_SR_image:Range[2], max_SR_image:Range[3]})
        final_MSE_loss = sess.run(MSE_loss, \
                {img_HR:test_HR, img_LR:test_LR, min_HR_image:Range[0], max_HR_image:Range[1], min_SR_image:Range[2], max_SR_image:Range[3]})
        final_MSE  = sess.run(MSE, \
                {img_HR:test_HR, img_LR:test_LR, min_HR_image:Range[0], max_HR_image:Range[1], min_SR_image:Range[2], max_SR_image:Range[3]})
        final_PSNR = sess.run(PSNR, \
                {img_HR:test_HR, img_LR:test_LR, min_HR_image:Range[0], max_HR_image:Range[1], min_SR_image:Range[2], max_SR_image:Range[3]})
        print("final test loss %f" % final_MSE_loss)
        print("final MSE  %f" % final_MSE)
        print("final PSNR %f" % final_PSNR)

        #final_test_writer.add_summary(sess.run(summary, {img_LR:test_LR}))
        #final_test_writer.flush()

    cv2.imwrite('reconstruct_first_band.jpg',I[0,:,:,0])
    cv2.imwrite('reconstruct_last_band.jpg',I[0,:,:,30])
    fobj = open('test_result.txt','w')
    fobj.writelines(['final test loss %f \nfinal MSE %f \nfinal PSNR %f' \
        % (final_MSE_loss, final_MSE, final_PSNR)])
    fobj.close()
