import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import xlwt
import random
import numpy as np

import tensorflow as tf
slim = tf.contrib.slim
import datetime
from PIL import Image
from keras.utils import to_categorical
# im
from sklearn import svm
from sklearn.model_selection import cross_val_score

# label-sublocation
sublocations=['Golgi+apparatus','Mitochondrion','Vesicles','Endoplasmic+reticulum'
             ,'Nucleolus','Nucleus','Cytoskeleton']
fold_name = '1_fold'
now_time = str(datetime.datetime.now())
date = now_time.split(' ')[0]
clork = now_time.split(' ')[-1]
hour = clork.split(':')[0]
min = clork.split(':')[1]

model_path_last = '~/model_saving/AlexNet/' + date + '_' + hour + '.' + min
# model_path = model_path_last+'/osteoporosis_classifier'
model_path=model_path_last+'/proteinModel'
print('model_save_path:', model_path)

def load_data_list():
    X_train_list = []
    Y_train_list = []
    dir = 'E:/hpaData/del_notGood/all/'
    # dir = '/home/linghe/dataset/train1/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        # label = int(image_name.split('_')[1])#orginal
        #label=sublocation
        label = ''
        sublocation=image_name.split('_')[2][:-1]
        for i in range(len(sublocations)):
            if(sublocation==sublocations[i]):
                label=i
        # label = image_name.split('_')[2][:-1]
        X_train_list.append(image_path)
        Y_train_list.append(label)

    X_test_list = []
    Y_test_list = []
    dir = './Data_python/AlexNet/data_fold_crop_aug_CLAHE/'+fold_name+'/test1/'
    # dir = '/home/linghe/dataset/test1/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        # label = int(image_name.split('_')[1])
        sublocation = image_name.split('_')[2][:-1]
        for i in range(len(sublocations)):
            if (sublocation == sublocations[i]):
                label = i
        X_test_list.append(image_path)
        Y_test_list.append(label)

    # X_all_list = []
    # Y_all_list = []
    # dir = './Data_python/AlexNet/data_fold_crop_aug_CLAHE/all/'
    # for first_path_name in os.listdir(dir):
    #     first_path = dir+first_path_name
    #     for image_name in os.listdir(first_path):
    #         image_path = first_path +'/'+ image_name
    #         # label = int(image_name.split('_')[1])
    #         sublocation = image_name.split('_')[2][:-1]
    #         for i in range(len(sublocations)):
    #             if (sublocation == sublocations[i]):
    #                 label = i
    #         X_all_list.append(image_path)
    #         Y_all_list.append(label)

    # return X_train_list, Y_train_list, X_test_list, Y_test_list, X_all_list, Y_all_list
    return X_train_list, Y_train_list, X_test_list, Y_test_list

# X_train_list, Y_train_list, X_test_list, Y_test_list, X_all_list, Y_all_list = load_data_list()
X_train_list, Y_train_list, X_test_list, Y_test_list = load_data_list()
train_size = len(X_train_list)
test_size = len(X_test_list)
# all_size = len(X_all_list)
classes = 7

# net input
train_batch_size = 19
# test_batch_size = 23
test_batch_size = 48
# all_batch_size = 58
epochs = 200
# batch_size = 19
# batch_size = 16
batch_size = 128
keep_prob_value = 1.0
lr_value = 0.01
# lr_value = 0.005
lambda_1 = 0.0005
bias_value = 0.0
norm_value_1 = 1
norm_value_2 = 1
lrn_bias = 1

print('train_size: ', train_size)
print('test_size: ', test_size)
# print('all_size: ', all_size)
print('classes: ', classes)
print('epochs:',epochs)



# normlization
def norm(x,size):
    return tf.nn.lrn(x,size,bias=lrn_bias,alpha=0.001/0.9,beta=0.75)

def get_weight_conv(name,shape,regularizer):
    kernel = tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
    if regularizer == True:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer( )(kernel))
    return kernel

def get_weight_fc(name,shape,regularizer):
    kernel = tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    if regularizer == True:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_1)(kernel))
    return kernel

def conv_op(input_op, name, kh, kw, n_out, dh, dw, bn, training, regularizer):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = get_weight_conv(scope+'w',[kh, kw, n_in, n_out],regularizer)
        x = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        if bn == True:
            x = tf.layers.batch_normalization(x, axis=3, training=training)
        else:
            bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            x = tf.nn.bias_add(x, biases)
        activation = tf.nn.relu(x, name=scope)
        return activation

def fc_op(input_op, name, n_out, regularizer):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = get_weight_fc(scope + 'w', [n_in, n_out], regularizer)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        # tf.nn.relu_layer  input_op  kernel bias,activation
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def next_batch(xs, ys, batch_size, start, size):
    start = start % round(size / batch_size)
    start = start * batch_size
    end = start + batch_size
    batch_xs_list = xs[start:end]
    batch_ys_list = ys[start:end]
    batch_xs = []
    for i in range(len(batch_xs_list)):
        image = Image.open(batch_xs_list[i])
        image=image.resize((299,299))
        # arr = np.array(image)
        # print(arr.shape)
        image_narray = np.asarray(image, dtype='float32')
        image_narray = image_narray * 1.0 / 255
        batch_xs.append(image_narray)
    batch_xs = np.stack(batch_xs)
    batch_xs = np.expand_dims(batch_xs, 3)
    batch_ys = to_categorical(batch_ys_list, classes)
    return batch_xs, batch_ys


def xception(inputs,
             num_classes=7,
             is_training=True,
             scope='xception'):
    '''
    The Xception Model!

    Note:
    The padding is included by default in slim.conv2d to preserve spatial dimensions.

    INPUTS:
    - inputs(Tensor): a 4D Tensor input of shape [batch_size, height, width, num_channels]
    - num_classes(int): the number of classes to predict
    - is_training(bool): Whether or not to train

    OUTPUTS:
    - logits (Tensor): raw, unactivated outputs of the final layer
    - end_points(dict): dictionary containing the outputs for each layer, including the 'Predictions'
                        containing the probabilities of each output.
    '''
    with tf.variable_scope('Xception') as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d],
                            outputs_collections=[end_points_collection]), \
             slim.arg_scope([slim.batch_norm], is_training=is_training):
            # ===========ENTRY FLOW==============
            # Block 1
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='valid', scope='block1_conv1')
            net = slim.batch_norm(net, scope='block1_bn1')
            net = tf.nn.relu(net, name='block1_relu1')
            net = slim.conv2d(net, 64, [3, 3], padding='valid', scope='block1_conv2')
            net = slim.batch_norm(net, scope='block1_bn2')
            net = tf.nn.relu(net, name='block1_relu2')
            residual = slim.conv2d(net, 128, [1, 1], stride=2, scope='block1_res_conv')
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            # Block 2
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv1')
            net = slim.batch_norm(net, scope='block2_bn1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv2')
            net = slim.batch_norm(net, scope='block2_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block2_max_pool')
            net = tf.add(net, residual, name='block2_add')
            residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
            residual = slim.batch_norm(residual, scope='block2_res_bn')

            # Block 3
            net = tf.nn.relu(net, name='block3_relu1')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv1')
            net = slim.batch_norm(net, scope='block3_bn1')
            net = tf.nn.relu(net, name='block3_relu2')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv2')
            net = slim.batch_norm(net, scope='block3_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
            net = tf.add(net, residual, name='block3_add')
            residual = slim.conv2d(net, 728, [1, 1], stride=2, scope='block3_res_conv')
            residual = slim.batch_norm(residual, scope='block3_res_bn')

            # Block 4
            net = tf.nn.relu(net, name='block4_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv1')
            net = slim.batch_norm(net, scope='block4_bn1')
            net = tf.nn.relu(net, name='block4_relu2')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv2')
            net = slim.batch_norm(net, scope='block4_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block4_max_pool')
            net = tf.add(net, residual, name='block4_add')

            # ===========MIDDLE FLOW===============
            for i in range(8):
                block_prefix = 'block%s_' % (str(i + 5))

                residual = net
                net = tf.nn.relu(net, name=block_prefix + 'relu1')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv1')
                net = slim.batch_norm(net, scope=block_prefix + 'bn1')
                net = tf.nn.relu(net, name=block_prefix + 'relu2')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv2')
                net = slim.batch_norm(net, scope=block_prefix + 'bn2')
                net = tf.nn.relu(net, name=block_prefix + 'relu3')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv3')
                net = slim.batch_norm(net, scope=block_prefix + 'bn3')
                net = tf.add(net, residual, name=block_prefix + 'add')

            # ========EXIT FLOW============
            residual = slim.conv2d(net, 1024, [1, 1], stride=2, scope='block12_res_conv')
            residual = slim.batch_norm(residual, scope='block12_res_bn')
            net = tf.nn.relu(net, name='block13_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block13_dws_conv1')
            net = slim.batch_norm(net, scope='block13_bn1')
            net = tf.nn.relu(net, name='block13_relu2')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block13_dws_conv2')
            net = slim.batch_norm(net, scope='block13_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block13_max_pool')
            net = tf.add(net, residual, name='block13_add')

            net = slim.separable_conv2d(net, 1536, [3, 3], scope='block14_dws_conv1')
            net = slim.batch_norm(net, scope='block14_bn1')
            net = tf.nn.relu(net, name='block14_relu1')
            net = slim.separable_conv2d(net, 2048, [3, 3], scope='block14_dws_conv2')
            net = slim.batch_norm(net, scope='block14_bn2')
            net = tf.nn.relu(net, name='block14_relu2')
            net = slim.avg_pool2d(net, [7, 7], scope='block15_avg_pool')
            # net = slim.avg_pool2d(net, [10, 10], scope='block15_avg_pool')
            # Replace FC layer with conv layer instead
        #     net = slim.conv2d(net, 2048, [1, 1], scope='block15_conv1')
        #     logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='block15_conv2')
        #     logits = tf.squeeze(logits, [1, 2], name='block15_logits')  # Squeeze height and width only
        #     predictions = slim.softmax(logits, scope='Predictions')
        #
        #     end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        #     end_points['Logits'] = logits
        #     end_points['Predictions'] = predictions
        #
        #
        # return  end_points,logits
            fc_128 = tf.layers.dense(net, 128, activation=tf.nn.relu)
            out = tf.layers.dense(fc_128,2,activation=None)
            # out = tf.layers.dense(fc_128,classes,activation=None)
            #out = tf.layers.dense(fc_128, classes, activation=tf.nn.sigmoid)
            out = tf.squeeze(out, name='block15_logits')
        return fc_128, out


def train(X_train_list,Y_train_list,lr_value):
    if not os.path.exists(model_path_last):
        os.makedirs(model_path_last)
    # intialization********************************
    features = tf.placeholder(tf.float32, [None, 299, 299, 1])
    labels = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    fc2,logits = xception(features,classes,True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # tf.add_to_collection('losses', cross_entropy)
    # loss = tf.add_n(tf.get_collection('losses'))

    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdadeltaOptimizer(lr_value).minimize(cross_entropy)


    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start to train')
        for it in range(epochs):
            # shuffle
            c = list(zip(X_train_list, Y_train_list))
            random.Random(random.randint(0,10000)).shuffle(c)
            X_train_list, Y_train_list = zip(*c)
            if train_size % batch_size != 0:
                for i in range(int(train_size / batch_size)):
                    batch_xs, batch_ys = next_batch(X_train_list, Y_train_list, batch_size, i, train_size)
                    train_step.run(
                        feed_dict={features: batch_xs, labels: batch_ys, keep_prob:keep_prob_value, training:True})
                    if (i + 1) % 1 == 0:
                        train_cost = sess.run(cross_entropy, feed_dict={features: batch_xs,
                                                                        labels: batch_ys, keep_prob:keep_prob_value, training:True})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              '  step %d, training loss %g' % (i + 1, train_cost))
            else:
                print('Wrong set! Change batch_size!')

            right_all = 0
            if test_size % test_batch_size == 0:
                for x in range(int(test_size/test_batch_size)):
                    batch_test_xs, batch_test_ys = next_batch(X_test_list, Y_test_list, test_batch_size, x, test_size)
                    y_pre = sess.run(logits, feed_dict={features: batch_test_xs, keep_prob: 1.0, training:False})
                    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(batch_test_ys, 1))
                    right_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(right_num, feed_dict={features: batch_test_xs, labels: batch_test_ys, keep_prob: 1.0, training:False})
                    right_all += result
                acc = right_all/test_size
                now_time = str(datetime.datetime.now())
                print(now_time, ' epoch_%s' % (it+1), '  test_acc %g' % (acc))
                # if acc>0.55:
                if acc>0.5:
                    saver.save(sess, model_path,global_step=it+1)
                    print('save epoch',it+1)
            else:
                print('Wrong set! Change test_batch_size!')
        print('Finished training!')

    sess.close()

def comput_right(logits, labels):

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    right_num = tf.reduce_sum(correct_prediction)
    return right_num

def evaluate(X_test_list, Y_test_list,model_path):

    # restore_model_path = './model_saving/Vgg_softmax/2018-08-26_10.36/osteoporosis_classifier'
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 299, 299, 1])
    y_ = tf.placeholder(tf.float32, [None, classes])
    training = tf.placeholder(tf.bool)

    fc2,logits = xception(x,7.0,False)
    right_number = comput_right(logits, y_)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Start to restore model')
        saver.restore(sess,model_path)
# './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

        print('Start to evaluate test data')
        # if all_size % all_batch_size == 0:
        if test_size % test_batch_size == 0:
            all_features = []
            # for i in range(int(all_size / all_batch_size)):
            for i in range(int(test_size / test_batch_size)):
                # batch_xs, batch_ys = next_batch(X_all_list, Y_all_list, all_batch_size, i, all_size)
                batch_xs, batch_ys = next_batch(X_test_list, Y_test_list, test_batch_size, i, test_size)
                logits_feature = sess.run(fc2,
                                          feed_dict={x: batch_xs, y_: batch_ys,
                                                     training: False})
                all_features.append((logits_feature))

                patch_right_num = sess.run(right_number,
                                           feed_dict={x: batch_xs, y_: batch_ys, training: False})
                # print(i, '/', int(all_size / all_batch_size), '   ', patch_right_num)
                print(i, '/', int(test_size / test_batch_size), '   ', patch_right_num)
            all_features = np.array(all_features)
            all_features = all_features.reshape([-1, 128])
            print(all_features.shape[1],all_features.shape[0])
            book = xlwt.Workbook()
            sheet1 = book.add_sheet(u'sheet1', cell_overwrite_ok=True)
            for i in range(all_features.shape[0]):
                sheet1.write(i, 0, Y_test_list[i])
                for j in range(all_features.shape[1]+1):
                    sheet1.write(i, j+1, float(all_features[i][j]))

            book.save('./alexnet_feature.xls')

    sess.close()

def svm_classifier(restore_model_path):
    # restore_model_path = './model_saving/Vgg_softmax/2018-08-26_10.36/osteoporosis_classifier'
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y_ = tf.placeholder(tf.float32, [None, classes])

    logits = xception(x, False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Start to restore model')
        saver.restore(sess,
                      restore_model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

        print('Start to extract features')
        all_features = []
        # if all_size % all_batch_size == 0:
        if test_size % test_batch_size == 0:
            # for i in range(int(all_size / all_batch_size)):
            for i in range(int(test_size / test_batch_size)):
                # all_batch_xs, all_batch_ys = next_batch(X_all_list, Y_all_list, all_batch_size, i, all_size)
                all_batch_xs, all_batch_ys = next_batch(X_test_list, Y_test_list, test_batch_size, i, test_size)
                logits_feature = sess.run(logits,
                                          feed_dict={x: all_batch_xs, y_: all_batch_ys})
                all_features.append((logits_feature))
            all_features = np.array(all_features)
        else:
            print('Wrong set! Change all_batch_size!')

    sess.close()

    all_features = all_features.reshape([-1, 2])
    print(all_features.shape)

    # labels = np.arange(all_size)
    labels = np.arange(test_size)
    # for i in range(all_size):
    for i in range(test_size):
        if i < test_size/2:
        # if i < all_size/2:
            labels[i] = 0
        else:
            labels[i] = 1

    print('start to classifer with svm')
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, all_features, labels, cv=10)
    print('scores :', scores)
    acc = scores.mean()
    print('acc: ', acc)

# train(X_train_list,Y_train_list,lr_value)
# evaluate(X_train_list, Y_train_list,'E:/myProgram/python/deepLearning/model/model_saving/2019-03-12_15.21/proteinModel-45')
# evaluate(X_test_list, Y_test_list,'E:/myProgram/python/deepLearning/model/model_saving/AlexNet/2018-10-29_19.53/proteinModel-1')
# E:\myProgram\python\deepLearning\model\model_saving
# svm_classifier('./model_saving/Vgg_softmax/2018-08-29_15.16/osteoporosis_classifier')
def readLabel(Y_train_list,X_train_list):
    for i in range(len(Y_test_list)):
        print(Y_test_list[i],X_test_list[i])
readLabel(Y_test_list,X_test_list)


