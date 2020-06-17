
from PIL import Image
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import xlwt
import random
import datetime
from keras.utils import to_categorical
from sklearn import svm
from sklearn.model_selection import cross_val_score
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# label-sublocation
sublocations=['Golgi+apparatus','Mitochondrion','Vesicles','Endoplasmic+reticulum'
             ,'Nucleolus','Nucleus','Cytoskeleton']

fold_name = '1_fold'

now_time = str(datetime.datetime.now())
date = now_time.split(' ')[0]
clork = now_time.split(' ')[-1]
hour = clork.split(':')[0]
min = clork.split(':')[1]

model_path_last = '~/model_saving/VggNet/' + date + '_' + hour + '.' + min
model_path = model_path_last+'/proteinModel'
# model_path = './model_saving/'
print('model_save_path:', model_path)


def load_data_list():
    X_train_list = []
    Y_train_list = []
    # dir = './Data_python/Vgg_Res/Vgg_Res_data_8/data_fold_aug_CLAHE_4/'+fold_name+'/train/'
    dir = './Data_python/AlexNet//data_fold_crop_aug_CLAHE/'+fold_name+'/train1/'
    # dir = '/home/linghe/dataset/train1/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        label = ''
        sublocation = image_name.split('_')[2][:-1]
        for i in range(len(sublocations)):
            if (sublocation == sublocations[i]):
                label = i
        X_train_list.append(image_path)
        Y_train_list.append(label)

    X_test_list = []
    Y_test_list = []
    # dir = './Data_python/Vgg_Res/Vgg_Res_data_8/data_fold_aug_CLAHE_4/'+fold_name+'/test/'
    dir = './Data_python/AlexNet//data_fold_crop_aug_CLAHE/'+fold_name+'/test1/'
    # dir = '/home/linghe/dataset/test1/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        sublocation = image_name.split('_')[2][:-1]
        for i in range(len(sublocations)):
            if (sublocation == sublocations[i]):
                label = i
        X_test_list.append(image_path)
        Y_test_list.append(label)

    # X_all_list = []
    # Y_all_list = []
    # dir = './Data_python/Vgg_Res/Vgg_Res_data_8/data_fold_aug_CLAHE_4/all/'
    # for first_path_name in os.listdir(dir):
    #     first_path = dir+first_path_name
    #     for image_name in os.listdir(first_path):
    #         image_path = first_path +'/'+ image_name
    #         label = image_name.split('_')[2][:-5]
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
# train_batch_size = 160
train_batch_size = 128
# test_batch_size = 50

test_batch_size = 48
# all_batch_size = 58
epochs = 100
# epochs = 100
# batch_size = 32
batch_size = 32
keep_prob_value = 1.0
lr_value = 0.01
# lr_value = 0.001
lambda_1 = 0.0005
bias_value = 0.0

print('train_size: ', train_size)
print('test_size: ', test_size)
# print('all_size: ', all_size)
print('classes: ', classes)
print('epochs:',epochs)

def get_weight_conv(name, shape, regularizer):
    kernel = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # kernel = tf.get_variable(name, shape, dtype=tf.float32,
    #                          initializer=tf.contrib.layers.variance_scaling_initializer())
    if regularizer == True:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_1)(kernel))
    return kernel


def get_weight_fc(name, shape, regularizer):
    kernel = tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    # kernel = tf.get_variable(name, shape, dtype=tf.float32,
    #                          initializer=tf.contrib.layers.variance_scaling_initializer())
    if regularizer == True:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_1)(kernel))
    return kernel


# con
def conv_op(input_op, name, kh, kw, n_out, dh, dw, bn, training, regularizer):
    input_op = tf.convert_to_tensor(input_op)
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # kernel = tf.get_variable(scope + "w",
        #                          shape=[kh, kw, n_in, n_out],
        #                          dtype=tf.float32,
        #                          initializer=tf.contrib.layers.xavier_initializer_conv2d())
        kernel = get_weight_conv(scope + 'w', [kh, kw, n_in, n_out], regularizer)
        x = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        if bn == True:
            x = tf.layers.batch_normalization(x, axis=3, training=training)
        else:
            bias_init_val = tf.constant(bias_value, shape=[n_out], dtype=tf.float32)
            biases = tf.Variable(bias_init_val, trainable=True, name='b')
            x = tf.nn.bias_add(x, biases)
        activation = tf.nn.leaky_relu(x, name=scope)
        return activation


# connection
def fc_op(input_op, name, n_out, regularizer):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # kernel = tf.get_variable(scope + 'w',
        #                          shape=[n_in, n_out],
        #                          dtype=tf.float32,
        #                          initializer=tf.contrib.layers.xavier_initializer())
        kernel = get_weight_fc(scope + 'w', [n_in, n_out], regularizer)
        biases = tf.Variable(tf.constant(bias_value, shape=[n_out], dtype=tf.float32), name='b')
        #
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation


# pool
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


# net structure
def VggNet(input_op, keep_prob, training):
    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, bn=True, training=training,
                      regularizer=False)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, bn=True, training=training,
                      regularizer=False)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, bn=True, training=training,
                      regularizer=False)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=True, training=training,
                      regularizer=False)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=True, training=training,
                      regularizer=False)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, bn=False, training=training,
                      regularizer=False)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1024, regularizer=False)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=128, regularizer=False)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # logits = fc_op(fc7_drop, name="fc8", n_out=2, regularizer=False)
    logits = fc_op(fc7_drop, name="fc8", n_out=classes, regularizer=False)
    return fc7,logits


def next_batch(xs, ys, batch_size, start, size):
    start = start % round(size / batch_size)
    start = start * batch_size
    end = start + batch_size
    batch_xs_list = xs[start:end]
    batch_ys_list = ys[start:end]
    batch_xs = []
    for i in range(len(batch_xs_list)):
        image = Image.open(batch_xs_list[i])
        image = image.resize((224,224))
        image_narray = np.asarray(image, dtype='float32')
        image_narray = image_narray * 1.0 / 255
        batch_xs.append(image_narray)
    batch_xs = np.stack(batch_xs)
    batch_xs = np.expand_dims(batch_xs, 3)
    batch_ys = to_categorical(batch_ys_list, classes)
    return batch_xs, batch_ys


def comput_right(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    right_num = tf.reduce_sum(correct_prediction)
    return right_num


def train(X_train_list, Y_train_list, X_test_list, Y_test_list):
    if not os.path.exists(model_path_last):
        os.makedirs(model_path_last)
    features = tf.placeholder(tf.float32, [None, 224, 224, 1])
    labels = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    step_ = tf.placeholder(tf.int8)

    fc7, logits = VggNet(features, keep_prob, training)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(lr_value, step_, 10, 0.1, staircase=True)
    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start to train')
        for it in range(epochs):
            # shuffle
            c = list(zip(X_train_list, Y_train_list))
            random.Random(random.randint(0, 10000)).shuffle(c)
            X_train_list, Y_train_list = zip(*c)

            if train_size % batch_size != 0:
                for i in range(int(train_size / batch_size)):
                    batch_xs, batch_ys = next_batch(X_train_list, Y_train_list, batch_size, i, train_size)
                    # print(batch_xs)
                    train_step.run(
                        feed_dict={step_:it+1,features: batch_xs, labels: batch_ys, keep_prob: keep_prob_value, training: True})
                    if (i + 1) % 100 == 0:
                        train_cost = sess.run(cross_entropy, feed_dict={step_:it+1,features: batch_xs,
                                                                        labels: batch_ys, keep_prob: 1.0,
                                                                        training: True})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              '  step %d, training loss %g' % (i + 1, train_cost))
            else:
                print('Wrong set, change batch_size!')

            if test_size % test_batch_size == 0:
                right_all = 0
                for x in range(int(test_size / test_batch_size)):
                    batch_test_xs, batch_test_ys = next_batch(X_test_list, Y_test_list, test_batch_size, x, test_size)
                    y_pre = sess.run(logits, feed_dict={features: batch_test_xs, keep_prob: 1.0, training: False})
                    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(batch_test_ys, 1))
                    right_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(right_num,
                                      feed_dict={features: batch_test_xs, labels: batch_test_ys, keep_prob: 1.0,
                                                 training: False})
                    right_all += result
                    if x == int(test_size / test_batch_size)-1:
                        test_cost = sess.run(cross_entropy, feed_dict={step_: it + 1, features: batch_test_xs,
                                                                        labels: batch_test_ys, keep_prob: 1.0,
                                                                        training: False})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              'test_loss %g' % (test_cost))
                acc = right_all / test_size
                now_time = str(datetime.datetime.now())
                print(now_time, ' epoch_%s' % (it + 1), '  test_acc %g' % (acc))
                if acc > 0.5:
                    saver.save(sess, model_path, global_step=it + 1)
                    print('save epoch', it + 1)
            else:
                print('Wrong set, change test_batch_size!')

        print('Finished training!')
        sess.close()

def readLabel(Y_train_list):
    print(Y_train_list)

def evaluate(X_train_list, Y_train_list, X_all_list, Y_all_list, evaluate_train, model_path):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 224, 224, 1])
    y_ = tf.placeholder(tf.float32, [None, classes])
    training = tf.placeholder(tf.bool)

    fc7, logits = VggNet(x, 1.0, False)
    right_number = comput_right(logits, y_)

    saver = tf.train.Saver()
    train_features = []
    test_features = []
    with tf.Session() as sess:
        print('Start to restore model')
        saver.restore(sess,
                      model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

        if evaluate_train == True:
            print('Start to evaluate train data')
            train_patch_right_num_all = 0
            # original_right_all = 0
            if train_size % train_batch_size == 0:
                for i in range(int(train_size / train_batch_size)):
                    batch_xs, batch_ys = next_batch(X_train_list, Y_train_list, train_batch_size, i, train_size)
                    #*************
                    logits_feature = sess.run(fc7,
                                              feed_dict={x: batch_xs, y_: batch_ys,
                                                         training: False})
                    train_features.append((logits_feature))
                    # **********************
                    right_num = sess.run(right_number,
                                         feed_dict={x: batch_xs, y_: batch_ys, training: False})
                    print(i, '/', int(train_size / train_batch_size), right_num)

                    train_patch_right_num_all += right_num

                print('patch_right_num_all:', train_patch_right_num_all)
                train_patch_acc = train_patch_right_num_all / train_size
                print('patch: train', '%s accuracy', train_patch_acc)
                # **********************
                train_features = np.array(train_features)
                train_features = train_features.reshape([-1, 128])
                book = xlwt.Workbook()
                sheet1 = book.add_sheet(u'sheet1', cell_overwrite_ok=True)
                for i in range(train_features.shape[0]):
                    for j in range(train_features.shape[1]):
                        sheet1.write(i, j, float(train_features[i][j]))
                book.save('./vggnet_featureTrain.xls')
                # **********************
        print('Start to evaluate test data')
        test_patch_right_num_all = 0
        original_right_all = 0
        tp = 0
        tn = 0
        if test_size % test_batch_size == 0:

            for i in range(int(test_size / test_batch_size)):
                batch_xs, batch_ys = next_batch(X_all_list, Y_all_list, test_batch_size, i, test_size)
                logits_feature = sess.run(fc7,
                                          feed_dict={x: batch_xs, y_: batch_ys,
                                                     training: False})

                test_features.append((logits_feature))

                patch_right_num = sess.run(right_number,
                                           feed_dict={x: batch_xs, y_: batch_ys, training: False})
                print(i, '/', int(test_size / test_batch_size), '   ', patch_right_num)
                # if patch_right_num > all_batch_size / 2:
                #     original_right_all += 1
                #     if i < int((test_size / test_batch_size) / 2):
                #         tn += 1
                #     else:
                #         tp += 1
                # test_patch_right_num_all += patch_right_num
            test_features = np.array(test_features)
            test_features = test_features.reshape([-1, 128])
            book = xlwt.Workbook()
            sheet1 = book.add_sheet(u'sheet1', cell_overwrite_ok=True)
            for i in range(test_features.shape[0]):
                for j in range(test_features.shape[1]):
                    sheet1.write(i, j, float(test_features[i][j]))
            book.save('./vggnet_featureTest.xls')
        # if test_size % test_batch_size == 0:
        #     for i in range(int(test_size / test_batch_size)):
        #         batch_xs, batch_ys = next_batch(X_test_list, Y_test_list, test_batch_size, i, test_size)
        #         patch_right_num = sess.run(right_number,
        #                                    feed_dict={x: batch_xs, y_: batch_ys, training: False})
        #         print(i, '/', int(test_size / test_batch_size), '   ', patch_right_num)
        #         if patch_right_num > test_batch_size / 2:
        #             original_right_all += 1
        #             if i < int((test_size / test_batch_size) / 2):
        #                 tn += 1
        #             else:
        #                 tp += 1
        #         test_patch_right_num_all += patch_right_num
        #     fp = int((test_size / test_batch_size) / 2) - tp
        #     fn = int((test_size / test_batch_size) / 2) - tn
        #
        #     PPV = tp / (tp + fp)
        #     TPR = tp / (tp + fn)
        #     F1_score = 2 * tp / (2 * tp + fp + fn)
        #
        #     print('test_patch_right_num_all:', test_patch_right_num_all)
        #     print('original_right_all:', original_right_all)
        #     patch_acc = test_patch_right_num_all / test_size
        #     original_acc = original_right_all / (test_size / test_batch_size)
        #     print('test_patch_accuracy', patch_acc)
        #     print('test_original_accuracy', original_acc)
        #     print('PPV:', PPV)
        #     print('TPR:', TPR)
        #     print('F1_score:', F1_score)
    sess.close()


def svm_classifier_split(X_train_list, Y_train_list, X_test_list, Y_test_list, restore_model_path):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 224, 224, 1])
    y_ = tf.placeholder(tf.float32, [None, classes])
    training = tf.placeholder(tf.bool)

    logits = VggNet(x, 1.0, False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Start to restore model')
        saver.restore(sess,
                      restore_model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

        print('Start to extract train features with logits')
        train_features = []
        for i in range(int(train_size / train_batch_size)):
            train_batch_xs, train_batch_ys = next_batch(X_train_list, Y_train_list, train_batch_size, i, train_size)
            logits_feature = sess.run(logits,
                                      feed_dict={x: train_batch_xs, y_: train_batch_ys,
                                                 training: False})
            train_features.append((logits_feature))
        train_features = np.array(train_features)

        print('Start to extract test features with logits')
        test_features = []
        for i in range(int(test_size / test_batch_size)):
            test_batch_xs, test_batch_ys = next_batch(X_test_list, Y_test_list, test_batch_size, i,
                                                      test_size)
            logits_feature = sess.run(logits,
                                      feed_dict={x: test_batch_xs, y_: test_batch_ys,
                                                 training: False})
            test_features.append((logits_feature))
        test_features = np.array(test_features)

    sess.close()

    train_features = train_features.reshape([-1, 2])
    test_features = test_features.reshape([-1, 2])
    print(train_features.shape)
    print(test_features.shape)

    print('start to classifer with svm')
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(train_features, Y_train_list)
    test = clf.predict(test_features)
    # print('labels',Y_test_list)
    # print('prediction',test)

    count = 0
    original = 0
    tp = 0
    tn = 0
    for i in range(int(len(test) / test_batch_size)):
        original_batch = 0
        for j in range(test_batch_size):
            if test[i * test_batch_size + j] == Y_test_list[i * test_batch_size + j]:
                original_batch += 1
                count += 1
        if original_batch > (test_batch_size / 2):
            original += 1
            if i < int((len(test) / test_batch_size) / 2):
                tn += 1
            else:
                tp += 1
    fp = int((test_size / test_batch_size) / 2) - tp
    fn = int((test_size / test_batch_size) / 2) - tn

    PPV = tp / (tp + fp)
    TPR = tp / (tp + fn)
    F1_score = 2 * tp / (2 * tp + fp + fn)


    predict_precision_batch = count / len(test)
    predict_precision_original = original / (len(test) / test_batch_size)

    print('predict_precision with logits_2', predict_precision_batch)
    print('predict_precision_original with logits_2', predict_precision_original)
    print('PPV logits_2:', PPV)
    print('TPR logits_2:', TPR)
    print('F1_score logits_2:', F1_score)

# train(X_train_list, Y_train_list, X_test_list, Y_test_list)
# evaluate(X_train_list, Y_train_list,X_all_list, Y_all_list, False,'./model_saving/VggNet/'+fold_name+'/2018-10-04_13.00/osteoporosis_classifier-2')
# test
# evaluate(X_train_list, Y_train_list,X_test_list, Y_test_list, False,'./model_saving/VggNet'+'/2018-11-11_20.30/proteinModel-21')
# train
evaluate(X_train_list, Y_train_list,X_test_list, Y_test_list, True,'/home/linghe/code/~/model_saving/VggNet/2018-11-30_14.56/proteinModel-30')
# svm_classifier_split(X_train_list, Y_train_list, X_test_list, Y_test_list,'./osteoporosis_classification/model_saving/VggNet/'+fold_name+'/2018-09-21_00.08/osteoporosis_classifier-8')
# readLabel(Y_train_list)