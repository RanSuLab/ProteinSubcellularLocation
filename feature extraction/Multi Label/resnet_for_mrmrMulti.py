import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import datetime
from sklearn import svm
from keras.utils import to_categorical
import xlwt
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)


# label-sublocation
sublocations=['golgiapparatus','mitochondrion','vesicles','endoplasmicreticulum'
             ,'nucleolus','nucleus','cytoskeleton','lysosome','cytoplasm']
fold_name = '1_fold'

now_time = str(datetime.datetime.now())
date = now_time.split(' ')[0]
clork = now_time.split(' ')[-1]
hour = clork.split(':')[0]
min = clork.split(':')[1]


model_path_last = '~/model_saving/ResNet/' + date + '_' + hour + '.' + min
model_path = model_path_last+'/proteinModel'
# model_path='./model_saving/AlexNet'
print('model_save_path:', model_path)
# log_dir = './log_saving/Res_softmax/'+ date + '_' + hour + '.' + min +'/'
def getArray(list):

    x = [[0] * 9] * len(list)
    y = np.array(x)
    for i in range(len(list)):
        for j in range(len(list[i])):
            y[i][list[i][j]]=1
    return y
class osteoporosis_classifier(object):

    def __init__(self, model_save_path=model_path):
        self.model_save_path = model_save_path

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        """
        Implementation of the identity block as defined in Figure 3
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- AlexNet list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            # first
            W_conv1 = self.get_weight_conv(block_name+'conv1', [1, 1, in_filter, f1], False)
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # second
            W_conv2 = self.get_weight_conv(block_name+'conv2', [kernel_size, kernel_size, f1, f2], False)
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # third

            W_conv3 = self.get_weight_conv(block_name+'conv3', [1, 1, f2, f3], False)
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)

            # final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result

    def convolutional_block(self, X_input, kernel_size, in_filter,
                            out_filters, stage, block, training, stride=2):
        """
        Implementation of the convolutional block as defined in Figure 4
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- AlexNet list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        training -- train or test
        stride -- Integer, specifying the stride to be used
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters

            x_shortcut = X_input
            # first
            W_conv1 = self.get_weight_conv(block_name+'conv1', [1, 1, in_filter, f1], False)
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # second
            W_conv2 = self.get_weight_conv(block_name+'conv2', [kernel_size, kernel_size, f1, f2], False)
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)
            X = tf.nn.relu(X)

            # third
            W_conv3 = self.get_weight_conv(block_name+'conv3', [1, 1, f2, f3], False)
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, training=training)

            # shortcut path
            W_shortcut = self.get_weight_conv(block_name+'conv4', [1, 1, in_filter, f3], False)
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='SAME')

            # final
            add = tf.add(x_shortcut, X)
            add_result = tf.nn.relu(add)

        return add_result

    def get_weight_conv(self, name, shape, regularizer):
        kernel = tf.get_variable(name, shape, dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # kernel = tf.get_variable(name,shape,dtype=tf.float32,initializer=tf.contrib.layers.variance_scaling_initializer())
        if regularizer == True:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_1)(kernel))
        return kernel

    def ResNet(self, x_input, classes):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
        Arguments:
        Returns:
        """
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3], [3, 3], [0, 0]]), 'CONSTANT')
        with tf.variable_scope('reference'):
            training = tf.placeholder(tf.bool, name='training')

            # stage 1
            w_conv1 = self.get_weight_conv('stage1_conv1',[7, 7, 1, 64], False)
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.relu(x)
            max_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

            # stage 2
            x = self.convolutional_block(max_pool, 3, 64, [64, 64, 256], 2, 'a', training=training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            # stage 3
            x = self.convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'b', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'c', training=training)
            x = self.identity_block(x, 3, 512, [128, 128, 512], 3, 'd', training=training)

            # stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            # stage 5
            x = self.convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
            x = self.identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)

            avg_pool = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

            flatten = tf.layers.flatten(avg_pool)
            x = tf.layers.dense(flatten, units=512, activation=tf.nn.relu)
            # x = tf.layers.dense(x,units=512,activation=tf.nn.relu)
            logits_128 = tf.layers.dense(x,units=128,activation=tf.nn.relu)

            # Dropout - controls the complexity of the model, prevents co-adaptation of features.
            # if drop_out == True:
            #     with tf.name_scope('dropout'):
            #         keep_prob = tf.placeholder(tf.float32)
            #         x = tf.nn.dropout(x, keep_prob_value)

            logits_2 = tf.layers.dense(logits_128, units=classes, activation=tf.nn.sigmoid)

        return logits_128, logits_2, training

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def inference(self,x):
        return tf.nn.softmax(x)

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(bias_value, shape=shape)
        return tf.Variable(initial)

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
            # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy = tf.keras.losses.binary_crossentropy(labels, logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost

    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_mean(correct_prediction)
        return accuracy_op

    def right_number(self, logits, labels):
        with tf.name_scope('right_number'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_op = tf.reduce_sum(correct_prediction)
        return accuracy_op

    def next_batch(self,xs, ys, batch_size, start, size):
        start = start % round(size / batch_size)
        start = start * batch_size
        end = start + batch_size
        batch_xs_list = xs[start:end]
        batch_ys_list = ys[start:end]
        batch_xs = []
        for i in range(len(batch_xs_list)):
            image = Image.open(batch_xs_list[i])
            image = image.resize((224, 224))
            image_narray = np.asarray(image, dtype='float32')
            image_narray = image_narray * 1.0 / 255
            batch_xs.append(image_narray)
        batch_xs = np.stack(batch_xs)
        batch_xs = np.expand_dims(batch_xs, 3)
        # batch_ys = to_categorical(batch_ys_list, classes)
        batch_ys = np.ndarray(shape=(len(batch_ys_list), 9), dtype=int, buffer=np.array(getArray(batch_ys_list)),
                              offset=0, order="C")
        return batch_xs, batch_ys

    def train(self, X_train_list, Y_train_list, X_test_list, Y_test_list):
        if not os.path.exists(model_path_last):
            os.makedirs(model_path_last)
        features = tf.placeholder(tf.float32, [None, 224, 224, 1])
        labels = tf.placeholder(tf.float32, [None, classes])
        step_ = tf.placeholder(tf.int8)

        logits_128, logits_2, train_mode = self.ResNet(features, classes)

        cross_entropy = self.cost(logits_2, labels)
        # tf.add_to_collection('losses', cross_entropy)
        # loss = tf.add_n(tf.get_collection('losses'))
        # tf.summary.scalar('cross_entropy',cross_entropy)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(lr_value, step_, 10, 0.9, staircase=True)
        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # merged = tf.summary.merge_all()
            # writer = tf.summary.FileWriter(log_dir+'/train/', sess.graph)
            # train_cost_list = []
            # test_acc_list = []
            print('Start to train')
            for it in range(epochs):
                # shuffle
                c = list(zip(X_train_list, Y_train_list))
                random.Random(random.randint(0, 10000)).shuffle(c)
                X_train_list, Y_train_list = zip(*c)

                for i in range(int(train_size / batch_size)):

                    batch_xs, batch_ys = self.next_batch(X_train_list, Y_train_list, batch_size, i, train_size)

                    train_step.run(
                        feed_dict={step_: it, features: batch_xs, labels: batch_ys, train_mode: True})

                    if (i + 1) % 100 == 0:
                        train_cost = sess.run(cross_entropy, feed_dict={step_: it, features: batch_xs,
                                                                        labels: batch_ys, train_mode: False})
                        # train_loss = sess.run(loss, feed_dict={step_: it, features: batch_xs,
                        #                                                 labels: batch_ys, train_mode: False})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              '  step %d, training cross_entropy: %g' % (i + 1, train_cost))


                right_all = 0
                for x in range(int(test_size / test_batch_size)):
                    batch_test_xs, batch_test_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, x,
                                                                   test_size)
                    y_pre = sess.run(logits_2, feed_dict={features: batch_test_xs, train_mode: False})
                    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(batch_test_ys, 1))
                    right_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(right_num,
                                      feed_dict={features: batch_test_xs, labels: batch_test_ys, train_mode: False})
                    right_all += result
                acc = right_all / test_size
                # test_acc_list.append(acc)
                now_time = str(datetime.datetime.now())
                print(now_time, ' epoch_%s' % (it + 1), '  test_acc %g' % (acc))

                if acc >= 0.3:
                    saver.save(sess, model_path, global_step=it + 1)
                    print('save epoch', it + 1)
            print('Finished training!')

            # saver.save(sess, self.model_save_path)
        sess.close()

    def evaluate(self, X_train_list, Y_train_list, X_test_list, Y_test_list, evaluate_train, model_path):

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 224, 224, 1])
        y_ = tf.placeholder(tf.float32, [None, classes])

        logits_128, logits_2, train_mode = self.ResNet(x, classes)
        right_number = self.right_number(logits_2, y_)
        output_softmax = self.inference(logits_2)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Start to restore model')
            saver.restore(sess,
                          model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

            if evaluate_train == True:
                print('Start to evaluate train data')
                train_patch_right_num_all = 0
                train_features = []
                # original_right_all = 0
                if train_size % train_batch_size == 0:
                    for i in range(int(train_size / train_batch_size)):
                        batch_xs, batch_ys = self.next_batch(X_train_list, Y_train_list, train_batch_size, i, train_size)
                        right_num = sess.run(right_number,
                                             feed_dict={x: batch_xs, y_: batch_ys, train_mode: False})
                        logits_feature = sess.run(logits_128,
                                                  feed_dict={x: batch_xs, y_: batch_ys,
                                                             train_mode: False})
                        train_features.append((logits_feature))
                        print(i,'/',int(train_size/train_batch_size), right_num)

                        # if right_num >= 35:
                        #     original_right_all += 1
                        train_patch_right_num_all += right_num
                    print('patch_right_num_all:', train_patch_right_num_all)
                    # print('original_right_all:',original_right_all)
                    train_patch_acc = train_patch_right_num_all / train_size
                    # original_acc = original_right_all/34
                    print('patch: train', '%s accuracy', train_patch_acc)
                    # print('original:', '%s accuracy' % name, original_acc)
                    train_features = np.array(train_features)
                    train_features = train_features.reshape([-1, 128])
                    book = xlwt.Workbook()
                    sheet1 = book.add_sheet(u'sheet1', cell_overwrite_ok=True)
                    for i in range(train_features.shape[0]):
                        #sheet1.write(i + 1, 0, sublocations[Y_test_list[i]])
                        for j in range(train_features.shape[1]):
                            sheet1.write(i + 1, j + 1, float(train_features[i][j]))
                    book.save('./resnet_featureNew39.xls')
            print('Start to evaluate test data')
            test_patch_right_num_all = 0
            original_right_all = 0
            tp = 0
            tn = 0
            if test_size % test_batch_size == 0:
                test_features = []
                for i in range(int(test_size / test_batch_size)):
                    batch_xs, batch_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, i, test_size)
                    logits_feature = sess.run(logits_128,
                                              feed_dict={x: batch_xs, y_: batch_ys,
                                                         train_mode: False})
                    test_features.append((logits_feature))

                    patch_right_num = sess.run(right_number,
                                         feed_dict={x: batch_xs, y_: batch_ys, train_mode: False})
                    print(i,'/',int(test_size/test_batch_size),'   ',patch_right_num)
                    if patch_right_num > test_batch_size/2:
                        original_right_all += 1
                        if i<int((test_size / test_batch_size)/2):
                            tn += 1
                        else:
                            tp += 1
                    test_patch_right_num_all += patch_right_num
                test_features = np.array(test_features)
                test_features = test_features.reshape([-1, 128])
                book = xlwt.Workbook()
                sheet1 = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
                for i in range(test_features.shape[0]):
                    sheet1.write(i+1, 0, sublocations[Y_test_list[i]])
                    for j in range(test_features.shape[1]):
                        sheet1.write(i+1,j+1,float(test_features[i][j]))
                book.save('./resnet_featureTest.xls')


                # fp = int((test_size / test_batch_size) / 2) - tp
                # fn = int((test_size / test_batch_size) / 2) - tn
                #
                # PPV = tp/(tp+fp)
                # TPR = tp/(tp+fn)
                # F1_score = 2*tp/(2*tp+fp+fn)
                #
                #
                # print('test_patch_right_num_all:', test_patch_right_num_all)
                # print('original_right_all:',original_right_all)
                # patch_acc = test_patch_right_num_all / test_size
                # original_acc = original_right_all/(test_size/test_batch_size)
                # print('test_patch_accuracy', patch_acc)
                # print('test_original_accuracy',  original_acc)
                # print('PPV:',PPV)
                # print('TPR:', TPR)
                # print('F1_score:', F1_score)

                # test_features = np.array(test_features)
                # test_features = test_features.reshape([-1, 2])
                # y_score = []
                # for i in range(len(test_features)):
                #     x = test_features[i][1]
                #     y_score.append(x)
                #
                #
                # fpr, tpr, threshold = roc_curve(Y_test_list, y_score)  ###
                # roc_auc = auc(fpr, tpr)  ###
                #
                # plt.figure()
                # lw = 2
                # plt.figure(figsize=(10, 10))
                # plt.plot(fpr, tpr, color='darkorange',
                #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###
                # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.05])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('ROC curve')
                # plt.legend(loc="lower right")
                # plt.show()

        sess.close()

    def svm_classifier_split(self, X_train_list, Y_train_list, X_test_list, Y_test_list, restore_model_path):

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 224, 224, 1])
        y_ = tf.placeholder(tf.float32, [None, classes])

        logits_128, logits_2, train_mode = self.ResNet(x, 2)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Start to restore model')
            saver.restore(sess,
                          restore_model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

            print('Start to extract train features with logits_2')
            train_features = []
            for i in range(int(train_size / train_batch_size)):
                train_batch_xs, train_batch_ys = self.next_batch(X_train_list, Y_train_list, train_batch_size, i, train_size)
                logits_feature = sess.run(logits_2,
                                          feed_dict={x: train_batch_xs, y_: train_batch_ys,
                                                     train_mode: False})
                train_features.append((logits_feature))
            train_features = np.array(train_features)

            print('Start to extract test features with logits_2')
            test_features = []
            for i in range(int(test_size / test_batch_size)):
                test_batch_xs, test_batch_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, i,
                                                                 test_size)
                logits_feature = sess.run(logits_2,
                                          feed_dict={x: test_batch_xs, y_: test_batch_ys,
                                                     train_mode: False})
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
        for i in range(int(len(test)/test_batch_size)):
            original_batch = 0
            for j in range(test_batch_size):
                if test[i*test_batch_size+j] == Y_test_list[i*test_batch_size+j]:
                    original_batch+=1
                    count += 1
            if original_batch>(test_batch_size/2):
                original+=1
                if i < int((len(test) / test_batch_size) / 2):
                    tn += 1
                else:
                    tp += 1
        fp = int((test_size / test_batch_size) / 2) - tp
        fn = int((test_size / test_batch_size) / 2) - tn

        PPV = tp / (tp + fp)
        TPR = tp / (tp + fn)
        F1_score = 2 * tp / (2 * tp + fp + fn)

        #compute acc
        predict_precision_batch = count / len(test)
        predict_precision_original = original/(len(test)/test_batch_size)

        print('predict_precision with logits_2', predict_precision_batch)
        print('predict_precision_original with logits_2', predict_precision_original)
        print('PPV logits_2:', PPV)
        print('TPR logits_2:', TPR)
        print('F1_score logits_2:', F1_score)


        # with tf.Session() as sess:
        #     print('Start to restore model')
        #     saver.restore(sess,
        #                   restore_model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path
        #
        #     print('Start to extract train features with logits_128')
        #     train_features = []
        #     for i in range(int(train_size / train_batch_size)):
        #         train_batch_xs, train_batch_ys = self.next_batch(X_train_list, Y_train_list, train_batch_size, i, train_size)
        #         logits_feature = sess.run(logits_128,
        #                                   feed_dict={x: train_batch_xs, y_: train_batch_ys,
        #                                              train_mode: False})
        #         train_features.append((logits_feature))
        #     train_features = np.array(train_features)
        #
        #     print('Start to extract test features with logits_128')
        #     test_features = []
        #     for i in range(int(test_size / test_batch_size)):
        #         test_batch_xs, test_batch_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, i,
        #                                                          test_size)
        #         logits_feature = sess.run(logits_128,
        #                                   feed_dict={x: test_batch_xs, y_: test_batch_ys,
        #                                              train_mode: False})
        #         test_features.append((logits_feature))
        #     test_features = np.array(test_features)
        #
        # sess.close()
        #
        # train_features = train_features.reshape([-1, 128])
        # test_features = test_features.reshape([-1, 128])
        # print(train_features.shape)
        # print(test_features.shape)
        #
        # print('start to classifer with svm')
        # clf = svm.SVC(kernel='linear', C=10)
        # clf.fit(train_features, Y_train_list)
        # test = clf.predict(test_features)
        # # print('labels',Y_test_list)
        # # print('prediction',test)
        #
        # count = 0
        # original = 0
        # tp = 0
        # tn = 0
        # for i in range(int(len(test)/test_batch_size)):
        #     original_batch = 0
        #     for j in range(test_batch_size):
        #         if test[i*test_batch_size+j] == Y_test_list[i*test_batch_size+j]:
        #             original_batch+=1
        #             count += 1
        #     if original_batch>test_batch_size/2:
        #         original+=1
        #         if i < int((len(test) / test_batch_size) / 2):
        #             tn += 1
        #         else:
        #             tp += 1
        # fp = int((test_size / test_batch_size) / 2) - tp
        # fn = int((test_size / test_batch_size) / 2) - tn
        #
        # PPV = tp / (tp + fp)
        # TPR = tp / (tp + fn)
        # F1_score = 2 * tp / (2 * tp + fp + fn)
        #
        #
        # #compute acc
        # predict_precision_batch = count / len(test)
        # predict_precision_original = original/(len(test)/test_batch_size)
        #
        # print('predict_precision with logits_128', predict_precision_batch)
        # print('predict_precision_original with logits_128', predict_precision_original)
        # print('PPV with logits_128:', PPV)
        # print('TPR with logits_128:', TPR)
        # print('F1_score with logits_128:', F1_score)





def load_data_list():
    X_train_list = []
    Y_train_list = []
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/data_aug/train/'
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/all/'
   #dir = '/dev/sda3/linghe/linghe/Multilabel/protein/'
    # dir = 'E:/hpaData/ProcessMultiLabel/dataset/data_augument/train/'
    # dir = 'E:/hpaData/ProcessMultiLabel/test/'
    # dir = '/home/linghe/dataset/all/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        # label = int(image_name.split('_')[1])#orginal
        # label=sublocation
        labels = []
        label = ''
        sublocation = (image_name.split('_')[2]).split('.')[0][:-1]
        sublocation1 = sublocation.split("+")
        for j in range(len(sublocation1)):
            for i in range(len(sublocations)):
                if (sublocation1[j].lower() == sublocations[i]):
                    label = i
                    labels.append(label)
        X_train_list.append(image_path)
        Y_train_list.append(np.array(labels))
    X_test_list = []
    Y_test_list = []
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/data_aug/test/'
    #dir = '/dev/sda3/linghe/linghe/Multilabel/protein1/'
    # dir = './Data_python/AlexNet/data_fold_crop_aug_CLAHE/'+fold_name+'/test1/'
    # dir = '/home/linghe/dataset/test1/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        labels = []
        label = ''
        sublocation = (image_name.split('_')[2]).split('.')[0][:-1]
        sublocation1 = sublocation.split("+")
        for j in range(len(sublocation1)):
            for i in range(len(sublocations)):
                if (sublocation1[j].lower() == sublocations[i]):
                    label = i
                    labels.append(label)
        X_test_list.append(image_path)
        Y_test_list.append(np.array(labels))
    return X_train_list, Y_train_list, X_test_list, Y_test_list
#*******************************************************************************************************
# def load_all_list(classes):
#     all_data = []
#     all_label = []
#     for i in range(2):
#         dir = './Data_python/Vgg_Res/Vgg_Res_data_8/original/%s/' % i
#         for image_name in os.listdir(dir):
#             image_path = dir + image_name
#             image = Image.open(image_path)
#             image = image.resize((224,224))
#             image_narray = np.asarray(image, dtype='float32')
#             image_narray = image_narray * 1.0 / 255
#             all_data.append(image_narray)
#             all_label.append(i)
#     all_data = np.stack(all_data)
#     all_data = np.expand_dims(all_data,3)
#     print(all_data.shape)
#     all_label = to_categorical(all_label, classes)
#     print(all_label.shape)
#     return all_data, all_label

def main(_):

    # global keep_prob_value
    global batch_size
    global epochs
    global lr_value
    global classes
    global train_size
    global test_size
    global test_batch_size
    global train_batch_size
    global lambda_1
    global bias_value


    lr_value = 0.01
    # lr_value = 0.001
    keep_prob_value = 1.0
    epochs = 100
    # epochs = 2
    bias_value = 0.0
    # batch_size = 50
    batch_size = 32
    # test_batch_size = 58
    test_batch_size = 18
    # train_batch_size = 100
    train_batch_size = 3
    lambda_1 = 0.01

    X_train_list, Y_train_list, X_test_list, Y_test_list = load_data_list()
    train_size = len(X_train_list)
    test_size = len(X_test_list)
    # classes = 2
    classes = 9
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    print('classes: ', classes)
    print('epochs:',epochs)

    model = osteoporosis_classifier()
   # model.train(X_train_list, Y_train_list, X_test_list, Y_test_list)
    model.evaluate(X_train_list, Y_train_list, X_test_list, Y_test_list, True, model_path='/dev/sda3/linghe/linghe/Multilabel/code/~/model_saving/ResNet/2019-12-27_11.12/proteinModel-39')
    # model.svm_classifier_split(X_train_list, Y_train_list,X_test_list, Y_test_list,'./model_saving/ResNet/'+fold_name+'/2018-09-13_00.25/osteoporosis_classifier-12')



if __name__ == '__main__':
    tf.app.run(main=main)
