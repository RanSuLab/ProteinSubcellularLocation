import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import datetime
from keras.utils import to_categorical
import xlwt
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.framework import arg_scope
# using classifier chains
# from skmultilearn.problem_transform import ClassifierChain
# from sklearn.naive_bayes import GaussianNB

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)


# label-sublocation
# sublocations = ['golgiapparatus', 'mitochondrion', 'vesicles', 'endoplasmicreticulum'
#     , 'nucleolus', 'nucleus', 'cytoskeleton', 'lysosome', 'cytoplasm','cytoplasm+mitochondrion',
#                 'endoplasmicreticulum+vesicles','cytoplasm+nucleus','vesicles+lysosome+golgiapparatus','lysosome+vesicles']
sublocations=['golgiapparatus','mitochondrion','vesicles','endoplasmicreticulum'
             ,'nucleolus','nucleus','cytoskeleton','lysosome','cytoplasm']
fold_name = '1_fold'

now_time = str(datetime.datetime.now())
date = now_time.split(' ')[0]
clork = now_time.split(' ')[-1]
hour = clork.split(':')[0]
min = clork.split(':')[1]


model_path_last = '~/model_saving/DenseNet/' + date + '_' + hour + '.' + min

model_path = model_path_last+'/proteinModel'
# model_path='./model_saving/denseNet'
print('model_save_path:', model_path)
# log_dir = './log_saving/Res_softmax/'+ date + '_' + hour + '.' + min +'/'
def getArray(list):
  
    x = [[0] * 9] * len(list)
    y = np.array(x)
    for i in range(len(list)):
        for j in range(len(list[i])):
            y[i][list[i][j]]=1
    #print(list)
    #print(y)
    return y
class osteoporosis_classifier(object):

    def __init__(self, model_save_path=model_path):
        self.model_save_path = model_save_path

    def conv_layer(self, input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride,
                                   padding='SAME',kernel_initializer=xavier_initializer_conv2d())
        return network

    def Global_Average_Pooling(self, x, stride=1):
        return global_avg_pool(x, name='Global_avg_pooling')

    def Batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return tf.cond(training,
                           lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda: batch_norm(inputs=x, is_training=training, reuse=True))

    def Drop_out(self, x, rate, training):
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def Relu(self, x):
        return tf.nn.relu(x)

    def Average_pooling(self, x, pool_size, stride=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Max_Pooling(self, x, pool_size, stride=2, padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Concatenation(self, layers):
        return tf.concat(layers, axis=3)

    def Linear(self, x, class_num):
        return tf.layers.dense(inputs=x, units=class_num, name='linear')

    def bottleneck_layer(self, x, scope, training, dropout_rate):
            # print(x)
            with tf.name_scope(scope):
                x = self.Batch_Normalization(x, training=training, scope=scope+'_batch1')
                x = self.Relu(x)
                x = self.conv_layer(x, filter=4 * growth_k, kernel=[1,1], layer_name=scope+'_conv1')
                x = self.Drop_out(x, rate=dropout_rate, training=training)

                x = self.Batch_Normalization(x, training=training, scope=scope+'_batch2')
                x = self.Relu(x)
                x = self.conv_layer(x, filter=growth_k, kernel=[3,3], layer_name=scope+'_conv2')
                x = self.Drop_out(x, rate=dropout_rate, training=training)

                # print(x)

                return x

    def transition_layer(self, x, scope, training, dropout_rate):
        with tf.name_scope(scope):
            x = self.Batch_Normalization(x, training=training, scope=scope+'_batch1')
            x = self.Relu(x)
            x = self.conv_layer(x, filter=growth_k, kernel=[1,1], layer_name=scope+'_conv1')
            x = self.Drop_out(x, rate=dropout_rate, training=training)
            x = self.Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name, training, dropout_rate):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0), training=training, dropout_rate=dropout_rate)

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = self.Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i+1), training=training, dropout_rate=dropout_rate)
                layers_concat.append(x)

            x = self.Concatenation(layers_concat)

            return x

    def DenseNet(self, input_x, dropout_rate):
        training = tf.placeholder(tf.bool, name='training')

        x = self.conv_layer(input_x, filter=2 * growth_k, kernel=[7,7], stride=2, layer_name='conv0')
        x = self.Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """




        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1',training=training,dropout_rate=dropout_rate)
        x = self.transition_layer(x, scope='trans_1',training=training,dropout_rate=dropout_rate)

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2',training=training,dropout_rate=dropout_rate)
        x = self.transition_layer(x, scope='trans_2',training=training,dropout_rate=dropout_rate)

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3',training=training,dropout_rate=dropout_rate)
        x = self.transition_layer(x, scope='trans_3',training=training,dropout_rate=dropout_rate)

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final',training=training,dropout_rate=dropout_rate)



        # 100 Layer
        x = self.Batch_Normalization(x, training=training, scope='linear_batch')
        x = self.Relu(x)
        x = self.Global_Average_Pooling(x)
        x = flatten(x)
        fc_128 = tf.layers.dense(x,128,activation=tf.nn.relu)
        # out = tf.layers.dense(fc_128,2,activation=None)
       # out = tf.layers.dense(fc_128,classes,activation=None)
        out = tf.layers.dense(fc_128,classes,activation=tf.nn.sigmoid)

        return fc_128,out,training

    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            #print(logits)
            # cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_conv)
           # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy = tf.keras.losses.binary_crossentropy(labels, logits)
           # cross_entropy = tf.losses.sigmoid_cross_entropy(labels, logits=logits)
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

    def next_batch(self, xs, ys, batch_size, start, size):

        start = start % round(size / batch_size)
        start = start * batch_size
        end = start + batch_size
        batch_xs_list = xs[start:end]
        batch_ys_list = ys[start:end]
        batch_xs = []
        for i in range(len(batch_xs_list)):
            image = Image.open(batch_xs_list[i])
            # image = image.resize((100,100))
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

        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        features = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        labels = tf.placeholder(tf.float32, [None, classes])
        step_ = tf.placeholder(tf.int8)

        logits_128, logits_2, train_mode = self.DenseNet(features, dropout_rate)

        cross_entropy = self.cost(logits_2, labels)
        # tf.add_to_collection('losses', cross_entropy)
        # loss = tf.add_n(tf.get_collection('losses'))
        # tf.summary.scalar('cross_entropy',cross_entropy)
        global_step = tf.Variable(0)
        
        learning_rate = tf.train.exponential_decay(lr_value,global_step, 10, 0.9, staircase=True)
        with tf.name_scope('optimizer'):
           update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
           with tf.control_dependencies(update_ops):
               train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cross_entropy)
               #print(sess.runtrain_step)
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
               # print("----------------------------------")
               # print(sess.run(train_step))
              #  print(sess.run(learning_rate))
               #learning_rate = tf.train.exponential_decay(lr_value, it, 10, 0.01, staircase=True)
               # with tf.name_scope('optimizer'):
                   # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    #with tf.control_dependencies(update_ops):
                       # train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cross_entropy)
                for i in range(int(train_size / batch_size)):

                    batch_xs, batch_ys = self.next_batch(X_train_list, Y_train_list, batch_size, i, train_size)
                    #print(batch_ys)
                    train_step.run(
                        feed_dict={step_: it, features: batch_xs, labels: batch_ys, dropout_rate: drop_out_rate_value,
                                   train_mode: True})

                    if (i + 1) % 100 == 0:
                        train_cost = sess.run(cross_entropy, feed_dict={step_: it, features: batch_xs,
                                                                        labels: batch_ys, train_mode: False,
                                                                        dropout_rate: 0.0})
                        # train_loss = sess.run(loss, feed_dict={step_: it, features: batch_xs,
                        #                                                 labels: batch_ys, train_mode: False})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              '  step %d, training cross_entropy: %g' % (i + 1, train_cost))

                right_all = 0
                for x in range(int(test_size / test_batch_size)):
                    batch_test_xs, batch_test_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, x,
                                                                   test_size)
                    y_pre = sess.run(logits_2,
                                     feed_dict={features: batch_test_xs, train_mode: False, dropout_rate: 0.0})
                    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(batch_test_ys, 1))
                    right_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(right_num,
                                      feed_dict={features: batch_test_xs, labels: batch_test_ys, train_mode: False,
                                                 dropout_rate: 0.0})
                    right_all += result
                    if x == int(test_size / test_batch_size)-1:
                        test_cost = sess.run(cross_entropy, feed_dict={step_: it + 1, features: batch_test_xs,
                                  labels: batch_test_ys,
                                  train_mode: False,dropout_rate:0.0})
                        now_time = str(datetime.datetime.now())
                        print(now_time, ' epoch_%s' % (it + 1),
                              'test_loss %g' % (test_cost))
                acc = right_all / test_size
                # test_acc_list.append(acc)
                now_time = str(datetime.datetime.now())
                print(now_time, ' epoch_%s' % (it + 1), '  test_acc %g' % (acc))

                if acc >= 0.35:
                    if not os.path.exists(model_path_last):
                        os.makedirs(model_path_last)
                    saver.save(sess, model_path, global_step=it + 1)
                    print('save epoch', it + 1)
            print('Finished training!')

            # saver.save(sess, self.model_save_path)
        sess.close()

    def evaluate(self,X_all_list, Y_all_list, model_path):

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
        y_ = tf.placeholder(tf.float32, [None, classes])

        logits_128, logits_2, train_mode = self.DenseNet(x,0.0)
        right_number = self.right_number(logits_2, y_)
        # output_softmax = self.inference(logits_2)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Start to restore model')
            saver.restore(sess,
                          model_path)  # './model_saving/resnet_softmax/2018-08-11_00.35/osteoporosis_classifier'  self.model_save_path

            print('Start to evaluate test data')
            if test_size % test_batch_size == 0:
                test_features = []
                right = 0
                for i in range(int(test_size / test_batch_size)):
                    batch_xs, batch_ys = self.next_batch(X_test_list, Y_test_list, test_batch_size, i, test_size)
                    logits_feature = sess.run(logits_128,
                                              feed_dict={x: batch_xs, y_: batch_ys,
                                                         train_mode: False})
                    test_features.append((logits_feature))

                    patch_right_num = sess.run(right_number,
                                         feed_dict={x: batch_xs, y_: batch_ys, train_mode: False})
                    print(i,'/',int(test_size/test_batch_size),'   ',patch_right_num)
                    right += patch_right_num
                test_features = np.array(test_features)
                print(right)
                test_features = test_features.reshape([-1, 128])
                book = xlwt.Workbook()
                sheet1 = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
                for i in range(test_features.shape[0]):
                    for j in range(test_features.shape[1]):
                        sheet1.write(i+1,j+1,float(test_features[i][j]))
                book.save('./densenet_feature47.xls')


        sess.close()


def load_data_list1():
    X_train_list = []
    Y_train_list = []
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/data_aug/train/'
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/all/'
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
    dir = '/dev/sda3/linghe/linghe/Multilabel/new/all/'
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

def load_data_list():
    X_train_list = []
    Y_train_list = []
    dir = '/dev/sda3/linghe/linghe/Multilabel/data_aug2/train/'
    # dir = 'E:/hpaData/ProcessMultiLabel/dataset/new/data_aug/train/'
    # dir = 'E:/hpaData/ProcessMultiLabel/test/'
    # dir = '/home/linghe/dataset/all/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        # label = int(image_name.split('_')[1])#orginal
        # label=sublocation
        label = ''
        sublocation = (image_name.split('_')[2]).split('.')[0][:-1].lower()
        if (sublocation == 'nucleus+cytoplasm'):
            sublocation = 'cytoplasm+nucleus'
        for i in range(len(sublocations)):
            if (sublocation == sublocations[i]):
                label = i
        # label = image_name.split('_')[2][:-1]
        X_train_list.append(image_path)
        Y_train_list.append(label)

    X_test_list = []
    Y_test_list = []
    dir = '/dev/sda3/linghe/linghe/Multilabel/data_aug2/test/'
    # dir = 'E:/hpaData/ProcessMultiLabel/dataset/new/data_aug/test/'
    for image_name in os.listdir(dir):
        image_path = dir + image_name
        # label = int(image_name.split('_')[1])
        sublocation = (image_name.split('_')[2]).split('.')[0][:-1].lower()
        if (sublocation == 'nucleus+cytoplasm'):
            sublocation = 'cytoplasm+nucleus'
        for i in range(len(sublocations)):
            if (sublocation == sublocations[i]):
                label = i
        X_test_list.append(image_path)
        Y_test_list.append(label)
    return X_train_list, Y_train_list, X_test_list, Y_test_list


# X_train_list, Y_train_list, X_test_list, Y_test_list,X_all_list, Y_all_list = load_data_list()
X_train_list, Y_train_list, X_test_list, Y_test_list = load_data_list1()
train_size = len(X_train_list)
test_size = len(X_test_list)
# all_size = len(X_all_list)
# classes = 2
classes = 9


def main(_):

    # global keep_prob_value
    global batch_size
    global epochs
    global lr_value
    global classes
    global train_size
    global test_size
    global test_batch_size
    global growth_k
    global nb_block
    global drop_out_rate_value
    # global all_batch_size
    global image_size

    lr_value = 0.01
    epochs = 100
    batch_size = 32
    test_batch_size = 32
    test_batch_size = 3
    # all_batch_size = 58
    growth_k = 24
    nb_block = 2
    drop_out_rate_value = 0.1
    # all_batch_size = 58
    # image_size = 100
    image_size = 224

    # X_train_list, Y_train_list, X_test_list, Y_test_list,X_all_list, Y_all_list = load_data_list()
    X_train_list, Y_train_list, X_test_list, Y_test_list = load_data_list1()
    train_size = len(X_train_list)
    test_size = len(X_test_list)
    # classes = 2
    classes = 9
    print('train_size: ', train_size)
    print('test_size: ', test_size)
    # print('all_size: ',all_size)
    print('classes: ', classes)
    print('epochs:',epochs)
   # print(Y_train_list)
    model = osteoporosis_classifier()
    #model.train(X_train_list, Y_train_list, X_test_list, Y_test_list)
    model.evaluate(X_test_list, Y_test_list, model_path='/dev/sda3/linghe/linghe/Multilabel/code/~/model_saving/DenseNet/'+'2019-12-27_08.06/proteinModel-47')
    # E:\myProgram\python\deepLearning\model\model_saving\AlexNet\2018 - 10 - 30_20.22
    # model.svm_classifier_split(X_train_list, Y_train_list,X_test_list, Y_test_list,'./model_saving/ResNet/'+fold_name+'/2018-09-13_00.25/osteoporosis_classifier-12')



if __name__ == '__main__':
    tf.app.run(main=main)
