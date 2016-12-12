#!/usr/bin/env python3
#coding: utf-8

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import random
import operator
import time
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


file_name = '/home/yida/Desktop/da_vincii/Data_2.pkl'
with open(file_name, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    names = save['names']
    del save



# generate labels
# for 10 objectives
image_size = 31
num_labels = 2
num_channels = 1

num_images = dataset.shape[0]
num_train = round(num_images*0.7)
num_valid = round(num_images*0.1)
num_test = round(num_images*0.2)

name2value = {'not_touching':0,'touching':1}
name2string = {'touching':'touching','not_touching':'not_touching'}
value2name = dict((value,name) for name,value in name2value.items())

labels = np.ndarray(num_images, dtype=np.int32)
index = 0
for name in names:
    labels[index] = name2value[name]
    index += 1



def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

rdataset, rlabels = randomize(dataset, labels)
train_dataset = rdataset[0:num_train,:,:]
train_labels = rlabels[0:num_train]
valid_dataset = rdataset[num_train:(num_train+num_valid),:,:]
valid_labels = rlabels[num_train:(num_train+num_valid)]
test_dataset = rdataset[(num_train+num_valid):,:,:]
test_labels = rlabels[(num_train+num_valid):]
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# indices = [random.randint(0,train_dataset.shape[0]) for x in range(3)]
# for index in indices:
#     image = train_dataset[index,:,:]
#     print(value2name[train_labels[index]])
#     plt.imshow(image,cmap='Greys_r')
#     plt.show()



print('......Reformatting......')

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

op = {}
# op[0] = 'tf.train.GradientDescentOptimizer'
# op[1] = 'tf.train.AdadeltaOptimizer'
# op[2] = 'tf.train.AdagradOptimizer'
# op[3] = 'tf.train.AdagradDAOptimizer'
# op[0] = 'tf.train.ProximalGradientDescentOptimizer'
op[0] = 'tf.train.MomentumOptimizer'
# op[0] = 'tf.train.AdamOptimizer'
# op[3] = 'tf.train.FtrlOptimizer'
# op[4] = 'tf.train.ProximalAdagradOptimizer'
# op[5] = 'tf.train.RMSPropOptimizer'
# op[0] = 'tf.train.AdagradDAOptimizer'


for k in range(1):
    for i in range(10):
        for j in range(5):
            batch_size = 100
            patch_size = 2+i
            if patch_size%2 ==0:
                shink = patch_size
            else:
                shink = patch_size +1
            kernel_size = 2
            depth1 = 1+j #the depth of 1st convnet
            depth2 = 3+j #the depth of 2nd convnet
            C5_units = 120
            F6_units = 84
            F7_units = 2
            folder = str(patch_size)+'*'+str(patch_size)+'with'+str(depth1)+'and'+str(depth2)+'_'+op[k]
            try:
                os.stat('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/')
            except:
                os.mkdir('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/')
            graph = tf.Graph()

            with graph.as_default():
                # Input data
                with tf.name_scope('input'):
                    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name = 'train_dataset')
                    # convolution's input is a tensor of shape [batch,in_height,in_width,in_channels]
                    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name = 'train_labels')
                    tf_valid_dataset = tf.constant(valid_dataset)
                    tf_test_dataset = tf.constant(test_dataset)

                # Variables(weights and biases)
                with tf.name_scope('conv1'):
                    C1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth1], stddev=0.1), name = 'C1_weights')
                    # convolution's weights are called filter in tensorflow
                    # it is a tensor of shape [kernel_hight,kernel_width,in_channels,out_channels]
                    C1_biases = tf.Variable(tf.zeros([depth1]), name = 'C1_biases')
                    filter1 = tf.reshape(C1_weights, [depth1,patch_size, patch_size, num_channels])
                    tf.image_summary('filter1', filter1)
                    tf.histogram_summary('C1_biases',C1_biases)

                # S1_weights # Sub-sampling doesn't need weights and biases
                # S1_biases
                with tf.name_scope('conv3'):
                    C3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=0.1), name = 'C3_weights')
                    C3_biases = tf.Variable(tf.constant(1.0, shape=[depth2]), name = 'C3_biases')
                    filter2 = tf.reshape(C3_weights, [depth2,patch_size, patch_size, depth1])
                    tf.histogram_summary('filter2', filter2)
                    tf.histogram_summary('C3_biases',C3_biases)

                # S4_weights
                # S4_biases

                # C5 actually is a fully-connected layer
                with tf.name_scope('fc1'):
                    C5_weights = tf.Variable(tf.truncated_normal([ (16- shink)* (16- shink)* depth2 /4, C5_units], stddev=0.1), name = 'C5_weights')
                    C5_biases = tf.Variable(tf.constant(1.0, shape=[C5_units]), name = 'C5_biases')
                    tf.histogram_summary('C5_weights',C5_weights)
                    tf.histogram_summary('C5_biases',C5_biases)

                with tf.name_scope('fc2'):
                    F6_weights = tf.Variable(tf.truncated_normal([C5_units,F6_units], stddev=0.1), name = 'F6_weights')
                    F6_biases = tf.Variable(tf.constant(1.0, shape=[F6_units]), name = 'F6_biases')
                    tf.histogram_summary('F6_weights',F6_weights)
                    tf.histogram_summary('F6_biases',F6_biases)
                # FC and logistic regression replace RBF
                with tf.name_scope('output'):
                    F7_weights = tf.Variable(tf.truncated_normal([F6_units,F7_units], stddev=0.1), name = 'F7_weights')
                    F7_biases = tf.Variable(tf.constant(1.0, shape=[F7_units]), name = 'F7_biases')
                    tf.histogram_summary('F7_weights',F7_weights)
                    tf.histogram_summary('F7_biases',F7_biases)
                # Model
                def model(data):
                    with tf.device('/gpu:2'):
                        with tf.name_scope('conv1_model'):
                            conv1 = tf.nn.conv2d(data, C1_weights, [1, 1, 1, 1], padding='SAME')
                            hidden1 = tf.nn.relu(conv1 + C1_biases, name = 'hidden1') # relu is better than tanh
                            print (hidden1.get_shape())

                        with tf.name_scope('maxpool1_model'):
                            max_pool1 = tf.nn.max_pool(hidden1,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
                            hidden2 = tf.nn.relu(max_pool1, name = 'hidden2')
                            print (hidden2.get_shape())

                        with tf.name_scope('conv2_model'):
                            conv2 = tf.nn.conv2d(hidden2, C3_weights, [1, 1, 1, 1], padding='VALID')
                            hidden3 = tf.nn.relu(conv2 + C3_biases, name = 'hidden3')
                            print (hidden3.get_shape())

                        with tf.name_scope('maxpool2_model'):
                            max_pool2 = tf.nn.max_pool(hidden3,[1,kernel_size,kernel_size,1],[1,2,2,1],'VALID')
                            hidden4 = tf.nn.relu(max_pool2, name = 'hidden4')
                            print (hidden4.get_shape())
                        with tf.name_scope('fc1_model'):
                            shape = hidden4.get_shape().as_list()
                            reshape = tf.reshape(hidden4, [shape[0], shape[1] * shape[2] * shape[3]])
                            print (reshape)
                            hidden5 = tf.nn.relu(tf.matmul(reshape, C5_weights) + C5_biases)
                        with tf.name_scope('fc2_model'):
                            fc2 = tf.matmul(hidden5,F6_weights)
                            hidden6 = tf.nn.relu(fc2 + F6_biases)
                        with tf.name_scope('output_model'):
                            fc3 = tf.matmul(hidden6,F7_weights, name = 'output')
                            output = fc3 + F7_biases

                        return output


                # Training computation.
                tf_train_dataset = tf.nn.dropout(tf_train_dataset,0.8) # input dropout
                logits = model(tf_train_dataset)
                with tf.name_scope('loss'):
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels), name = 'loss')
                    tf.scalar_summary('loss', loss)

                # Optimizer.
                with tf.name_scope('optimizer'):
                    optimizer = eval(op[k]+'(0.001).minimize(loss)')

                # Predictions for the training, validation, and test data.
                train_prediction = tf.nn.softmax(logits)
                valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
                test_prediction = tf.nn.softmax(model(tf_test_dataset))
                saver = tf.train.Saver(tf.all_variables())
                # merge = tf.merge_all_summaries()

            start_time = time.time()

            num_steps = 50000
            # config = tf.ConfigProto()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.gpu_options.allocator_type = 'BFC'
            config.log_device_placement = True


            sess = tf.Session(graph=graph, config = config)
            # writer = tf.train.SummaryWriter('/home/yida/Desktop/da_vincii/31*31_images/board/'+folder+'/',sess.graph)
            with sess as session:
              tf.initialize_all_variables().run()
              print('Initialized')
              for step in range(num_steps):
                offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l, predictions = session.run(
                  [optimizer, loss, train_prediction], feed_dict=feed_dict)
                # writer.add_summary( summaries, step)
                if (step % 500 == 0):
                    f=open('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/'+folder+'_record.txt','a')
                    f.write('Minibatch loss at step %d: %f' % (step, l))
                    f.write('\n')
                    f.write('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    f.write('\n')
                    f.write('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
                    f.write('\n')

                    f.close()
                    print ('progress: %0.2f'%(step/float(num_steps)))


              f=open('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/'+folder+'_record.txt','a')
              f.write('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
              f.write('\n')
              end_time = time.time()
              duration = (end_time - start_time)/60
              f.write("Excution time: %0.2fmin" % duration)
              f.write('\n')
              f.write('--------------------------------------')
              f.write('\n')
              f.close()
              save_path = saver.save(session, "/home/yida/Desktop/da_vincii/31*31_images/"+folder+"/damodel.ckpt")
              print("Model saved in file: %s" % save_path)


              sender = 'zyd0724@gmail.com'
              receiver = 'zyd0724@hotmail.com'
              subject = 'python email test'
              smtpserver = 'smtp.gmail.com'
              username = 'zyd0724@gmail.com'
              password = 'googlesb'

              msgRoot = MIMEMultipart('related')
              msgRoot['Subject'] = folder


              att = MIMEText(open('/home/yida/Desktop/da_vincii/31*31_images/'+folder+'/'+folder+'_record.txt', 'rb').read(), 'base64', 'utf-8')
              att["Content-Type"] = 'application/octet-stream'
              att["Content-Disposition"] = 'attachment; filename="text.txt"'
              msgRoot.attach(att)

              smtp = smtplib.SMTP()
              smtp.connect('smtp.gmail.com',587)
              smtp.ehlo()
              smtp.starttls()
              smtp.login(username, password)
              smtp.sendmail(sender, receiver, msgRoot.as_string())
              smtp.quit()
  # i_test = 0
  # while(i_test!=''):
  #       i_test = input("Input an index of test image (or Enter to quit): ")
  #       label = test_labels[int(i_test),:].tolist()
  #       #print("Correct label: "+value2name[label.index(1)])
  #       image = test_dataset[int(i_test),:,:,:].reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
  #       prediction = tf.nn.softmax(model(image))
  #       pre_dict = dict(zip(list(range(num_labels)),prediction.eval()[0]))
  #       sorted_pre_dict = sorted(pre_dict.items(), key=operator.itemgetter(1))
  #       name1 = value2name[sorted_pre_dict[-1][0]]
  #       name1 = name2string[name1]
  #       value1 = str(sorted_pre_dict[-1][1])
  #       name2 = value2name[sorted_pre_dict[-2][0]]
  #       name2 = name2string[name2]
  #       value2 = str(sorted_pre_dict[-2][1])
  #       tile = name1+': '+value1+'\n'+name2+': '+value2
  #       image = image.reshape((image_size,image_size)).astype(np.float32)
  #       plt.imshow(image,cmap='Greys_r')
  #       plt.suptitle(tile, fontsize=12)
  #       plt.xlabel(value2name[label.index(1)], fontsize=12)
  #       plt.show()
