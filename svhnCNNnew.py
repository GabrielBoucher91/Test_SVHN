from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#A = loadmat('train_32x32')
#B = loadmat('test_32x32')

#X_train = A['X']
#y_train = A['y']
#X_test = B['X']
#y_test = B['y']


# https://github.com/thomalm/svhn-multi-digit
# un bon lien pour se baser


#########################################################
# Preprocessing

# Le chiffre 0 est associé à l'étiquette 10, on lui associe l'étiquette 0
#y_train[y_train==10] = 0
#y_test[y_test==10] = 0

# Pour faciliter l'accès aux données
#X_train = X_train.transpose(3,0,1,2).astype(np.float32)
#X_test = X_test.transpose(3,0,1,2).astype(np.float32)

#X_train,y_train = shuffle(X_train,y_train)
#X_test,y_test = shuffle(X_test,y_test)


# Permet de d'avoir un vecteur r pour chaque donnée
#enc = OneHotEncoder().fit(y_train)
#y_train = enc.transform(y_train).toarray().astype(np.float32)
#y_test = enc.transform(y_test).toarray().astype(np.float32)
#
#print(np.shape(X_train))
#print(np.shape(y_train))
tf.reset_default_graph()
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_true = tf.placeholder(tf.float32, shape=[None, 10])


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
#    tf.summary.histogram("weights", w)
#    tf.summary.histogram("biases", b)
#    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
#    tf.summary.histogram("weights", w)
#    tf.summary.histogram("biases", b)
#    tf.summary.histogram("activations", act)
    return act

conv1 = conv_layer(x,1,32,"conv1")
conv_out = conv_layer(conv1,32,64,"conv2")

flattened = tf.reshape(conv_out,[-1,7*7*64])

fc1 = fc_layer(flattened,7*7*64,1024)
relu = tf.nn.relu(fc1)
logits = fc_layer(fc1,1024,10,"fc2")


# Building simple model
#with tf.name_scope('Conv_1'):
#    weights1 = tf.Variable(tf.truncated_normal([5, 5, 1, 28],stddev=0.1))
#    biases1 = tf.Variable(tf.constant(0.1,shape=[28]))
#    conv1 = tf.nn.conv2d(x, weights1, strides=[1,1,1,1], padding='SAME')
#    conv1_out = tf.nn.relu(conv1+biases1)
#    pool1 = tf.nn.max_pool(conv1_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#    tf.summary.histogram('Weights1', weights1)
#
#with tf.name_scope('Conv_2'):
#    weights2 = tf.Variable(tf.truncated_normal([5, 5, 28, 56],stddev=0.1))
#    biases2 = tf.Variable(tf.constant(0.1,shape=[56]))
#    conv2 = tf.nn.conv2d(pool1, weights2, strides=[1,1,1,1], padding='SAME')
#    conv2_out = tf.nn.relu(conv2+biases2)
#    pool2 = tf.nn.max_pool(conv2_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#    tf.summary.histogram('Weights2', weights2)
#
#flat_out = tf.reshape(pool2, [-1, 7*7*56])
#
#with tf.name_scope('Fully_1'):
#    weights3 = tf.Variable(tf.truncated_normal([7*7*56, 300]))
#    biases3 = tf.Variable(tf.truncated_normal([300]))
#    relu3 = tf.nn.relu(tf.matmul(flat_out, weights3)+biases3)
#    tf.summary.histogram('Weights3', weights3)
#
#with tf.name_scope('Fully_2'):
#    weights4 = tf.Variable(tf.truncated_normal([300, 100]))
#    biases4 = tf.Variable(tf.truncated_normal([100]))
#    relu4 = tf.nn.relu(tf.matmul(relu3, weights4)+biases4)
#
#with tf.name_scope('Fully_3'):
#    weights5 = tf.Variable(tf.truncated_normal([100, 10]))
#    biases5 = tf.Variable(tf.truncated_normal([10]))
#    relu5 = tf.nn.relu(tf.matmul(relu4, weights5)+biases5)
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#tf.summary.scalar('Entropy', cross_entropy)


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#tf.summary.scalar('Accuracy', accuracy)
sess.run(tf.global_variables_initializer())

#tf.global_variables_initializer().run()
#merge = tf.summary.merge_all()
#writer = tf.summary.FileWriter("TensorBoard_data/Train")
#writer.add_graph(sess.graph)
#writer2 = tf.summary.FileWriter("TensorBoard_data/Test")


for i in range(150):
    print('New epoch')
    #batch_x = X_train[128*i:128*(i+1)]
    #batch_y = y_train[128*i:128*(i+1)]
    batch_x,batch_y = mnist.train.next_batch(100)
    batch_x = np.reshape(batch_x,[-1,28,28,1])
    print('Train')
    sess.run(optimizer, feed_dict={x:batch_x, y_true:batch_y})
#    writer.add_summary(s,i)
    print('Test')
    acc_new = sess.run(accuracy, feed_dict={x:batch_x, y_true:batch_y})
#    writer2.add_summary(s, i)
    print(acc_new)
    print(i)
