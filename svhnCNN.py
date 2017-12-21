from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
A = loadmat('train_32x32')
B = loadmat('test_32x32')

X_train = A['X']
y_train = A['y']
X_test = B['X']
y_test = B['y']


# https://github.com/thomalm/svhn-multi-digit
# un bon lien pour se baser


#########################################################
# Preprocessing

# Le chiffre 0 est associé à l'étiquette 10, on lui associe l'étiquette 0
y_train[y_train==10] = 0
y_test[y_test==10] = 0



# Pour faciliter l'accès aux données
X_train = X_train.transpose(3,0,1,2).astype(np.float32)
X_test = X_test.transpose(3,0,1,2).astype(np.float32)

train_mean = np.mean(X_train,axis=0)
train_std = np.std(X_train,axis=0)

X_train = (X_train-train_mean)/train_std
X_test = (X_test-train_mean)/train_std

X_train,y_train = shuffle(X_train,y_train)
X_test,y_test = shuffle(X_test,y_test)
indexes = np.arange(73257)


# Permet de d'avoir un vecteur r pour chaque donnée
enc = OneHotEncoder().fit(y_train)
y_train = enc.transform(y_train).toarray().astype(np.float32)
y_test = enc.transform(y_test).toarray().astype(np.float32)

print(np.shape(X_train))
print(np.shape(y_train))

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 10])
k = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

tf.summary.image('Data_in', x, 3)
# Building simple model
with tf.name_scope('Conv_1'):
    weights1 = tf.Variable(tf.truncated_normal([3, 3, 3, 50], stddev=0.05))
    biases1 = tf.Variable(tf.constant(0.05, shape=[50]))
    conv1 = tf.nn.conv2d(x, weights1, strides=[1,1,1,1], padding='SAME')
    conv1_out = tf.nn.relu(conv1+biases1)
    weights11 = tf.Variable(tf.truncated_normal([3, 3, 50, 50], stddev=0.05))
    biases11 = tf.Variable(tf.constant(0.05, shape=[50]))
    conv11 = tf.nn.conv2d(conv1_out, weights11, strides=[1,1,1,1], padding='SAME')
    conv11_out = tf.nn.relu(conv11+biases11)
    pool1 = tf.nn.max_pool(conv11_out, [1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tf.summary.histogram('Weights1', weights1)

with tf.name_scope('Conv_2'):
    weights2 = tf.Variable(tf.truncated_normal([3, 3, 50, 50],stddev=0.05))
    biases2 = tf.Variable(tf.constant(0.05, shape=[50]))
    conv2 = tf.nn.conv2d(pool1, weights2, strides=[1,1,1,1], padding='SAME')
    conv2_out = tf.nn.relu(conv2+biases2)
    conv2_d = tf.nn.dropout(conv2_out, k)
    pool2 = tf.nn.max_pool(conv2_d, [1,2,2,1], strides=[1,2,2,1], padding='SAME')
    tf.summary.histogram('Weights2', weights2)

flat_out = tf.reshape(pool2, [-1, 8*8*50])

with tf.name_scope('Fully_1'):
    weights3 = tf.Variable(tf.truncated_normal([8*8*50, 128],stddev=0.03))
    biases3 = tf.Variable(tf.constant(0.05, shape=[128]))
    relu3 = tf.nn.relu(tf.matmul(flat_out, weights3)+biases3)
    relu3_d = tf.nn.dropout(relu3, k)
    tf.summary.histogram('Weights3', weights3)

with tf.name_scope('Fully_2'):
    weights4 = tf.Variable(tf.truncated_normal([128, 64],stddev=0.03))
    biases4 = tf.Variable(tf.constant(0.05, shape=[64]))
    relu4 = tf.nn.relu(tf.matmul(relu3_d, weights4)+biases4)
    relu4_d = tf.nn.dropout(relu4, k)

with tf.name_scope('Fully_3'):
    weights5 = tf.Variable(tf.truncated_normal([64, 10],stddev=0.05))
    biases5 = tf.Variable(tf.constant(0.05, shape=[10]))
    relu5 = tf.matmul(relu4_d, weights5)+biases5
    tf.summary.histogram('Relu5', relu5)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=relu5))
optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
tf.summary.scalar('Entropy', cross_entropy)


correct_pred = tf.equal(tf.argmax(relu5, -1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('Accuracy', accuracy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
merge = tf.summary.merge_all()
writer = tf.summary.FileWriter("TensorBoard_data_SVHN/Train")
writer.add_graph(sess.graph)
writer2 = tf.summary.FileWriter("TensorBoard_data_SVHN/Test")


for i in range(1000):
    print('New epoch')
    if i > 400:
        learning_rate = 0.0005
    else:
        learning_rate = 0.003
    batch_x = X_train[indexes[:128]]
    batch_y = y_train[indexes[:128]]
    #batch_x, batch_y = mnist.train.next_batch(128)
    #batch_x = np.reshape(batch_x,[-1,28,28,1])
    print('Train')
    s, _ = sess.run([merge, optimizer], feed_dict={x:batch_x, y_true:batch_y, k:0.8, lr:learning_rate})
    writer.add_summary(s,i)
    print('Test')
    #s, acc_new = sess.run([merge, accuracy], feed_dict={x:np.reshape(mnist.test.images,[-1,28,28,1])[:300], y_true:mnist.test.labels[:300]})
    s, acc_new = sess.run([merge, accuracy], feed_dict={x:X_test[:200], y_true:y_test[:200], k:1})
    writer2.add_summary(s, i)
    np.random.shuffle(indexes)
    print(acc_new)
    print(i)
