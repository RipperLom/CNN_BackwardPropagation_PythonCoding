import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np

def G(z, derive = False):
    if derive:
        return z * (1 - z)
    else:
        return 1 / (1 + np.exp(-z))


mnist = read_data_sets('data', one_hot=True)

w1_ = np.random.standard_normal([9, 9, 1, 20]) * 0.1
b1_ = np.random.standard_normal([1, 20]) * 0.1
w2_ = np.random.standard_normal([2000, 10]) * 0.1
b2_ = np.random.standard_normal([1, 10]) * 0.1

epochs = 10
batchSize = 100
learningRate = 1e-2

with tf.name_scope('convolve'):
    one_X_img = tf.placeholder(dtype=tf.float32, shape=[1, 28, 28, 1])
    convolve = tf.placeholder(dtype=tf.float32, shape=[20, 20, 1, 20])

    one_kernal = tf.nn.conv2d(one_X_img, convolve, strides=[1, 1, 1, 1], padding='VALID')
    one_kernal = tf.reshape(one_kernal, shape=[9, 9, 1, 20])

with tf.name_scope('layer1'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    w1 = tf.placeholder(dtype=tf.float32, shape=[9, 9, 1, 20])
    b1 = tf.placeholder(dtype=tf.float32, shape=[1, 20])

    X_img = tf.reshape(X, shape=[-1, 28, 28, 1])
    z1 = tf.nn.conv2d(X_img, w1, strides=[1, 1, 1, 1], padding='VALID') + b1
    s1 = tf.sigmoid(z1)
    L1 = tf.nn.avg_pool(s1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

with tf.name_scope('cost'):
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
    z2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z2, labels=Y))


with tf.name_scope('accuracy'):
    s2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    judge = tf.equal(tf.argmax(s2, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(judge, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(epochs):
    numOfBatch = int(mnist.train.num_examples / batchSize)
    avgCost = 0
    for j in range(numOfBatch):
        batchX, batchY = mnist.train.next_batch(batch_size=batchSize)
        # forwardPropagation
        # layer1 ++++++++++++++++++++++++++++++++++++++
        feed_dict = {X: batchX, w1: w1_, b1: b1_}
        z1_, s1_, L1_ = session.run([z1, s1, L1], feed_dict=feed_dict)
        L1_flat_ = np.reshape(L1_, newshape=[-1, 2000])
        # layer2 ++++++++++++++++++++++++++++++++++++++
        z2_ = np.dot(L1_flat_, w2_) + b2_
        s2_ = G(z2_)
        # cost ++++++++++++++++++++++++++++++++++++++++
        feed_dict = {Y: batchY, z2: z2_}
        cost_ = session.run(cost, feed_dict=feed_dict)

        # backwardPropagation
        m_ = batchSize
        # layer2 ++++++++++++++++++++++++++++++++++++++
        X_img_ = np.reshape(batchX, newshape=[-1, 28, 28, 1])
        dz2_ = s2_ - batchY
        dw2_ = (1 / m_) * np.dot(L1_flat_.T, dz2_)
        db2_ = np.mean(dz2_, axis=0)
        db2_ = np.reshape(db2_, newshape=[1, 10])

        # layer1 ++++++++++++++++++++++++++++++++++++++
        # ==========================
        dL1_flat_ = np.dot(dz2_, w2_.T)
        # ==========================
        dL1_ = np.reshape(dL1_flat_, newshape=[-1, 10, 10, 20])
        # ==========================
        ds1_ = np.zeros([m_, 20, 20, 20])
        for k in range(20):
            one_dL1_ = np.reshape(dL1_[:, :, :, k], newshape=[-1, 10, 10])
            one_ds1_ = np.kron(one_dL1_, np.ones([2, 2])).reshape([-1, 20, 20, 1]) / 4
            ds1_[:, :, :, k] = one_ds1_[:, :, :, 0]
        # ==========================
        dz1_ = ds1_ * s1_ * (1 - s1_)
        # ==========================
        dw1_ = np.zeros([9, 9, 1, 20])
        for m in range(m_):
            one_X_img_ = np.reshape(X_img_[m, :, :, :], newshape=[1, 28, 28, 1])
            convolve_ = np.reshape(dz1_[m, :, :, :], newshape=[20, 20, 1, 20])
            one_kernal_ = session.run(one_kernal, feed_dict={one_X_img: one_X_img_, convolve: convolve_})
            dw1_ += one_kernal_
        dw1_ = dw1_ / m_
        # ==========================
        db1_ = np.mean(np.reshape(dz1_, newshape=[-1, 20]), axis=0)
        db1_ = np.reshape(db1_, newshape=[1, 20])

        # update++++++++++++++++++++++++++++++++++++++++
        w1_ = w1_ - learningRate * dw1_
        b1_ = b1_ - learningRate * db1_
        w2_ = w2_ - learningRate * dw2_
        b2_ = b2_ - learningRate * db2_

        avgCost += cost_ / numOfBatch

    print('epoch', (i + 1), 'Cost', avgCost)


'''
000000 epoch 1 Cost 0.7285813373869118
000000 epoch 2 Cost 0.3221266726065763
000000 epoch 3 Cost 0.26143733019178567
000000 epoch 4 Cost 0.2178894449905915
000000 epoch 5 Cost 0.18599715965037983
000000 epoch 6 Cost 0.16401437470181424
000000 epoch 7 Cost 0.1460933880728079
000000 epoch 8 Cost 0.13309557309374226
000000 epoch 9 Cost 0.12221747296100305
000000 epoch 10 Cost 0.11367286562411623
accuracy 0.9695
'''

# layer1 +++++++++++++++++++
feed_dict = {X: mnist.test.images, w1: w1_, b1: b1_}
L1_ = session.run(L1, feed_dict=feed_dict)
L1_flat_ = np.reshape(L1_, newshape=[-1, 2000])

# layer2 +++++++++++++++++++
z2_ = np.dot(L1_flat_, w2_) + b2_
s2_ = G(z2_)

# acc
feed_dict = {Y: mnist.test.labels, s2: s2_}
acc = session.run(accuracy, feed_dict=feed_dict)
print('accuracy', acc)