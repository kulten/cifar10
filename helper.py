import _pickle as Pickle
import numpy as np
import tensorflow as tf
import math
def extract(file):
    fo = open(file, 'rb')
    x = Pickle.load(fo,encoding='bytes')
    fo.close()
    return x

def pre_process_data(file,classes):
    dict = extract(file)
    x_dict = np.array(dict[b'data'],dtype=float)
    x = x_dict.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8")
    y_raw = np.array(dict[b'labels'],dtype=float)
    with tf.Session() as sess:
        y = sess.run(tf.one_hot(y_raw,classes))
    return x,y

def batching(x,y,size):
    batches = []
    m = x.shape[0]
    num_batches = math.floor(m/size)
    for i in range(0,num_batches):
        x_batch = x[i*size:(i+1)*size,:,:,:]
        y_batch = y[i*size:(i+1)*size,:]
        batch = (x_batch,y_batch)
        batches.append(batch)
    if m % size !=0:
        x_batch = x[num_batches*size:,:,:,:]
        y_batch = y[num_batches*size:,:]
        batch = (x_batch,y_batch)
        batches.append(batch)
    return batches

def weights(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)

def biases(shape):
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)
