import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from helper import pre_process_data, extract, batching, weights, biases
'''
I put all the helper functions in helper.py to make the code cleaner
'''

def measure_accuracy(batch,to_keep):
    accuracies = []
    for minibatches in batch:
        (X_batch,Y_batch) = minibatches
        temp_accuracies = sess.run(accuracy,feed_dict={X: X_batch,Y: Y_batch,prob:to_keep})
        accuracies.append(temp_accuracies)
    num = len(accuracies)
    final = round(sum(accuracies)/num,3)
    return final

LEARNING_RATE = 0.0006
VALIDATION_SIZE = 2500
BATCH_SIZE = 64
EPOCHS = 10
DROP_OUT = 0.8
img_cat = 10 #total number of unique class labels
x_1,y_1 = pre_process_data('data_batch_1',img_cat)
x_2,y_2 = pre_process_data('data_batch_2',img_cat)
x_3,y_3 = pre_process_data('data_batch_3',img_cat)
x_4,y_4 = pre_process_data('data_batch_4',img_cat)
x_5,y_5 = pre_process_data('data_batch_5',img_cat)
X_test,Y_test = pre_process_data('test_batch',img_cat)
images_raw = np.concatenate((x_1,x_2,x_3,x_4,x_5)) #combine all the processed images from the seperate files
labels = np.concatenate((y_1,y_2,y_3,y_4,y_5)) #combine all the processed labels from the seperate files
images =images_raw/255 #normalize the image
img_h = images.shape[1] #image height
img_w = images.shape[2] #image width
img_c = images.shape[3] #number of channels
X_validation =images[:VALIDATION_SIZE]
Y_validation =labels[:VALIDATION_SIZE]
X_train = images[VALIDATION_SIZE:]
Y_train = labels[VALIDATION_SIZE:]
validation_batch = batching(X_validation,Y_validation,BATCH_SIZE)
training_batch = batching(X_train,Y_train,BATCH_SIZE)
testing_batch = batching(X_test,Y_test,BATCH_SIZE)
'''
Model Architecture:
 Conv => relu => pool => dropout => Conv => relu => pool => dropout => Conv => relu => pool => dropout (continued in next line)
 linear => relu => dropout => linear => relu => linear => softmax
'''
X = tf.placeholder('float',shape = [None,img_h,img_w,img_c],name="X")
Y = tf.placeholder('float',shape = [None,img_cat],name="Y")
prob = tf.placeholder('float', shape=()) #placeholder to feed in the desired value of DROP_OUT, with a value of 1 during testing
W1 = weights([3,3,3,32])
conv_1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME',name = "conv_1")
relu_1 = tf.nn.relu(conv_1,name = "relu_1")
pool_1 = tf.nn.max_pool(relu_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME',name = "pool_1")
dropout_1 = tf.nn.dropout(pool_1,prob,name = "dropout_1")
W2 = weights([3,3,32,64])
conv_2 = tf.nn.conv2d(dropout_1,W2, strides = [1,1,1,1], padding = 'SAME',name = "conv_2")
relu_2 = tf.nn.relu(conv_2,name = "relu_2")
pool_2 = tf.nn.max_pool(relu_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME',name = "pool_2")
dropout_2 = tf.nn.dropout(pool_2,prob,name = "dropout_2")
W3 = weights([3,3,64,128])
conv_3 = tf.nn.conv2d(dropout_2,W3, strides = [1,1,1,1], padding = 'SAME',name = "conv_3")
relu_3 = tf.nn.relu(conv_3,name = "relu_3")
pool_3 = tf.nn.max_pool(relu_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME',name = "pool_3")
dropout_3 = tf.nn.dropout(pool_3,prob,name = "dropout_4")
flat = tf.contrib.layers.flatten(dropout_3)
linear_1 = tf.contrib.layers.fully_connected(flat,1024,activation_fn=None)
relu_4 = tf.nn.relu(linear_1,name = "relu_4")
dropout_4 = tf.nn.dropout(relu_4,prob,name = "dropout_5")
W4 = weights([1024,150])
b4 = biases([150])
linear_2 = tf.matmul(dropout_4, W4) + b4
relu_5 = tf.nn.relu(linear_2)
W5 = weights([150,img_cat])
b5 = biases([img_cat])
linear_3 = tf.matmul(relu_5, W5) + b5 #raw values with no activation
y_hat = tf.nn.softmax(linear_3,name = "y_hat")
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = linear_3, labels = Y),name ="cost")
optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(Y,1),name ="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'),name ="accuracy")
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)
costs = [] #store costs
model_accuracy = [] #store validation accuracy
num_b = len(training_batch) - 2 #find the number of batches to generate a random number
with tf.Session() as sess:
    sess.run(init)
    for i in range(EPOCHS):
        for batches in training_batch:
            (X_batch,Y_batch) = batches
            _ , temp_cost = sess.run([optimize,cost],feed_dict={X:X_batch,Y:Y_batch,prob:DROP_OUT})
        rand = np.random.randint(0,num_b) #generate a random number
        (X_rand_batch,Y_rand_batch) = training_batch[rand] #sample a random batch from training set
        train_acc = sess.run(accuracy,feed_dict={X:X_rand_batch, Y:Y_rand_batch,prob:1.0})
        valid_ac = measure_accuracy(validation_batch,1)
        temp_cost = round(float(temp_cost),5)
        train_acc = round(float(train_acc),3)
        costs.append(temp_cost)
        model_accuracy.append(valid_ac)
        print("EPOCH: {0} Cost: {1} TRAINING ACCURACY: {2} VALIDATION ACCURACY: {3}".format(i,temp_cost,train_acc,valid_ac))
    test_ac = measure_accuracy(testing_batch,1.0) #test set accuracy
    final_train_ac = measure_accuracy(training_batch,1.0) #final train set accuracy
    print("TEST SET ACCURACY: {0} TRAIN SET ACCURACY: {1}".format(round(test_ac,3),round(final_train_ac,3)))
    print("Save model? yes/no")
    selection = input()
    if(selection == "yes"):
        saver.save(sess,"model/cifar10.ckpt") #saves the model, did not upload model directory because of large file sizes 
    #display cost graph
    plt.figure(1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('epoch')
    plt.title("Learning rate = " + str(LEARNING_RATE)+" Drop out = "+str(DROP_OUT))
    if(selection == "yes"):
        plt.savefig('cost.png')
    #display accuracy graph
    plt.figure(2)
    plt.plot(np.squeeze(model_accuracy))
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.title("Learning rate = " + str(LEARNING_RATE)+" Drop out = "+str(DROP_OUT))
    if(selection == "yes"):
        plt.savefig('accuracy.png')
    plt.show()
