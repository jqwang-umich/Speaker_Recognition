import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

train_X = np.load('trainingData.npy')
test_X = np.load("testingData.npy")
train_y = np.load('trainingLabel.npy')
test_y = np.load("testingLabel.npy")

training_iters = 100
learning_rate = 0.0001
batch_size = 256

#shape:28*28
n_input = 28
#(0-10)
n_classes = 11

#both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2(x):
    return tf.nn.avg_pool(x,ksize=[1,2,2,1],  strides=[1, 2, 2, 1], padding='SAME')

def addLayer(input,inputLayer,outputLayer,pool = False):
    Weight_conv1 = weight_variable([ para_size_conv, para_size_conv, inputLayer, outputLayer ])
    b_conv1 = bias_variable([ outputLayer ])
    h_conv1 = tf.nn.relu(conv2d(input, Weight_conv1) + b_conv1)
    if pool:
        h_pool1 = avg_pool_2(h_conv1)
    else:
        h_pool1 = h_conv1
    return h_pool1

para_size_input = 28
para_size_conv = 3
para_size_layer1 = 2
para_size_layer2 = 2
para_size_layer3 = 2
para_size_layer4 = 2
para_size_pool = 2

output_conv = 0
para_size_layer_end = para_size_layer3
para_fc_image_size = para_size_input

def conv_net(x):

    output1 = addLayer(x, 1, para_size_layer1)
    output2 = addLayer(output1, para_size_layer1, para_size_layer2)
    output3 = addLayer(output2, para_size_layer2, para_size_layer3)

    output_conv = output3
    para_size_layer_end = para_size_layer3
    para_fc_image_size = para_size_input

    Weight_full_connection = weight_variable([ para_fc_image_size * para_fc_image_size * para_size_layer_end,2 ])
    output_flat = tf.reshape(output_conv, [ -1, para_fc_image_size * para_fc_image_size * para_size_layer_end ])
    h_fc1 = tf.nn.relu(tf.matmul(output_flat, Weight_full_connection))

    W_fc2 = weight_variable([2, 11 ])
    out = tf.nn.softmax(tf.matmul(h_fc1, W_fc2))

    return out

pred = conv_net(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here you check whether the index of the maximum value of the predicted is equal to the actual labelled
# and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
#
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]
#             print(batch_x.shape)
#             print(batch_y.shape)
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!"+str(i))

        # Calculate accuracy for all test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

    np.save("train_loss.npy",train_loss)
    np.save("train_accuracy.npy",train_accuracy)
    np.save("test_loss.npy", test_loss)
    np.save("test_accuracy.npy", test_accuracy)