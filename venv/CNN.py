import numpy as np
import tensorflow as tf

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

def nextBatch(n):
    listX = []
    listY = []
    data = np.load('trainingData.npy')
    label = np.load('trainingLabel.npy')
    length = len(data)
    num = [i for i in range(length)]
    np.random.shuffle(num)
    for i in range(n):
        listX.append(data[num[i]])
        listY.append(label[num[i]])
    return listX,listY

def addLayer(input,inputLayer,outputLayer,pool = False):
    Weight_conv1 = weight_variable([ para_size_conv, para_size_conv, inputLayer, outputLayer ])
    b_conv1 = bias_variable([ outputLayer ])
    h_conv1 = tf.nn.relu(conv2d(input, Weight_conv1) + b_conv1)
    if pool:
        h_pool1 = avg_pool_2(h_conv1)
    else:
        h_pool1 = h_conv1
    return h_pool1

# session = tf.InteractiveSession(config=tf.ConfigProto(device_count={'gpu':0}))
session = tf.InteractiveSession()
listTestAcc = []
listTrainAcc = []
trainingData = np.load("trainingData.npy")
testingData = np.load("testingData.npy")
trainingLabel = np.load("trainingLabel.npy")
testingLabel = np.load("testingLabel.npy")

para_size_input = 28
para_size_conv = 3
para_size_layer1 = 32
para_size_layer2 = 64
para_size_layer3 = 128
para_size_layer4 = 128
para_size_pool = 2


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# print(x.shape)
x_image = tf.reshape(x,[-1,para_size_input,para_size_input,1])

output1 = addLayer(x_image,1,para_size_layer1)
output2 = addLayer(output1,para_size_layer1,para_size_layer2)
output3 = addLayer(output2,para_size_layer2,para_size_layer3)

output_conv = output3
para_size_layer_end = para_size_layer3
para_fc_image_size = para_size_input
# output4 =
# Weight_conv1 = weight_variable([para_size_conv,para_size_conv,1,para_size_layer1])
# b_conv1 = bias_variable([para_size_layer1])
# h_conv1 = tf.nn.relu(conv2d(x_image,Weight_conv1)+ b_conv1)
# # h_pool1 = avg_pool_2(h_conv1)
# h_pool1 = h_conv1
#
# Weight_conv2 = weight_variable([para_size_conv,para_size_conv,para_size_layer1,para_size_layer2])
# b_conv2 = bias_variable(([para_size_layer2]))
# h_conv2 = tf.nn.relu(conv2d(h_pool1,Weight_conv2)+b_conv2)
# # h_pool2 = avg_pool_2(h_conv2)
# h_pool2 = h_conv2
#
# Weight_conv3 = weight_variable([para_size_conv,para_size_conv,para_size_layer2,para_size_layer3])
# b_conv3 = bias_variable((para_size_layer3))
# h_conv3 = tf.nn.relu(conv2d(h_pool2,Weight_conv3)+b_conv3)
# h_pool3 = h_conv3

Weight_full_connection = weight_variable([para_fc_image_size * para_fc_image_size * para_size_layer_end,1024])
output_flat = tf.reshape(output_conv,[-1,para_fc_image_size * para_fc_image_size * para_size_layer_end])
h_fc1 = tf.nn.relu(tf.matmul(output_flat,Weight_full_connection))

W_fc2 = weight_variable([1024,11])
y_out = tf.nn.softmax(tf.matmul(h_fc1,W_fc2))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    # batchX, batchY = nextBatch(40)
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: trainingData, y: trainingLabel})
        test_accuracy = accuracy.eval(feed_dict={x:testingData, y:testingLabel})
        print("test data ",i,": ",test_accuracy)
        print("train data ",i,": ",train_accuracy)
        listTestAcc.append(test_accuracy)
        listTrainAcc.append(train_accuracy)
    if i%100 == 0:
        np.save('BatchedTestAcc',listTestAcc)
        np.save('BatchedTrainAcc',listTrainAcc)
        # print("saved")
    # print(i)

    train_step.run(feed_dict = {x:trainingData, y:trainingLabel})


print("test accuracy %g"%accuracy.eval(feed_dict={x:testingData, y:testingLabel}))
np.save('BatchedTestAcc',listTestAcc)
np.save('BatchedTrainAcc',listTrainAcc)


