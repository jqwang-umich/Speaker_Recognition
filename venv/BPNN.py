import tensorflow as tf
import numpy as np

# sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def addLayer(input, inputSize, outputSize, stimulate = 1):
    W1 = weight_variable([ inputSize,outputSize ])
    b1 = bias_variable([ outputSize])
    a1 = tf.matmul(input,W1) + b1
    if stimulate == 0:
        z1 = a1
    else:
        z1 = tf.nn.relu(a1)
    return z1

    return z1

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


sess = tf.InteractiveSession(config=tf.ConfigProto(device_count={'gpu':0}))
listTestAcc = []
listTrainAcc = []
# plan to build a BPNN with 2 hidden layers, size m1 and m2 are parameters
# input reshaped to one column

m1 = 1024
m2 = 1024
size1 = 16
size2 = 16

testingData = np.load("testingData.npy")
testingLabel = np.load("testingLabel.npy")
data = np.load('trainingData.npy')
label = np.load('trainingLabel.npy')

x = tf.placeholder(tf.float32,[None,size1,size2,1])
y = tf.placeholder(tf.float32)
x_vector = tf.reshape(x,shape = [-1,size1*size2])

W1 = weight_variable([ size1*size2,m1 ])
b1 = bias_variable([ m1 ])
a1 = tf.matmul(x_vector, W1) + b1
z1 = tf.nn.relu(a1)

W2 = weight_variable([ m1,m2 ])
b2 = bias_variable([ m2 ])
a2 = tf.matmul(z1, W2) + b2
z2 = tf.nn.relu(a2)


W3 = weight_variable([ m2,11 ])
b3 = bias_variable([ 11 ])
a3 = tf.matmul(z2, W3) + b3
z3 = a3

y_out = tf.nn.softmax(z3)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_out),reduction_indices=[0]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

for i in range(20000):
    # batchX, batchY = nextBatch(50)
    if i%20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: data, y: label})
        test_accuracy = accuracy.eval(feed_dict={x:testingData, y:testingLabel})
        print("test data ",i,": ",test_accuracy)
        print("train data ",i,": ",train_accuracy)
        listTestAcc.append(test_accuracy)
        listTrainAcc.append(train_accuracy)
    # if i%100 == 0:
    #     np.save('BatchedTestAcc',listTestAcc)
    #     np.save('BatchedTrainAcc',listTrainAcc)
    #     # print("saved")
    # # print(i)

    train_step.run(feed_dict = {x:data, y:label})


print("test accuracy %g"%accuracy.eval(feed_dict={x:testingData, y:testingLabel}))
np.save('BPNNBatchedTestAcc',listTestAcc)
np.save('BPNNBatchedTrainAcc',listTrainAcc)
