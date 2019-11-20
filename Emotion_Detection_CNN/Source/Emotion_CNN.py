#<editor-fold> Import Statements
import matplotlib.pyplot as plot
#%matplotlib inline

import numpy

#import tensorflow
import tensorflow.compat.v1 as tensorflow
tensorflow.disable_v2_behavior()
#</editor-fold> Import Statements

#<editor-fold> Unzipping the file and getting each picture.
CIFAR_DIR = "./cifar-10-batches-py/"

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        cifar_dict = pickle.load(fo, encoding = "bytes")

    return cifar_dict
#</editor-fold> Unzipping the file and getting each picture.


#<editor-fold> Initial Set Up
dirs = ["batches.meta", "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

all_data = [0, 1, 2, 3, 4, 5, 6]

for i, direc in zip(all_data, dirs):
    all_data[i] = unpickle(CIFAR_DIR + direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

#print(CIFAR_DIR + direc)
#print(batch_meta)

X = data_batch1[b"data"]
#Take the data of 10,000 pictures of 3 colors of size 32 by 32 and
#reshape the X array so that it now holds 10,000 pictures that are 32 by 32 with colors.
#Also limit the data type in the array to 8 bit integers.
X = X.reshape(10000, 3, 32, 32).transpose(0, 3, 2, 1).astype("uint8")
print(X[0].max())
print((X[0] / 255).max())
#</editor-fold> Initial Set Up

#<editor-fold> Helper Functions
def one_hot_encode(vector, values = 10):
    """
    For use to one hot encode the 10 possible labels.
    This will translate the labels from words/strings to a string of ints.
    Helps the CNN determine which label to apply.
    """
    n = len(vector)
    out = numpy.zeros( (n, values) )
    out[range(n), vector] = 1
    return out

def init_weights(shape):
    init_random_dist = tensorflow.truncated_normal(shape, stddev = 0.1)
    return tensorflow.Variable(init_random_dist)

def init_bias(shape):
    init_bias_values = tensorflow.constant(0.1, shape = shape)
    return tensorflow.Variable(init_bias_values)

def conv2d(x, W):
    return tensorflow.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = "SAME" )

def max_pool_2by2(x):
    return tensorflow.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias( [shape[3]] )
    return tensorflow.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int( input_layer.get_shape()[1] )
    W = init_weights( [input_size, size] )
    b = init_bias( [size] )
    return tensorflow.matmul(input_layer, W) + b

#</editor-fold> Helper Functions

#<editor-fold> CIFAR Class
class CIFARHelper():

    def __init__(self):
        self.i = 0

        self.all_training_batches = [data_batch1, data_batch2, data_batch3, data_batch4, data_batch5]
        self.test_batch = [test_batch]

        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def set_up_images(self):
        print("Setting up the training images and labels!")

        self.training_images = numpy.vstack( [ d[b"data"] for d in self.all_training_batches ] )
        training_images_length = len(self.training_images)

        self.training_images = self.training_images.reshape(training_images_length, 3, 32, 32).transpose(0, 3, 2, 1) / 255
        self.training_labels = one_hot_encode(numpy.hstack( [ d[b"labels"] for d in self.all_training_batches ] ), 10)

        print("Setting up the testing images and labels!")

        self.test_images = numpy.vstack( [ d[b"data"] for d in self.test_batch ] )
        testing_images_length = len(self.test_images)

        self.test_images = self.test_images.reshape(testing_images_length, 3, 32, 32).transpose(0, 3, 2, 1) / 255
        #CNN will give the labels for the testing batch.
        self.test_labels = one_hot_encode(numpy.hstack( [ d[b"labels"] for d in self.test_batch ] ), 10)

    def next_batch(self, batch_size):
        x = self.training_images[self.i: self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i: self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y
#</editor-fold> CIFAR Class


#<editor-fold> Main Program
ch = CIFARHelper()
ch.set_up_images()

x = tensorflow.placeholder( tensorflow.float32, shape = [None, 32, 32, 3] )
y_true = tensorflow.placeholder( tensorflow.float32, shape = [None, 10] )

hold_prob = tensorflow.placeholder(tensorflow.float32)

#Create the convolutional layers
convo_1 = convolutional_layer(x, shape = [4, 4, 3, 32])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape = [4, 4, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)
#Done creating the convolutional layers.

#Flatten the output to a 1D vector.
convo_2_flat = tensorflow.reshape(convo_2_pooling, [-1, 8 * 8 * 64])

#Make the fully connected layers.
full_layer_one = tensorflow.nn.relu(normal_full_layer(convo_2_flat, 1024))
full_one_dropout = tensorflow.nn.dropout(full_layer_one, keep_prob = hold_prob)
#Done making the fully connected layers.

#Set up output
y_pred = normal_full_layer(full_one_dropout, 10)

#Apply loss function
cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred))

#Create optimizer
optimizer = tensorflow.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cross_entropy)

#Create a variable to initalize all the global tensorflow variables.
init = tensorflow.global_variables_initializer()

#Run the CNN by using a graph session.
with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())

    for i in range(500):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict = {x: batch[0], y_true: batch[1], hold_prob: 0.5 })

        #Print out a message every 100 steps.
        if i % 100 == 0:
            print("Currently on step {}".format(i))
            print("Accuracy is: ")

            #Test the train model
            matches = tensorflow.equal(tensorflow.argmax(y_pred, 1), tensorflow.argmax(y_true, 1))
            acc = tensorflow.reduce_mean(tensorflow.cast(matches, tensorflow.float32))

            print( sess.run(acc, feed_dict = {x: ch.test_images, y_true: ch.test_labels, hold_prob: 1.0}) )
            print("\n")

#Displaying an image from our batch.
# plot.imshow(X[0])
# plot.show()
#plot.close()
#</editor-fold> Main Program
