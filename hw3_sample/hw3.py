import os
import glob
import tensorflow as tf
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

DATA_DIR = "./CroppedYale" # original dataset folder name
PNG_DATA_DIR = "DATA_PNG" # new dataset folde name
NUM = 35 # each class train data number
WIDTH = 120
HEIGHT = 165
CHANNELS = 1
NUM_CLASSES = 100 # output classes number
BATCH_SIZE = 5 # batch size
ITERATION = 100 # iteration number
TEST_NUM = 1122 # test data number
KEEP_NUM = 0.7 # the probability of each neuron being kept

def Image2PNG():
	for index, classes in enumerate(os.listdir(DATA_DIR)): # read all object name in DATA_DIR to classes
		fullPath = os.path.join(DATA_DIR, classes, "") # combine PATH
		if os.path.isdir(fullPath): # if PATH is not a folder then continue
			pass
		else:
			continue
		globList = glob.glob(fullPath+'*.pgm') # find all filename extension is .bmp to globList
		print (fullPath)
		length_dir = len(fullPath) # cal length of fullPath
		i = 1
		for glob_path in globList: 
			label = int(glob_path[length_dir+5:length_dir+7])-1 # set label from image name
			if i <= 35: # save to another folder become a new dataset
				Image.open(glob_path).save(
							PNG_DATA_DIR+'\\train\\'+str(label).zfill(3)
							+'-label_'+str(i).zfill(2)+'-num'+'.png')
			else:
				if glob_path[length_dir+12:length_dir+13] == 'Am':
					pass
				Image.open(glob_path).save(
							PNG_DATA_DIR+'\\test\\'+str(label).zfill(3)
							+'-label_'+str(i).zfill(2)+'-num'+'.png')
			i += 1

def read_DatasList(data_type):
	fullLabel = []
	fullPath = os.path.join(PNG_DATA_DIR, data_type, "")
	globList = glob.glob(fullPath+'*.png')
	length_dir = len(fullPath)
	for glob_path in globList:
		fullLabel.append(int(glob_path[length_dir:length_dir+3])) # append all label to fullList
	images = tf.convert_to_tensor(globList, dtype=tf.string)
	labels = tf.one_hot(fullLabel, NUM_CLASSES) # change label to one hot type
	return images, labels

def read_Data_to_batch(fullList, fullLabel):
	input_queue = tf.train.slice_input_producer( # slice input
									[fullList, fullLabel]
									# , shuffle=False
									)
	file_content = tf.read_file(input_queue[0]) # read PATH
	imgs = tf.image.resize_images(  # decode image and resize
					tf.image.decode_png(file_content), [WIDTH, HEIGHT])
	labels = input_queue[1]
	imgs.set_shape([WIDTH, HEIGHT, CHANNELS]) # setting image shape
	return imgs, labels

def next_batch(img, label, batch_size=BATCH_SIZE):
	image_batch, label_batch = tf.train.batch( # setting batch size
								[img, label], 
								batch_size=batch_size
								)

	# image_batch, label_batch = tf.train.shuffle_batch(
	# 							[img, label], 
	# 							batch_size=batch_size, 
	# 							capacity=capacity, 
	# 							min_after_dequeue=min_after_dequeue
	# 							)

	return image_batch, label_batch

def weight_variable(shape): # random weight
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape): # random bias
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, strides_size, padding_type="VALID"): # convolution
	return tf.nn.conv2d(x, W,  # strides = [batch, width, height, channels]
				strides=[1, strides_size, strides_size, 1], padding=padding_type)

def max_pooling(x, k_size, strides_size, padding_type="VALID"): # pooling
	return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1],  # ksize = [batch, width, height, channels]
				strides=[1, strides_size, strides_size, 1], padding=padding_type)

def CNNet(x, keep_prob):
	# input layer
	x_input = tf.reshape(x, [-1, WIDTH, HEIGHT, CHANNELS]) # 120*165*3
	# hidden layer 1
	w_conv1 = weight_variable([7, 7, CHANNELS, 16])
	b_conv1 = bias_variable([16])
	h_conv1 = tf.nn.relu(conv2d(x_input, w_conv1, 1, "VALID") + b_conv1) # 114*159*16
	h_pool1 = max_pooling(h_conv1, 3, 3, "SAME") # 38*53*16
	# hidden layer 2
	w_conv2 = weight_variable([3, 3, 16, 32])
	b_conv2 = bias_variable([32])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 1, "VALID") + b_conv2) # 36*51*32
	h_pool2 = max_pooling(h_conv2, 3, 3, "SAME") # 12*17*32
	# fully connected layer 1
	h_pool2_flat = tf.reshape(h_pool2, [-1, 12*17*32])
	w_fc1 = weight_variable([12*17*32, 512])
	b_fc1 = bias_variable([512])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	# fully connected layer 2
	w_fc2 = weight_variable([512, NUM_CLASSES])
	b_fc2 = bias_variable([NUM_CLASSES])
	h_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
	# output layer
	prediction = tf.nn.softmax(h_fc2)
	return prediction

def main():
	# read datalist or CSV
	fullList_train, fullLabel_train = read_DatasList('train')
	fullList_test, fullLabel_test = read_DatasList('test')
	# read data to a slice/batch
	train_img, train_label = read_Data_to_batch(fullList_train, fullLabel_train)
	test_img, test_label = read_Data_to_batch(fullList_test, fullLabel_test)
	# read the next batch
	train_image_batch, train_label_batch = next_batch(train_img, train_label, BATCH_SIZE)
	test_image_batch, test_label_batch = next_batch(test_img, test_label, TEST_NUM)
	# placeholder
	keep_prob = tf.placeholder(tf.float32) # dropout probability
	x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNELS])
	y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
	# call to Net
	y_prediction = CNNet(x, keep_prob)
	# loss
	cost = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(y, y_prediction)) # loss
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cost) # optimizer
	# accuracy
	correct = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
	# initializer
	init_g = tf.global_variables_initializer()
	# session
	with tf.Session() as sess:
		# initialize variables
		sess.run(init_g)
		# create a corrdinator
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# iteration
		for i in range(ITERATION):
			if coord.should_stop():
				print('corrd break!!!!!!')
				break
			# load train data batch
			example_train, l_train = sess.run([train_image_batch, train_label_batch])
			# train
			_, loss = sess.run([train_step, cost], 
							feed_dict={x: example_train, y: l_train, keep_prob: KEEP_NUM})
			if (i % 10 == 0) & (i != 0):
				# load test data batch
				example_test, l_test = sess.run([test_image_batch, test_label_batch])
				# test accuracy
				test_acc = sess.run(accuracy, 
								feed_dict={x: example_test, y: l_test, keep_prob: 1})
				print('iter: ', i)
				print('loss: ', loss)
				print('test_acc: ', test_acc)
		coord.request_stop()
		coord.join(threads)

if tf.gfile.Exists(PNG_DATA_DIR): # if the path already exists do nothing
	pass
else:
	tf.gfile.MakeDirs(PNG_DATA_DIR)
	tf.gfile.MakeDirs(PNG_DATA_DIR+'\\train')
	tf.gfile.MakeDirs(PNG_DATA_DIR+'\\test')
	Image2PNG()

main()
