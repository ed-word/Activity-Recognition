import numpy as np
import tensorflow as tf


max_num_of_frames = 250
from Gen_data import Gen_data
train_input, train_output, test_input, test_output = Gen_data(max_num_of_frames)


X = tf.placeholder(tf.float32, [None, max_num_of_frames, 230400])  #240*320*3 = 230400
Y = tf.placeholder(tf.float32, [None, 3])


num_hidden = 32
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
Cells, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


Cells = tf.transpose(Cells, [1, 0, 2])
b = tf.constant( int(Cells.get_shape()[0])-1, dtype=tf.int32 )
last_cell = tf.gather(Cells, b)


n_output = Y.get_shape()[1]
Output_layer = {
	'weight' :
	tf.Variable(tf.truncated_normal([num_hidden, int(n_output)])),
	'bias' :
	tf.Variable(tf.constant(0.001, shape=[n_output]))
}


y = tf.matmul(last_cell,Output_layer['weight']) + Output_layer['bias']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))


lr = 0.01
optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




batch_size = 1
no_of_batches = int(len(train_input) / batch_size)
epoch = 50

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epoch):
		print("\nEpoch ",i)
		ptr = 0
		for j in range(no_of_batches):
			inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
			ptr+=batch_size
			err,_,acc = sess.run([cross_entropy,optimizer,accuracy], feed_dict={X: inp, Y: out})
			print("Error: ",err,"\tAccuracy: ",acc)


acc_list = []
print("\nTESTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i,_ in enumerate(test_input):
		temp = accuracy.eval(feed_dict={X: test_input[i-1:i], Y: test_output[i-1:i]})
		acc_list.append(temp)
		print("Accuracy: ", temp)

acc_list = np.nan_to_num(np.array(acc_list))
print("Final Acc: ", np.mean(np.array(acc_list)))