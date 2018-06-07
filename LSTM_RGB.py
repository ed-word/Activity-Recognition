import numpy as np
import tensorflow as tf
import os
import pandas as pd

#Training Hyper-parameters
batch_size = 128
epoch = 1000
learning_rate = 0.001
display_step = 10

#Network Hyper-parameters
num_layers = 5
n_hidden = 128
n_classes = 2
max_num_of_frames = 256




#Network
#Input
x = tf.placeholder(tf.float32, [None, max_num_of_frames, 2048], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')
seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

#Cost-Sensitive
w = tf.constant([[0,30],[10,0]], dtype=tf.float32)
TP = w[1][1]
TN = w[0][0]
FP = w[0][1]
FN = w[1][0]

#Weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='w')
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]), name='b')
}

#LSTM
def dynamicRNN(x, seq_len, weights, biases):
    
    x = tf.transpose(x, [1,0,2])
    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, sequence_length=seq_len, time_major=True)

    outputs = tf.transpose(outputs, [1, 0, 2])
    b_size = tf.shape(outputs)[0]
    index = tf.range(0, b_size)*max_num_of_frames + (seq_len - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seq_len, weights, biases)

# Define loss and optimizer
prediction = tf.to_float(tf.argmax(pred,1), name='prediction')
actual = tf.to_float(tf.argmax(y,1), name='actual')

cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = TP*cost*prediction*actual + FP*cost*prediction*(1-actual) + FN*cost*(1-prediction)*(actual) + TN*cost*(1-prediction)*(1-actual)
cost = tf.reduce_mean(cost, name='cost')

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, name='optimizer')

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


#Data Generation
dirrr = os.getcwd()
featdir = dirrr + '/FeaturesRGB'
labeldir = dirrr + '/Labels/X.csv'

labels = pd.read_csv(labeldir)
Xlabel = labels['Path']
Ylabel = labels['ChangeOfPace']
Ylabel = np.array(Ylabel)
i = np.zeros((Ylabel.shape[0],n_classes))
i[np.arange(Ylabel.shape[0]), Ylabel] = 1
Ylabel = i


test_x, test_y = Xlabel[:100], Ylabel[:100]
dev_x, dev_y = Xlabel[100:200], Ylabel[100:200]
train_x, train_y = Xlabel[200:], Ylabel[200:]


def generate_batch(bx, by, start, end):
    Xlabel = bx[start:end]
    by = by[start:end]
    bx = []
    bseq_len = []
    for x in Xlabel:
        if(x[0]!='F'):
            x = x[:-4]
        x = x.replace('/','_')
        x += '.npy'
        x = np.load(os.path.join(featdir,x))
        bseq_len.append(len(x))
        noOfPads = max_num_of_frames -len(x)
        x = np.append(x, np.zeros((noOfPads,2048)), axis=0)
        bx.append(x)
    return (bx, by, bseq_len)




# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=50)

no_of_batches = int(len(train_x ) / batch_size)

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, epoch+1):
        loss = 0
        acc = 0
        for j in range(no_of_batches):
            batch_x, batch_y, batch_seqlen = generate_batch(train_x, train_y, start=j*batch_size, end=(j+1)*batch_size)
            _, l, a = sess.run([optimizer,cost,accuracy], feed_dict={x: batch_x, y: batch_y, seq_len: batch_seqlen})
            loss += l
            acc += a
        if(step%display_step==0 or step==1):
            loss = float(loss/no_of_batches)
            acc = float(acc/no_of_batches)
            print("Step ",str(step),", Minibatch Loss= ", str(loss),", Training Accuracy= ", str(acc))
            saver.save(sess,'Model/mainRGB'+str(step)+'.ckpt')

    print("Optimization Finished!")



    def fpfn(fp,fn,p,act,positives):
        if(act==1):
            positives+=1
        if(p!=act):
            if(act==1):
                fn+=1
            else:
                fp+=1
        return (fp,fn,positives)

    def cal(batch_x,batch_y,batch_seqlen,flag=True):
        fp=0
        fn=0
        positives = 0
        acc = 0
        for xx,yy,sl in zip(batch_x,batch_y,batch_seqlen):
            xx = np.reshape(xx,(1,256,-1))
            yy = np.reshape(yy,(1,-1))
            sl = np.reshape(sl,(1))
            p,a,act = sess.run([prediction,accuracy,actual], feed_dict={x: xx, y: yy, seq_len: sl})
            fp,fn,positives = fpfn(fp,fn,p,act,positives)
            acc += a
            if(flag==True):
                print("Pred: ", p, "Actual: ",act)
        print("Positives: ",positives)
        return (acc,fp,fn)

    print("\nTrain Set")
    batch_x, batch_y, batch_seqlen = generate_batch(train_x, train_y, start=0, end=len(train_x))
    acc,fp,fn = cal(batch_x,batch_y,batch_seqlen,False)
    print("Trainset Accuracy:", float(acc)/len(train_x))
    print("False Positive: ",fp,"  False Negative: ",fn)

    print("\nDev Set")
    batch_x, batch_y, batch_seqlen = generate_batch(dev_x, dev_y, start=0, end=len(dev_x))
    acc,fp,fn = cal(batch_x,batch_y,batch_seqlen)    
    print("Devset Accuracy:", float(acc)/len(dev_x))
    print("False Positive: ",fp,"  False Negative: ",fn)

    print("\nTest Set")
    batch_x, batch_y, batch_seqlen = generate_batch(test_x, test_y, start=0, end=len(test_x))
    acc,fp,fn = cal(batch_x,batch_y,batch_seqlen)
    print("Testset Accuracy:", float(acc)/len(test_x))
    print("False Positive: ",fp,"  False Negative: ",fn)