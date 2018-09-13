import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
n=240
x=np.linspace(1,2*n, 2*n)
def rsig():
    y=0.01*np.sin(0.1*(x-n))
    np.random.shuffle(y)
    z=6+y+random.uniform(0,0.8)+random.uniform(-0.8,1.2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.4,0.6)*np.sin(x-random.uniform(0,2*n))+random.uniform(-0.4,0.2)*np.sin(x-random.uniform(0,2*n))
    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.4,0.6)*np.sin(x-random.uniform(0,2*n))+random.uniform(-0.4,0.2)*np.sin(x-random.uniform(0,2*n))
    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.4,0.6)*np.sin(x-random.uniform(0,2*n))+random.uniform(-0.4,0.2)*np.sin(x-random.uniform(0,2*n))
    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.51,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-0.21,1.5)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.1)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,2)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    +random.uniform(-1,1.7)*np.tanh(x-random.uniform(0,2*n))+random.uniform(-1.2,1)*np.tanh(x-random.uniform(0,2*n))    
    return z
num_sample=1000
pl1=np.array([rsig() for _ in range(num_sample)])
pl2=np.array([rsig() for _ in range(num_sample)])
pl3=np.array([5-0.4*np.array(pl1[_])+0.6*np.array(pl2[_]) for _ in range(num_sample)])
S=[pl1,pl2,pl3]
S=np.transpose(S,[1,2,0])
H=[]
K=[]
g=[]
for _ in range(num_sample):
    a=rsig()
    b=rsig()
    c=random.uniform(4,5)-a+b
    H.append(np.transpose(np.array([a[:n],b[:n],c[:n]])))
    K.append(np.transpose(np.array([a[n:],b[n:],c[n:]])))
H=np.array(H)
K=np.array(K)
g=[H,K]
np.transpose(H,[1,0,2]).shape
plt.figure(figsize=(12,6))
plt.plot(x,pl1[0],"--",label="p1")
plt.plot(x,pl2[0],"--",label="p2")
plt.plot(x,pl3[0],"--",label="p3")
plt.legend(loc="best")
plt.title("p1, p2, p3 plots")
input_dim=3
output_dim=3
num_layer=1
hidden_dim=100
lambda_regular=0.002
num_epoch=200
batch_size=10
tf.reset_default_graph()
sess = tf.InteractiveSession()
with tf.variable_scope('Seq2seq'):
    enc_inp=[tf.placeholder(tf.float32,shape=(None,input_dim)) for t in range(n)]
    expected_sparse_output=[tf.placeholder(tf.float32,shape=(None,output_dim)) for t in range(n)]
    dec_inp=[tf.zeros_like(enc_inp[0],dtype=tf.float32,name="Go")]+enc_inp[:-1]
    cells=[]
    for i in range(num_layer):
        cells.append(tf.contrib.rnn.GRUCell(hidden_dim))
    cell=tf.contrib.rnn.MultiRNNCell(cells)
    dec_outputs,dec_state_mem =tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(enc_inp,dec_inp,cell)
    w_out=tf.Variable(tf.random_normal([hidden_dim,output_dim]))
    b_out=tf.Variable(tf.random_normal([output_dim]))
    outputscalling=tf.Variable(1.0)
    reshaped_output=[outputscalling*(tf.matmul(i,w_out)+b_out) for i in dec_outputs]
    output_loss=0
    reg_loss=0
    for y,Y in zip(reshaped_output,expected_sparse_output):
        output_loss+=tf.reduce_mean(tf.nn.l2_loss(y-Y))
    for var in tf.trainable_variables():
        if not ("Bias" in var.name or "Output_" in var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(var))
    loss=output_loss+lambda_regular*reg_loss
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.007,decay=0.94,momentum=1/2)
    train_optimize=optimizer.minimize(loss)
def train_batch(i,f):
    X=np.transpose(H[i:f],[1,0,2])
    Y=np.transpose(K[i:f],[1,0,2])
    feed_dict={enc_inp[t]:X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]:Y[t] for t in range(len(expected_sparse_output))})
    _,loss_t=sess.run([train_optimize,loss],feed_dict)
    return loss_t
def test_batch(i,f):
    X=np.transpose(H[i:f],[1,0,2])
    Y=np.transpose(K[i:f],[1,0,2])
    feed_dict={enc_inp[t]:X[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]:Y[t] for t in range(len(expected_sparse_output))})
    loss_t=sess.run([loss],feed_dict)
    return loss_t[-1]
sess.run(tf.global_variables_initializer())
train_losses=[]
test_losses=[]
for _ in range(num_epoch):
    if _%int(num_epoch/100)==0:
        print(">",end="")
print(" ")
for _ in range(num_epoch):
    train_loss=train_batch(_*batch_size,_*batch_size+batch_size)
    train_losses.append(train_loss)
    if _%int(num_epoch/100)==0:
        print("<",end="")
print()
print("training complete")    
print ("sample test")
n_predict=4
H_sam=[]
K_sam=[]
g_sam=[]
for _ in range(n_predict):
    a=rsig()
    b=rsig()
    c=random.uniform(4,5)-a+b
    H_sam.append(np.transpose(np.array([a[:n],b[:n],c[:n]])))
    K_sam.append(np.transpose(np.array([a[n:],b[n:],c[n:]])))
H_sam=np.array(H_sam)
K_sam=np.array(K_sam)
g=[H_sam,K_sam]
#feed session and get predictions
X=np.transpose(H_sam,[1,0,2])
Y=np.transpose(K_sam,[1,0,2])
feed_dict={enc_inp[t]:X[t] for t in range(len(enc_inp))}
output=np.array(sess.run([reshaped_output],feed_dict))
#plot
for i in range(n_predict):
    plt.figure(figsize=(14,8))
    for j in range(input_dim):
        past=X[:,i,j]
        expected=Y[:,i,j]
        pred=output[0,:,i,j]
        plt.plot(range(n),past,label="past"+str(j))
        plt.plot(range(n,2*n),expected,label="future"+str(j))
        plt.plot(range(n,2*n),pred,label="prediction"+str(j))
    plt.legend(loc="best")
    plt.show()
print(np.array(sess.run([reshaped_output],feed_dict)).shape)

