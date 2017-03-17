import tensorflow as tf
from datetime import datetime
import mnist

print('Loading test images...')
testimages=mnist.test_images()
labels=mnist.test_labels()
testlabels=[]
for label in labels:
    a=[0,0,0,0,0,0,0,0,0,0]
    a[label]=1
    testlabels.append(a)
print('Test images loaded...')


print('Loading train images...')
trainimages=mnist.train_images()
labels=mnist.train_labels()
trainlabels=[]
for label in labels:
    a=[0,0,0,0,0,0,0,0,0,0]
    a[label]=1
    trainlabels.append(a)
print('Train images loaded...')

weights={
    'conv1': tf.Variable(tf.random_normal([3,3,1,64])),
    'conv2': tf.Variable(tf.random_normal([3,3,64,64])),
    'conv3': tf.Variable(tf.random_normal([3,3,64,64])),
    'conv4': tf.Variable(tf.random_normal([3,3,128,64])),
    'conv5': tf.Variable(tf.random_normal([3,3,128,64])),
    'conv6': tf.Variable(tf.random_normal([3,3,192,64])),
    'conv7': tf.Variable(tf.random_normal([3,3,192,64])),
    'conv8': tf.Variable(tf.random_normal([3,3,256,64])),
    'conv9': tf.Variable(tf.random_normal([3,3,256,64])),
    'conv10': tf.Variable(tf.random_normal([3,3,320,64])),
    'conv11': tf.Variable(tf.random_normal([3,3,320,64])),
    'conv12': tf.Variable(tf.random_normal([3,3,384,64])),
    'conv13': tf.Variable(tf.random_normal([3,3,384,64])),
    'conv14': tf.Variable(tf.random_normal([3,3,448,64])),
    'conv15': tf.Variable(tf.random_normal([3,3,448,64])),
    'conva': tf.Variable(tf.random_normal([3,3,192,192])),
    'convb': tf.Variable(tf.random_normal([3,3,256,256])),
    'fc': tf.Variable(tf.random_normal([512,10]))
}

biases={
    'conv1': tf.Variable(tf.random_normal([64])),
    'conv2': tf.Variable(tf.random_normal([64])),
    'conv3': tf.Variable(tf.random_normal([64])),
    'conv4': tf.Variable(tf.random_normal([64])),
    'conv5': tf.Variable(tf.random_normal([64])),
    'conv6': tf.Variable(tf.random_normal([64])),
    'conv7': tf.Variable(tf.random_normal([64])),
    'conv8': tf.Variable(tf.random_normal([64])),
    'conv9': tf.Variable(tf.random_normal([64])),
    'conv10': tf.Variable(tf.random_normal([64])),
    'conv11': tf.Variable(tf.random_normal([64])),
    'conv12': tf.Variable(tf.random_normal([64])),
    'conv13': tf.Variable(tf.random_normal([64])),
    'conv14': tf.Variable(tf.random_normal([64])),
    'conv15': tf.Variable(tf.random_normal([64])),
    'conva': tf.Variable(tf.random_normal([192])),
    'convb': tf.Variable(tf.random_normal([256])),
    'fc': tf.Variable(tf.random_normal([10]))
}

x=tf.placeholder('float',[None,28,28])
y=tf.placeholder('float',[None,10])

learning_rate=0.1

def conv(x,weight,bias,stride):
    return tf.nn.relu(tf.add(tf.nn.conv2d(x,weight,strides=[1,stride,stride,1],padding='SAME'),bias))

def concat(x,y):
    return tf.concat([x,y],3)

def net():
    rex=tf.reshape(x,[-1,28,28,1])
    conv1=conv(rex,weights['conv1'],biases['conv1'],2)
    conv2=conv(conv1,weights['conv2'],biases['conv2'],1)
    conv3=conv(conv2,weights['conv3'],biases['conv3'],1)
    conv3=concat(conv3,conv1)
    conv4=conv(conv3,weights['conv4'],biases['conv4'],1)
    conv4=concat(conv4,conv2)
    conv5=conv(conv4,weights['conv5'],biases['conv5'],1)
    conv5=concat(conv5,conv3)
    conv6=conv(conv5,weights['conv6'],biases['conv6'],1)
    conv6=concat(conv6,conv4)
    conva=conv(conv6,weights['conva'],biases['conva'],2)
    conv7=conv(conv6,weights['conv7'],biases['conv7'],1)
    conv7=concat(conv7,conv5)
    convb=conv(conv7,weights['convb'],biases['convb'],2)
    conv8=conv(conv7,weights['conv8'],biases['conv8'],2)
    conv8=concat(conv8,conva)
    conv9=conv(conv8,weights['conv9'],biases['conv9'],1)
    conv9=concat(conv9,convb)
    conv10=conv(conv9,weights['conv10'],biases['conv10'],1)
    conv10=concat(conv10,conv8)
    conv11=conv(conv10,weights['conv11'],biases['conv11'],1)
    conv11=concat(conv11,conv9)
    conv12=conv(conv11,weights['conv12'],biases['conv12'],1)
    conv12=concat(conv12,conv10)
    conv13=conv(conv12,weights['conv13'],biases['conv13'],1)
    conv13=concat(conv13,conv11)
    conv14=conv(conv13,weights['conv14'],biases['conv14'],1)
    conv14=concat(conv14,conv12)
    conv15=conv(conv14,weights['conv15'],biases['conv15'],1)
    conv15=concat(conv15,conv13)

    avg=tf.nn.avg_pool(conv15,ksize=[1,7,7,1],strides=[1,1,1,1],padding='VALID')
    avg=tf.reshape(avg,[-1,512])
    fc=tf.add(tf.matmul(avg,weights['fc']),biases['fc'])
    return fc

predict_y=net()
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
optimize=tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs=30
    batchsize=100
    print('Model 8 : WCRN-2 (15)')
    print(str(datetime.now()))
    for epoch in range(epochs):
        epochloss=0
        if (epoch+1)%25==0: learning_rate/=10
        for batch in range(int(60000/batchsize)):
            batch_x,batch_y=trainimages[batchsize*batch:batchsize*(batch+1)],trainlabels[batchsize*batch:batchsize*(batch+1)]
            _,batchloss=sess.run([optimize,loss],feed_dict={x:batch_x,y:batch_y})
            epochloss+=batchloss
        correct=0
        for batch in range(int(10000/batchsize)):
            batch_x,batch_y=testimages[batchsize*batch:batchsize*(batch+1)],testlabels[batchsize*batch:batchsize*(batch+1)]
            predict=sess.run(predict_y,feed_dict={x:batch_x})
            for label,plabel in zip(batch_y,predict):
                max,index=plabel[0],0
                for i in range(1,10):
                    if plabel[i]>max: max,index=plabel[i],i
                if label[index]==1: correct+=1
        print('Epoch',epoch+1,'completed. Loss :',epochloss,'Correct :',correct)
    print(str(datetime.now()))
