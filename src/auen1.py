import sys
import numpy as np
import argparse

from keras.layers import Input, Dense, Dropout
from keras.models import Model

parser = argparse.ArgumentParser(description='Basic autoencoder')
parser.add_argument('--train',required=True, help= "training data")
parser.add_argument('--test',required=True, help="validation data")
parser.add_argument('--err_out',required=True, help="put reconstruction error in this file")
parser.add_argument('--dropout',default=0.1, help="dropout rate")
parser.add_argument('--topology',default="300x100",help="network topolgy")
parser.add_argument('--act_type',default="relu",help="activiation type (e.g. relu, sigmoid)")
parser.add_argument('--normalizer',default=1,help="normalize constant for feature vector")
parser.add_argument('--epoch',default=1,help="epochs")
parser.add_argument('--mbatch',default=50,help="mini batch size")

args=parser.parse_args()

## initialize key parameters and input variables
train_file=args.train
test_file=args.test
ofname=args.err_out
EPOCH = int(args.epoch) #1
BATCH = int(args.mbatch) #50
atype=args.act_type #'relu' # 'sigmoid'
F_MAX = args.normalizer #1
train =np.loadtxt(train_file)
test =np.loadtxt(test_file)
fsize=test.shape[1]
assert fsize == train.shape[1]

# input feature vector diminsion
P     = fsize  
# Dropout rate
DR    = args.dropout      

layer_str=args.topology
layers=[]
if layer_str.find("x") == -1 :
   layers = [ int(layer_str) ]
else :
   layer_vals = layer_str.split('x')
   for val in layer_vals :
      layers.append( int(val))

assert layers != []
print "layers",layers

x_train = train[:, 0:P] / F_MAX
x_test = test[:, 0:P] / F_MAX

input_vector = Input(shape=(P,))

assert len(layers) >= 1
x = Dense(layers[0], activation=atype)(input_vector)
for it in range(1,len(layers),1) :
   x = Dropout(DR)(x)
   x = Dense(layers[it], activation=atype)(x)
encoded = x

for it in range(1,len(layers),1) :
   x = Dense(P, activation=atype)(x)

x = Dense( P, activation=atype)(x)
decoded = x

ae = Model(input_vector, decoded)
ae.summary()

depth=len(layers)
depth_idx=len(layers)-1
encoded_input = Input(shape=(layers[depth_idx],))
encoder = Model(input_vector, encoded)
## generalize this to arbitrary depth
if depth == 3 :
   decoder = Model(encoded_input, ae.layers[-1](ae.layers[-2](ae.layers[-3](encoded_input))))
elif depth == 2 :
   decoder = Model(encoded_input, ae.layers[-1](ae.layers[-2](encoded_input)))
elif depth == 1 : 
   decoder = Model(encoded_input, ae.layers[-1](encoded_input))
else :
   print "Unexpected depth",depth
   sys.exit(0)


ae.compile(optimizer='rmsprop', loss='mean_squared_error')

history = ae.fit(x_train, x_train,
       batch_size=BATCH,
       nb_epoch=EPOCH,
       validation_data=[x_test, x_test])

#print(history.history.keys())
oname2=ofname+".avg"
of2=open(oname2,'w')
of=open(ofname,'w')
encoded_image = encoder.predict(x_test)
decoded_image = decoder.predict(encoded_image)
num_examples=x_test.shape[0]
print "num test examples",num_examples
re_vec = np.zeros((num_examples,))
for it in range(0,len(decoded_image),1) :
   #dist = np.linalg.norm( decoded_image[it] - x_test[it] )
   dist = np.sum( (decoded_image[it] - x_test[it])**2 )
   re_vec[it]=dist
   of.write(str(dist)+"\n")

avg=np.average(re_vec)
std=np.std(re_vec)
of2.write(str(avg)+"\t"+str(std))

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

#plt.hist(diffs, bins='auto')
#plt.title("Histogram of Errors with 'auto' bins")
#plt.savefig('histogram.png')
