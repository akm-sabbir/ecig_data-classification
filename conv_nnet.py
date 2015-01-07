import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load_data import mnist
import cPickle
from sklearn.datasets import fetch_mldata
import csv
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from sklearn.cross_validation import train_test_split
def float_val(val):
	return np.asarray(val, dtype=theano.config.floatX)
def init_w(shape):
	return theano.shared(float_val(np.random.randn(*shape) * 0.01))
def rectify(mat):
	return T.maximum(mat, 0.)
def dropout(data_mat, prob = 0.,srand):
	if prob > 0:
		retain_prob = 1 - prob
		data_mat *= srand.binomial(X.shape, prob = retain_prob, dtype=theano.config.floatX)
		data_mat /= retain_prob
	return data_mat
def RMSpropagation(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
	updates = []
	gradients = T.grad(cost = cost, wrt = params)
	for p, g in zip(params, gradients):
		mom = theano.shared(p.get_value() * 0.)
		mom_new = rho * mom + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(mom_new + epsilon)
		g = g / gradient_scaling
		updates.append((mom, mom_new))
		updates.append((p, p - lr * g))
	return updates
def softmax(mat):
	e_x = T.exp(mat - mat.max( axis = 1 ).dimshuffle(0, 'x'))
	return e_x / e_x.sum( axis = 1 ).dimshuffle(0, 'x')

def model(mat, w1, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
	srand = RandomStreams()
	l1a = rectify(conv2d(mat, w1, border_mode='full'))
	l1 = max_pool_2d(l1a, (2, 2))
	l1 = dropout(l1, p_drop_convi,srand)
	l2a = rectify(conv2d(l1, w2))
	l2 = max_pool_2d(l2a, (2, 2))
	l2 = dropout(l2, p_drop_conv,srand)
	l3a = rectify(conv2d(l2, w3))
	l3b = max_pool_2d(l3a, (2, 2))
	l3 = T.flatten(l3b, outdim=2)
	l3 = dropout(l3, p_drop_conv,srand)
	l4 = rectify(T.dot(l3, w4))
	l4 = dropout(l4, p_drop_hidden,srand)

	p_x_y = softmax(T.dot(l4, w_o))
	return l1, l2, l3, l4, p_x_y

def read_data_source(path_name = None,target = None):
	with open(path_name,'rU') as files:
		data = csv.reader(files,delimiter = ',')
		X = []
		y = []
		count = 0
		for rows in data:
			if(count == 0):
				count += 1
				continue
			if(target == None):
				X.append(np.array(rows))
			else:
				X.append(np.array(row[target + 1 : ]))
				y.appen(np.array(row[target]))
	print(str(np.shape(X)))
	print(str(len(y)))
	X = floatX(np.vstack(X))
	X = X/255.
	if(target != None):
		return (X,np.array(y,dtype = np.uint8))
	else:
		return X

def load_model():
	model_list = []
	model_reader = file('model_r/obj.save','rb')
	for i in range(2):
		model_list.append(cPickle.load(model_reader))
	model_reader.close()
	return model_list
def test_op():
	data = read_data_source('test.csv')
	models = load_model()
	data = data.reshape(-1,1,28,28)
	pred_y = models[1](data)
	#print(np.mean(pred_y == train_y))
	
	with open('prediction/conv_prediction.txt','w+') as prediction_result:
		for each in pred_y:
			prediction_result.write(str(each)+'\n')
	
	return
def main_op():
	print('start loading data')
	trX, teX, trY, teY = mnist(one_hot_T = True)
	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)
	data_X = T.ftensor4()
	data_Y = T.fmatrix()
	w1 = init_w((32, 1, 3, 3))
	w2 = init_w((64, 32, 3, 3))
	w3 = init_w((128, 64, 3, 3))
	w4 = init_w((128 * 3 * 3, 625))
	w_o = init_w((625, 10))

	noise_l1, noise_l2, noise_l3, noise_l4, noise_px_y = model(data_X, w1, w2, w3, w4, w_o, 0.2, 0.5)
	l1, l2, l3, l4, px_y = model(data_X, w1, w2, w3, w4, w_o, 0., 0.)
	y_x = T.argmax(px_y, axis = 1)
	cost = T.mean(T.nnet.categorical_crossentropy(noise_px_y, data_Y))
	params = [w1, w2, w3, w4, w_o]
	updates = RMSpropagation(cost, params, lr = 0.001)
	train = theano.function(inputs = [data_X, data_Y], outputs = cost, updates = updates, allow_input_downcast = True)
	predict = theano.function(inputs = [data_X], outputs = y_x, allow_input_downcast = True)
	print('end of symbol formation')
	count = 0
	#file_writer =  open('conv_nnet_file_output.txt','a+')
	for i in range(100):
		file_writer =  open('conv_nnet_file_output.txt','a+')
		try:
			file_writer.write('start '+ str(count) + ' iteration for training: ' )
			for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
				cost = train(trX[start:end], trY[start:end])
			file_writer.write(str(np.mean(np.argmax(teY, axis=1) == predict(teX))) +' ' + str(len(teX)) )
			count += 1
			file_writer.write('end of operation')
		finally:
			file_writer.close()
			with open('model/obj.save','wb') as model_writer:
				cPickle.dump(train,model_writer,protocol = cPickle.HIGHEST_PROTOCOL)
				cPickle.dump(predict,model_writer,protocol = cPickle.HIGHEST_PROTOCOL)
	
	return
#main_op()
test_op()
