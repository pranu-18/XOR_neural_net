import numpy as np 
import time

input_layers = 10
output_layers = 10
hidden_layers = 10
samples = 300

learning_rate = 0.01
momentum = 0.8


np.random.seed(0)

def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def tan_prime(x):
	return 1-np.tanh(x)**2

def train(x,y,W1,W2,b1,b2):
	#froward propagation
	A = np.dot(x,W1) + b1
	Z = np.tanh(A)

	B = np.dot(Z,W2) + b2
	Y = sigmoid(B)

	#back propagation
	Ew2 = Y-y
	Ew1 = tan_prime(A)*np.dot(W2,Ew2)

	dw2 = np.outer(Z, Ew2)
	dw1 = np.outer(x, Ew1)

	loss = -np.mean(y*np.log(Y) + (1-y)*np.log(1-Y))

	return loss, (dw1,dw2,Ew1,Ew2)

def predict(x,W1,W2,b1,b2):
	A = np.dot(x,W1) + b1
	B = np.dot(np.tanh(A),W2) + b2
	return (sigmoid(B)>0.5).astype(int)

W1 = np.random.normal(scale=0.1, size = (input_layers,hidden_layers)) 
W2 = np.random.normal(scale=0.1, size=(hidden_layers,output_layers))

b1 = np.zeros(hidden_layers)
b2 = np.zeros(output_layers)

params = [W1,W2,b1,b2]

X = np.random.binomial(1,0.5, size=(samples,input_layers))
y = X^1

for epoch in range(1000):
	err = []
	upd = [0]*len(params)

	t0 = time.clock()
	for i in range(X.shape[0]):

		loss, grad = train(X[i], y[i], *params)

		for j in range(len(params)):
			params[j] -= upd[j]

		for j in range(len(params)):
			upd[j] = learning_rate*grad[j] + momentum*upd[j]

		err.append(loss)

	print("Epoch: %d, loss: %.8f, Time: %.4f"%(epoch, np.mean(err), time.clock()-t0))

x = np.random.binomial(1,0.5,(input_layers))
print("XOR prediction:")
print(x)
print(predict(x, *params))
