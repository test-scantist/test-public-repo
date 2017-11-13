import numpy as np
import theano
import theano.tensor as T

from load import mnist
from misc.utils import save_images, save_plot
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

corruption_level = 0.1
training_epochs = 100
learning_rate = 0.1
batch_size = 128
moment = 0.1
beta = 0.5
rho = 0.05

config = 2

if config == 2:
    momentum = True
else:
    momentum = False


def init_weights(n_visible, n_hidden):
    lim = 4 * np.sqrt(6. / (n_hidden + n_visible))
    initial_w = np.asarray(np.random.uniform(low=-lim, high=lim,
                           size=(n_visible, n_hidden)), dtype=theano.config.floatX)
    return theano.shared(value=initial_w, name='W', borrow=True)


def init_bias(n):
    return theano.shared(value=np.zeros(n, dtype=theano.config.floatX),
                         borrow=True)


class sgd(object):
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
	self.moment = 0.1

    def updates(self, cost, params, learning_rate, momentum=False):
        updates = []
	grads = T.grad(cost, params)
        for n, (param, grad) in enumerate(zip(params, grads)):
	    if momentum:
            	memory = self.memory_[n]
            	update = self.moment * memory - learning_rate * grad
            	updates.append((memory, update))
            	updates.append((param, param + update))
	    else:
            	updates.append((param, param - learning_rate * grad))
        return updates

def cross_entropy(x, h):
    c = T.mean(T.sum(x * T.log(h) + (1 - x) * T.log(1 - h), axis=1))
    return c


def cost_sparse(h):
    c = beta*T.shape(h)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
                - beta*rho*T.sum(T.log(T.mean(h, axis=0)+1e-6)) \
                - beta*(1-rho)*T.sum(T.log(1-T.mean(h, axis=0)+1e-6))
    return c


# To hold data and labels
x = T.fmatrix('x')
y_ = T.fmatrix('y_')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x


# S-DAE Layer 1
W1 = init_weights(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose()

# S-DAE Layer 2
W2 = init_weights(900, 625)
b2 = init_bias(625)
b2_prime = init_bias(900)
W2_prime = W2.transpose()

# S-DAE Layer 3
W3 = init_weights(625, 400)
b3 = init_bias(400)
b3_prime = init_bias(625)
W3_prime = W3.transpose()

# Classification layer
W4 = init_weights(400, 10)
b4 = init_bias(10)

# Network definition
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)

y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)

y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)

y_classify = T.nnet.softmax(T.dot(y3, W4) + b4)
out_classify = T.argmax(y_classify, axis=1)

if config == 2:
    cost1 = -cross_entropy(x, z1) + cost_sparse(y1)
    cost2 = -cross_entropy(y1, z2) + cost_sparse(y2)
    cost3 = -cross_entropy(y2, z3) + cost_sparse(y3)
else:
    cost1 = -cross_entropy(x, z1)
    cost2 = -cross_entropy(y1, z2)
    cost3 = -cross_entropy(y2, z3)

costffn = T.mean(T.nnet.categorical_crossentropy(y_classify, y_))

params1 = [W1, b1, b1_prime]
opt1 = sgd(params1)
updates1 = opt1.updates(cost1, params1, learning_rate, momentum=momentum)
params2 = [W2, b2, b2_prime]
opt2 = sgd(params2)
updates2 = opt2.updates(cost2, params2, learning_rate, momentum=momentum)
params3 = [W3, b3, b3_prime]
opt3 = sgd(params3)
updates3 = opt3.updates(cost3, params3, learning_rate, momentum=momentum)
paramsffn = [W1, b1, W2, b2, W3, b3, W4, b4]
optffn = sgd(paramsffn)
updatesffn = optffn.updates(costffn, paramsffn, learning_rate, momentum=momentum)

corrupted = theano.function(inputs=[x], outputs=tilde_x, allow_input_downcast=True)
train_da1 = theano.function(inputs=[x], outputs=cost1, updates=updates1, allow_input_downcast=True)
train_da2 = theano.function(inputs=[y1], outputs=cost2, updates=updates2, allow_input_downcast=True)
train_da3 = theano.function(inputs=[y2], outputs=cost3, updates=updates3, allow_input_downcast=True)
train_ffn = theano.function(inputs=[x, y_], outputs=costffn, updates=updatesffn,
                            allow_input_downcast=True)
test_ffn = theano.function(inputs=[x], outputs=out_classify, allow_input_downcast=True)
enc1 = theano.function(inputs=[x], outputs=y1, allow_input_downcast=True)
enc2 = theano.function(inputs=[y1], outputs=y2, allow_input_downcast=True)
enc3 = theano.function(inputs=[y2], outputs=y3, allow_input_downcast=True)
dec3 = theano.function(inputs=[y3], outputs=z3, allow_input_downcast=True)
dec2 = theano.function(inputs=[y2], outputs=z2, allow_input_downcast=True)
dec1 = theano.function(inputs=[y1], outputs=z1, allow_input_downcast=True)


def main():
    trX, teX, trY, teY = mnist()
    func_map = {
        0: [train_da1, enc1, W1, 28, 30],
        1: [train_da2, enc2, W2, 30, 25],
        2: [train_da3, enc3, W3, 25, 20]
    }

    data = trX
    d_t = teX[:100]
    for hidden in range(3):
        print('Training S-DAE Layer: %d' % (hidden+1))
        d = []
        for epoch in range(training_epochs):
            c = []
            for i in range(0, len(data), batch_size):
                c.append(func_map[hidden][0](data[i:i+batch_size]))
            d.append(np.mean(c, dtype='float64'))
            print('epoch %d:\tloss = %.2f' % (epoch, d[-1]))
        d_t = func_map[hidden][1](d_t)
        w = func_map[hidden][2].get_value()
        data = func_map[hidden][1](data)
        save_plot(d, "ae_%d_layer_%d" % (config, hidden+1))
        save_images(np.reshape(np.transpose(w)[:100], [-1] + [func_map[hidden][3]]*2), [10, 10],
                    'weights_%d_layer_%d.png' % (config, hidden+1))
        save_images(np.reshape(d_t, [-1] + [func_map[hidden][4]]*2), [10, 10],
                    'activations_%d_layer_%d.png' % (config, hidden+1))

    de3 = dec3(d_t)
    de2 = dec2(de3)
    de = dec1(de2)
    cr = corrupted(teX[:100])

    save_images(np.reshape(de[:100], [100, 28, 28]), [10, 10], 'clean_%d.png' % (config), spacing=0)
    save_images(np.reshape(cr, [100, 28, 28]), [10, 10], 'corrupted_%d.png' % (config), spacing=0)
    print ('Training feed-forward network')
    d, a = [], []
    best_acc = 0
    for epoch in range(training_epochs):
        c = []
        for i in range(0, len(data), batch_size):
            c.append(train_ffn(trX[i:i+batch_size], trY[i:i+batch_size]))
        d.append(np.mean(c, dtype='float64'))
        a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
	if a[-1] > best_acc:
		best_acc = a[-1]
        print('epoch %d:\tloss = %.2f\tacc = %.2f' % (epoch, d[-1], a[-1]))
    save_plot(d, "ae_classify_%d_train_error" % (config),
              ylabel="Categorical Cross Entropy Loss")
    save_plot(a, "ae_classify_%d_test_acc" % (config), label="test_accuracy",
              ylabel="Accuracy")
    print('Best accuracy = %.2f' % (best_acc*100))

if __name__ == "__main__":
    main()
