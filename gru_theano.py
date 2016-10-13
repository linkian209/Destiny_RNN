import time
import operator
import theano
import theano.tensor as T
import numpy as np
for theano.gradient import grad_clip

class GRUTheano:

  # Initialization function
  def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):

    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate

    # Initialize network parameters
    E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
    U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
    W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
    V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    b = np.zeros((6, hidden_dim))
    x = np.zeros(word_dim)
    # Create Theano shared variables
    self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
    self.U = theano.shared(name='U', value=W.astype(theano.config.floatX))
    self.W = theano.shared(name='W', value=U.astype(theano.config.floatX))
    self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
    self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
    self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
    # SGD parameters
    self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
    self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
    self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
    self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
    self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
    self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

    # Theano
    self.theano = {}
    self.__theano_build__()

  def __theano_build__(self):
    E, U, W, V, b, c = self.E, self.U, self.W, self.V, self.b, self.c

    # Theano Vectors
    x = T.ivector('x')
    y = T.ivector('y')

    def forwardPropStep(x_t, s_t1_prev, s_t2_prev):
      # Word embedding layer
      x_e = E[:,x_t]

      # GRU Layer 1
      # Optimize by doing major multiplactions now
      U_c = U.transpose().dot(diag(x_e, x_e, x_e, 1, 1, 1)).transpose()
      W_c = W.transpose().dot(diag(s_t1_prev, s_t1_prev, 1, s_t2_prev, s_t2_prev, 1)).transpose()

      z_t1 = T.nnet.hard_sigmoid(U_c[0] + W_c[0] + b[0])
      r_t1 = T.nnet.hard_sigmoid(U_c[1] + W_c[1] + b[1])
      c_t1 = T.tanh(U_c[2] + W[2].dot(s_t1_prev * r_t1) + b[2])
      s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

      # GRU Layer 2
      # Do some more large matrix multiplaction
      U_c = U.transpose().dot(diag(1, 1, 1, s_t1, s_t1, s_t1)).transpose()

      z_t2 = T.nnet.hard_sigmoid(U_c[3] + W_c[3] + b[3])
      r_t2 = T.nnet.hard_sigmoid(U_c[4] + W_c[4] + b[4])
      c_t2 = T.tanh(U_c[5] + W[5].dot(s_t2_prev * r_t2) + b[5])
      s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

      # Final calculation
      o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

      return [o_t, s_t1, s_t2]

    # Theano Looping, using scan
    [o, s, s2], updates = theano.scan(
      forward_prop_step,
      sequences=x,
      truncate_gradient=self.bptt_truncate,
      outputs_info=[None,
                    dict(initial=T.zeros(self.hidden_dim)),
                    dict(initial=T.zeros(self.hidden_dim))])

    # Prediction and error params
    prediction = T.argmax(o, axis=1)
    o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

    # Total Cost (can add regularization here)
    cost = o_error

    # Gradients
    dE = T.grad(cost, E)
    dU = T.grad(cost, U)
    dW = T.grad(cost, W)
    dV = T.grad(cost, V)
    db = T.grad(cost, b)
    dc = T.grad(cost, c)

    # Theano functions
    self.predict = theano.function([x], o)
    self.predictClass = theano.function([x], prediction)
    self.ceError = theano.function([x, y], cost)
    self.bptt = theano.function([x, y], [dE, dU, dW, dV, db, dc])

    # SGD Parameters
    learning_rate = T.scalar('learning_rate')
    decay = T.scalar('decay')

    # rmsprop cache updates
    mE = decay * self.mE + (1 - decay) * dE ** 2
    mU = decay * self.mU + (1 - decay) * dU ** 2
    mW = decay * self.mW + (1 - decay) * dW ** 2
    mV = decay * self.mV + (1 - decay) * dV ** 2
    mb = decay * self.mb + (1 - decay) * db ** 2
    mc = decay * self.mc + (1 - decay) * dc ** 2

    # SGD Step function
    # This function is the function that trains the model
    self.sgdStep = theano.function(
      [x, y, learning_rate, theano.Param(decay, default=0.9)],
      [],
      updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
               (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
               (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
               (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
               (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
               (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
               (self.mE, mE),
               (self.mU, mU),
               (self.mW, mW),
               (self.mV, mV),
               (self.mb, mb),
               (self.mc, mc)])

  # Calculate Loss Functions
  def calculateTotalLoss(self, X, Y):
    return np.sum([self.ceError(x,y) for x,y in zip(X,Y)])

  def calculateLoss(self, X, Y):
    # Get average loss per word
    num_words = np.sum([len(y) for y in Y])
    return self.calculateTotalLoss(X,Y) / float(num_words)

