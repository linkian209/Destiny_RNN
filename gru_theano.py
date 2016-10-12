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
