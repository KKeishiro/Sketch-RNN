# internal imports
import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import (Input, Bidirectional, LSTM, Dense, Lambda,
        Reshape, RepeatVector, concatenate, maximum, multiply, subtract)
from keras.initializers import Zeros, RandomNormal
from keras.optimizers import Adam
from keras import backend as K



def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())


def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      data_set=['owl.npz'],  # Our dataset.
      num_epochs=1000,  # Total number of epochs of training. Keep large.
      max_seq_len=250,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      enc_rnn_size=256,  # Size of encoder.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.001,  # Learning rate.
      decay_rate=0.9999,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
  )
  return hparams


# below is where we need to do MDN (Mixture Density Network) splitting of
# distribution params
def get_mixture_coef(output, n_out):
  """Returns the tf slices containing mdn dist params."""
  # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
  z = output
  z = tf.reshape(z, [-1, n_out])
  z_pen_logits = z[:, 0:3]  # pen states
  z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

  # process output z's into MDN paramters

  # softmax all the pi's and pen states:
  z_pi = tf.nn.softmax(z_pi)
  z_pen = tf.nn.softmax(z_pen_logits)

  # exponentiate the sigmas and also make corr between -1 and 1.
  z_sigma1 = K.exp(z_sigma1)
  z_sigma2 = K.exp(z_sigma2)
  z_corr = tf.tanh(z_corr)

  r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
  return r


class SketchRNN():
  """SketchRNN model definition."""

  def __init__(self, hps):
    self.hps = hps
    self.build_model(hps)

  def build_model(self, hps):
    # VAE model = encoder + Decoder
    # build encoder model
    encoder_inputs = Input(shape=(hps.max_seq_len, 5), name='encoder_input')
    # (batch_size, max_seq_len, 5)
    encoder_lstm = LSTM(hps.enc_rnn_size,
                use_bias=True,
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                recurrent_dropout=1.0-hps.recurrent_dropout_prob,
                return_sequences=True,
                return_state=True)
    bidirectional = Bidirectional(encoder_lstm)
    (unused_outputs, # (batch_size, max_seq_len, enc_rnn_size * 2)
    last_h_fw, unused_c_fw, # (batch_size, enc_rnn_size) * 2
    last_h_bw, unused_c_bw) = bidirectional(encoder_inputs)
    last_h = concatenate([last_h_fw, last_h_bw], 1)
    # (batch_size, enc_rnn_size*2)

    normal_init = RandomNormal(stddev=0.001)
    self.z_mean = Dense(hps.z_size,
                  activation='linear',
                  use_bias=True,
                  kernel_initializer=normal_init,
                  bias_initializer='zeros')(last_h) # (batch_size, z_size)
    self.z_presig = Dense(hps.z_size,
                  activation='linear',
                  use_bias=True,
                  kernel_initializer=normal_init,
                  bias_initializer='zeros')(last_h) # (batch_size, z_size)

    def sampling(args):
      z_mean, z_presig = args
      self.sigma = K.exp(0.5 * z_presig)
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      epsilon = K.random_normal((batch, dim), 0.0, 1.0)
      batch_z = z_mean + self.sigma * epsilon

      return batch_z # (batch_size, z_size)

    self.batch_z = Lambda(sampling,
                    output_shape=(hps.z_size,))([self.z_mean, self.z_presig])

    # instantiate encoder model
    self.encoder = Model(
                    encoder_inputs,
                    [self.z_mean, self.z_presig, self.batch_z], name='encoder')
    # self.encoder.summary()

    # build decoder model
    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    self.n_out = (3 + hps.num_mixture * 6)

    decoder_inputs = Input(shape=(hps.max_seq_len, 5), name='decoder_input')
    # (batch_size, max_seq_len, 5)
    overlay_x = RepeatVector(hps.max_seq_len)(self.batch_z)
    # (batch_size, max_seq_len, z_size)
    actual_input_x = concatenate([decoder_inputs, overlay_x], 2)
    # (batch_size, max_seq_len, 5 + z_size)

    self.initial_state_layer = Dense(hps.dec_rnn_size * 2,
            activation='tanh',
            use_bias=True,
            kernel_initializer=normal_init)
    initial_state = self.initial_state_layer(self.batch_z)
    # (batch_size, dec_rnn_size * 2)
    initial_h, initial_c = tf.split(initial_state, 2, 1)
    # (batch_size, dec_rnn_size), (batch_size, dec_rnn_size)
    self.decoder_lstm = LSTM(hps.dec_rnn_size,
            use_bias=True,
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            recurrent_dropout=1.0-hps.recurrent_dropout_prob,
            return_sequences=True,
            return_state=True
            )

    output, last_h, last_c = self.decoder_lstm(
                          actual_input_x, initial_state=[initial_h, initial_c])
    # [(batch_size, max_seq_len, dec_rnn_size), ((batch_size, dec_rnn_size)*2)]
    self.output_layer = Dense(self.n_out, activation='linear', use_bias=True)
    output = self.output_layer(output)
    # (batch_size, max_seq_len, n_out)

    last_state = [last_h, last_c]
    self.final_state = last_state

    # instantiate SketchRNN model
    self.sketch_rnn_model = Model(
                  [encoder_inputs, decoder_inputs],
                  output,
                  name='sketch_rnn')
    # self.sketch_rnn_model.summary()

  def vae_loss(self, inputs, outputs):
    # KL loss
    kl_loss = 1 + self.z_presig - K.square(self.z_mean) - K.exp(self.z_presig)
    self.kl_loss = -0.5 * K.mean(K.sum(kl_loss, axis=-1))
    self.kl_loss = K.maximum(self.kl_loss, K.constant(self.hps.kl_tolerance))

    # the below are inner functions, not methods of Model
    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
      norm1 = subtract([x1, mu1])
      norm2 = subtract([x2, mu2])
      s1s2 = multiply([s1, s2])
      # eq 25
      z = (K.square(tf.divide(norm1, s1)) + K.square(tf.divide(norm2, s2)) -
           2 * tf.divide(multiply([rho, multiply([norm1, norm2])]), s1s2))
      neg_rho = 1 - K.square(rho)
      result = K.exp(tf.divide(-z, 2 * neg_rho))
      denom = 2 * np.pi * multiply([s1s2, K.sqrt(neg_rho)])
      result = tf.divide(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
      """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
      # This represents the L_R only (i.e. does not include the KL loss term).

      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                             z_corr)
      epsilon = 1e-6
      # result1 is the loss wrt pen offset (L_s in equation 9 of
      # https://arxiv.org/pdf/1704.03477.pdf)
      result1 = multiply([result0, z_pi])
      result1 = K.sum(result1, 1, keepdims=True)
      result1 = -K.log(result1 + epsilon)  # avoid log(0)

      fs = 1.0 - pen_data[:, 2]  # use training data for this
      fs = tf.reshape(fs, [-1, 1])
      # Zero out loss terms beyond N_s, the last actual stroke
      result1 = multiply([result1, fs])

      # result2: loss wrt pen state, (L_p in equation 9)
      result2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=pen_data, logits=z_pen_logits)
      result2 = tf.reshape(result2, [-1, 1])
      result2 = multiply([result2, fs])

      result = result1 + result2
      return result

    # reshape target data so that it is compatible with prediction shape
    target = tf.reshape(inputs, [-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
    pen_data = concatenate([eos_data, eoc_data, cont_data], 1)

    out = get_mixture_coef(outputs, self.n_out)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                            o_pen_logits, x1_data, x2_data, pen_data)

    self.r_loss = tf.reduce_mean(lossfunc)

    kl_weight = self.hps.kl_weight_start
    self.loss = self.r_loss + self.kl_loss * kl_weight
    return self.loss

  def model_compile(self, model):
      adam = Adam(lr=self.hps.learning_rate, clipvalue=self.hps.grad_clip)
      model.compile(loss=self.vae_loss, optimizer=adam)
