# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts defines the architectures of the networks used
in the project.

"""

import tensorflow as tf

class dqn:
    """ DQN that inputs delta_f and outputs Q of +P/-P"""
    def __init__(self):
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([1,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.inputs,self.W1)+self.b1)
        
        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)
        
        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)
        
        self.b4 = tf.Variable(self.initializer([1,2]))
        self.W4 = tf.Variable(self.initializer([50,2]))
        self.Qout = tf.matmul(self.h3,self.W4)+self.b4
        self.predict = tf.argmax(self.Qout,1)
        
        # Below we obtain the loss by taking the sum of squares difference 
        self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 16
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class dueling_dqn:
    def __init__(self):
        """ Dueling DQN that inputs delta_f and outputs Q of +P/-P"""
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([1,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.inputs,self.W1)+self.b1)
        
        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)
        
        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)
        self.hA,self.hV = tf.split(self.h3,num_or_size_splits=2,axis=1)
        
        # Advantage 
        self.AW = tf.Variable(self.initializer([25,2]))
        self.Advantage = tf.matmul(self.hA,self.AW)
        
        # Value
        self.VW = tf.Variable(self.initializer([25,1]))
        self.Value = tf.matmul(self.hV,self.VW)
        
        # Combine both values
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        # Below we obtain the loss by taking the sum of squares difference 
        self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 16
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class drqn():
    """ DRQN that inputs delta_f and outputs Q of +P/-P"""
    def __init__(self,h_size,cell,sc):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,2]))
        self.W4 = tf.Variable(self.initializer([50,2]))
        self.Qout = tf.matmul(self.h3,self.W4)+self.b4
        self.predict = tf.argmax(self.Qout,1)

        # Mask to propagate the gradients back through the net
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])       
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[32,1])

        # Below we obtain the loss by taking the sum of squares difference 
        self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.error = tf.square(self.nextQ-self.Qout)
        self.error = tf.multiply(self.error,self.mask)
        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 20
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class dueling_drqn():
    """ Dueling DRQN that inputs delta_f and outputs Q of +P/-P"""
    def __init__(self,h_size,cell,sc):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)
        self.hA,self.hV = tf.split(self.h3,num_or_size_splits=2,axis=1)
        
        # Advantage 
        self.AW = tf.Variable(self.initializer([25,2]))
        self.Advantage = tf.matmul(self.hA,self.AW)
        
        # Value
        self.VW = tf.Variable(self.initializer([25,1]))
        self.Value = tf.matmul(self.hV,self.VW)
        
        # Combine both values
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.predict = tf.argmax(self.Qout,1)

        # Mask to propagate the gradients back through the net
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])       
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[32,1])

        # Below we obtain the loss by taking the sum of squares difference 
        self.nextQ = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.error = tf.square(self.nextQ-self.Qout)
        self.error = tf.multiply(self.error,self.mask)
        self.loss = tf.reduce_mean(self.error)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 20
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class drqn_discrete_action_pg():
    """ DRQN that inputs delta_f and outputs probability of +/- P"""
    def __init__(self,h_size,cell,sc):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.r = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.logits = tf.matmul(self.h3,self.W4)+self.b4
        self.p = tf.nn.sigmoid(self.logits)
        
        # Sampling from the network
        self.pi = tf.contrib.distributions.Bernoulli(self.p)
        self.pi_sample = self.pi.sample()
        self.log_pi = self.pi.log_prob(self.a)

        # Mask to propagate the gradients back through the net
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])       
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[-1,1])

        # Below we obtain the loss by taking the sum of squares difference 
        self.error = tf.multiply(self.log_pi,self.r)
        self.error = tf.multiply(self.error,self.mask)
        self.loss = -tf.reduce_sum(self.error)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 10
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class drqn_continuous_action_pg():
    """ DRQN that inputs delta_f and outputs increment of +/- P"""
    def __init__(self,h_size,cell,sc):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.r = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.mu = tf.matmul(self.h3,self.W4)+self.b4
        
        # Sampling from the network
        self.pi = tf.contrib.distributions.Normal(self.mu,1.0)
        self.pi_sample = self.pi.sample()
        self.log_pi = self.pi.log_prob(self.a)

        # Mask to propagate the gradients back through the net
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])       
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[-1,1])

        # Below we obtain the loss by taking the sum of squares difference 
        self.error = tf.multiply(self.log_pi,self.r)
        self.error = tf.multiply(self.error,self.mask)
        self.loss = -tf.reduce_sum(self.error)
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Params
        self.n_params = 10
        self.network_params = tf.trainable_variables()[self.n_params:]
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class actor_ddpg():
    """ Actor network that estimates the policy of the ddpg algorithm"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.a_unscaled = tf.nn.tanh(tf.matmul(self.h3,self.W4)+self.b4)
        self.a = tf.multiply(self.a_unscaled,0.1)
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, 1])
        
        # Take the gradients and combine
        self.unnormalized_actor_gradients = tf.gradients(
                    self.a, self.network_params, -self.critic_gradient)
        
        # Normalize dividing by the size of the batch (gradients sum all over the batch)
        self.actor_gradients = list(map(lambda x: tf.div(x,32),
                                        self.unnormalized_actor_gradients))
        
        # Optimization of the actor
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class critic_ddpg():
    """ Critic network that estimates the policy of the ddpg algorithm"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.s = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.inputs = tf.concat([self.s,self.a],axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,2])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.Q = tf.matmul(self.h3,self.W4)+self.b4
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # Obtained from the target network (double architecture)
        self.target_Q = tf.placeholder(tf.float32, [None, 1])
        
        # Loss function and optimization of the critic
        self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Get the gradient for the actor
        self.critic_gradients = tf.gradients(self.Q,self.a)
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class actor_maddpg():
    """ Actor network that estimates the policy of the maddpg algorithm"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.inputs = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,1])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.a_unscaled = tf.nn.tanh(tf.matmul(self.h3,self.W4)+self.b4)
        self.a = tf.multiply(self.a_unscaled,0.1)
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, 1])
        
        # Take the gradients and combine
        self.unnormalized_actor_gradients = tf.gradients(
                    self.a, self.network_params, -self.critic_gradient)
        
        # Normalize dividing by the size of the batch (gradients sum all over the batch)
        self.actor_gradients = list(map(lambda x: tf.div(x,32),
                                        self.unnormalized_actor_gradients))
        
        # Optimization of the actor
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class critic_maddpg():
    """ Critic network that estimates the policy of the maddpg algorithm"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.s = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a_o = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.inputs = tf.concat([self.s,self.a,self.a_o],axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,3])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.Q = tf.matmul(self.h3,self.W4)+self.b4
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # Obtained from the target network (double architecture)
        self.target_Q = tf.placeholder(tf.float32, [None, 1])
        
        # Loss function and optimization of the critic
        self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Get the gradient for the actor
        self.critic_gradients = tf.gradients(self.Q,self.a)
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class actor_maddpg_cost():
    """ Actor network that estimates the policy of the maddpg algorithm when
    cost is included"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.frequency = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.power = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.inputs = tf.concat([self.frequency,self.power],axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,2])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.a_unscaled = tf.nn.tanh(tf.matmul(self.h3,self.W4)+self.b4)
        self.a = tf.multiply(self.a_unscaled,0.1)
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # This gradient will be provided by the critic network
        self.critic_gradient = tf.placeholder(tf.float32, [None, 1])
        
        # Take the gradients and combine
        self.unnormalized_actor_gradients = tf.gradients(
                    self.a, self.network_params, -self.critic_gradient)
        
        # Normalize dividing by the size of the batch (gradients sum all over the batch)
        self.actor_gradients = list(map(lambda x: tf.div(x,32),
                                        self.unnormalized_actor_gradients))
        
        # Optimization of the actor
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.apply_gradients(zip(self.actor_gradients, self.network_params))
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]
        
class critic_maddpg_cost():
    """ Critic network that estimates the policy of the maddpg algorithm when
    cost is included"""
    def __init__(self,h_size,cell,sc,num_variables):
        
        # Define the model (input-hidden layers-output)
        self.frequency = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.power = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.power_o = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.a_o = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.inputs = tf.concat([self.frequency,self.power,self.power_o,self.a,self.a_o],axis=1)
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # LSTM to encode temporal information
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])   # batch size
        self.trainLength = tf.placeholder(dtype=tf.int32)           # trace lentgth
        self.rnnInp = tf.reshape(self.inputs,[self.batch_size,self.trainLength,5])
        
        self.state_in = cell.zero_state(self.batch_size,tf.float32)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnnInp,cell=cell,
                                  dtype=tf.float32,initial_state=self.state_in,scope=sc+'_rnn')
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # MLP on top of LSTM
        self.b1 = tf.Variable(self.initializer([1,1000]))
        self.W1 = tf.Variable(self.initializer([h_size,1000]))
        self.h1 = tf.nn.relu(tf.matmul(self.rnn,self.W1)+self.b1)

        self.b2 = tf.Variable(self.initializer([1,100]))
        self.W2 = tf.Variable(self.initializer([1000,100]))
        self.h2 = tf.nn.relu(tf.matmul(self.h1,self.W2)+self.b2)

        self.b3 = tf.Variable(self.initializer([1,50]))
        self.W3 = tf.Variable(self.initializer([100,50]))
        self.h3 = tf.nn.relu(tf.matmul(self.h2,self.W3)+self.b3)

        self.b4 = tf.Variable(self.initializer([1,1]))
        self.W4 = tf.Variable(self.initializer([50,1]))
        self.Q = tf.matmul(self.h3,self.W4)+self.b4
        
        # Take params of the main actor network
        self.network_params = tf.trainable_variables()[num_variables:]
        
        # Obtained from the target network (double architecture)
        self.target_Q = tf.placeholder(tf.float32, [None, 1])
        
        # Loss function and optimization of the critic
        self.loss = tf.reduce_mean(tf.square(self.target_Q-self.Q))
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.upd = self.optimizer.minimize(self.loss)
        
        # Get the gradient for the actor
        self.critic_gradients = tf.gradients(self.Q,self.a)
        
    def createOpHolder(self,params,tau):
        """ Use target network op holder if needed"""
        self.update_network_params = [self.network_params[i].assign(tf.multiply(params[i], tau) +
                                     tf.multiply(self.network_params[i], 1. - tau))
                                     for i in range(len(self.network_params))]