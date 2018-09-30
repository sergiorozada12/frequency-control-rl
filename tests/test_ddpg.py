# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: Test the performance of different models and algorithms

""" 
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import tensorflow as tf
import matplotlib.pyplot as plt
import architectures
import numpy as np
import dynamics as dn
import rl

# Model and reward to test algorithms
model_path = "../models/ddpg"
reward_path = "cummulative_reward_ddpg.pickle"

reward = rl.readData(reward_path)

# Instances of the environment
generator = dn.Node(powerSetPoint=3.15)
load = dn.Node(powerSetPoint=-3.30)
area = dn.Area(frequencySetPoint=50,M=0.1,D=0.0160)
area.calculateDeltaF([generator,load])

# Define list of powers and frequencies
power = []
frequencies = []

# Let's tensorflow this
tf.reset_default_graph()
graph = tf.train.import_meta_graph(model_path+".meta")

steps = 100
h_size = 100

with tf.Session() as session:      
    # Restore values of the graph
    graph.restore(session, model_path)
    
    # Create the actor
    lstm_actor = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    actor = architectures.actor_ddpg(h_size,lstm_actor,'actor_test',len(tf.trainable_variables()))
    n_params = len(actor.network_params)
    actor.createOpHolder(tf.trainable_variables(),1)
    
    # Initialize variables and copy params
    init = tf.variables_initializer(actor.network_params)
    session.run(init)
    session.run(actor.update_network_params)

    # State of the network
    state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
    
    for i in range(steps):
        
        # Store values 
        power.append(generator.getPower())
        frequencies.append(area.getFrequency())
        
        # Get state and take the best action
        current_f = area.getDeltaF()
        
        a, new_state = session.run([actor.a,actor.rnn_state], 
                            feed_dict={actor.inputs: np.array(current_f).reshape(1,1),
                            actor.state_in: state, actor.batch_size:1,actor.trainLength:1})
        a = a[0,0]
        
        # Take the action, modify environment and get the reward
        generator = rl.setContinuousPower(a,generator)
        area.calculateDeltaF([generator,load])
        
        # Set state again
        state = new_state
        
plt.figure(1)
plt.scatter(np.arange(len(reward)),reward)
plt.xlabel('Episodes')
plt.ylabel('Cumm. reward per episode')

plt.figure(2)
plt.plot(power)
plt.plot([3.3]*100)
plt.xlabel('Steps')
plt.ylabel('Power (MW)')
plt.legend(['Agent power','Power setpoint'])

plt.figure(3)
plt.plot(frequencies)
plt.plot([50]*100)
plt.xlabel('Steps')
plt.ylabel('Frequency (Hz)')
plt.legend(['System frequency','Frequency setpoint'])