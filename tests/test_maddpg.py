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
model_path = "../models/maddpg"
reward_path = "cummulative_reward_maddpg.pickle"

reward = rl.readData(reward_path)

# Instances of the environment
generator_1 = dn.Node(powerSetPoint=1.5)
generator_2 = dn.Node(powerSetPoint=1.5)
load = dn.Node(powerSetPoint=-3.15)
area = dn.Area(frequencySetPoint=50,M=0.1,D=0.0160)
area.calculateDeltaF([generator_1,generator_2,load])

# Define list of powers and frequencies
power_1 = []
power_2 = []
frequencies = []

# Let's tensorflow this
tf.reset_default_graph()
graph = tf.train.import_meta_graph(model_path+".meta")

steps = 100
h_size = 100

with tf.Session() as session:      
    # Restore values of the graph
    graph.restore(session, model_path)
    
    # Create the actors
    lstm_actor_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    actor_1 = architectures.actor_maddpg(h_size,lstm_actor_1,'actor_1_test',len(tf.trainable_variables()))
    actor_1.createOpHolder(tf.trainable_variables(),1)
    
    n_params = len(actor_1.network_params)
    
    lstm_actor_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
    actor_2 = architectures.actor_maddpg(h_size,lstm_actor_2,'actor_2_test',len(tf.trainable_variables()))
    actor_2.createOpHolder(tf.trainable_variables()[n_params*4:],1)
    
    # Initialize variables and copy params
    init_1 = tf.variables_initializer(actor_1.network_params)
    init_2 = tf.variables_initializer(actor_2.network_params)
    
    session.run(init_1)
    session.run(init_2)
    
    session.run(actor_1.update_network_params)
    session.run(actor_2.update_network_params)

    # State of the network
    state_1 = (np.zeros([1,h_size]),np.zeros([1,h_size]))
    state_2 = (np.zeros([1,h_size]),np.zeros([1,h_size]))
    
    for i in range(steps):
        
        # Store values 
        power_1.append(generator_1.getPower())
        power_2.append(generator_2.getPower())
        frequencies.append(area.getFrequency())
        
        # Get state and take the best action
        current_f = area.getDeltaF()
        
        a_1, new_state_1 = session.run([actor_1.a,actor_1.rnn_state], 
                            feed_dict={actor_1.inputs: np.array(current_f).reshape(1,1),
                            actor_1.state_in: state_1, actor_1.batch_size:1,actor_1.trainLength:1})
        a_2, new_state_2 = session.run([actor_2.a,actor_2.rnn_state], 
                            feed_dict={actor_2.inputs: np.array(current_f).reshape(1,1),
                            actor_2.state_in: state_2, actor_2.batch_size:1,actor_2.trainLength:1})
        a_1 = a_1[0,0]
        a_2 = a_2[0,0]
        
        # Take the action, modify environment and get the reward
        generator_1 = rl.setContinuousPower(a_1,generator_1)
        generator_2 = rl.setContinuousPower(a_2,generator_2)
        area.calculateDeltaF([generator_1,generator_2,load])
        
        # Set state again
        state_1 = new_state_1
        state_2 = new_state_2
        
plt.figure(1)
plt.scatter(np.arange(len(reward)),reward)
plt.xlabel('Episodes')
plt.ylabel('Cumm. reward per episode')

plt.figure(2)
plt.plot(power_1)
plt.plot(power_2)
plt.plot(np.sum([np.array(power_1),np.array(power_2)],axis=0))
plt.plot([3.15]*100)
plt.xlabel('Steps')
plt.ylabel('Power (MW)')
plt.legend(['Gen 1 power','Gen 2 power','Total power','Power setpoint'])

plt.figure(3)
plt.plot(frequencies)
plt.plot([50]*100)
plt.xlabel('Steps')
plt.ylabel('Frequency (Hz)')
plt.legend(['System frequency','Frequency setpoint'])