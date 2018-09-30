# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts implements a continuous policy gradient approach. The
idea is to use a recurrent neural net approximator to estimate the parameters of
the probability distribution of the actions. This method is called Monte Carlo 
Policy Gradient or REINFORCE.
The updates are done at the end of the episode using the accumulated reward as
score function.

""" 

import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architectures

""" Definition of the model using tensorflow"""

tf.reset_default_graph()
h_size = 100
lstm = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
net = architectures.drqn_continuous_action_pg(h_size,lstm,'main')

# Instantiate the model to initialize the variables
init = tf.global_variables_initializer()

""" Training of the model """

# Parameters of the learning operation
gamma = 0.9
episodes = 100000
steps = 100
cumm_r = 0
trace = 8
epsilon = 0.9
cumm_r_list = []

# Launch the learning
with tf.Session() as session:
    session.run(init)
    
    # Iterate all the episodes
    for i in range(episodes):
        print("\nEPISODE: ",i)
        
        # Store cummulative reward per episode and all rewards of the episode
        cumm_r_list.append(cumm_r)
        cumm_r = 0
        r_episode = []
        f_episode = []
        a_episode = []
        
        # Instances of the environment
        generator = dn.Node(powerSetPoint=3.15)
        load = dn.Node(powerSetPoint=-3.15+ (-0.25+np.random.rand()/2))
        area = dn.Area(frequencySetPoint=50,M=0.1,D=0.0160)
        area.calculateDeltaF([generator,load])
        
        # Initial state for the LSTM
        state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Calculate the probability of the action and take the action
            current_f = area.getDeltaF()
            a,new_state = session.run([net.pi_sample,net.rnn_state],feed_dict={
                                                net.inputs:np.array(current_f).reshape(1,1),
                                                net.batch_size:1,net.trainLength:1,
                                                net.state_in:state})
            a = a[0,0] + epsilon*np.random.normal(0.0,1)
            state = new_state
            
            # Take the action, modify environment and get the reward
            generator = rl.setContinuousPower(a,generator)
            area.calculateDeltaF([generator,load])
            r = rl.getSimpleReward(area.getDeltaF())
            
            # Store the rewards and states
            cumm_r += r
            r_episode.append(r)
            f_episode.append(current_f)
            a_episode.append(a)
            
            # Print some data to observe the evolution of the system
            print("Delta f: ",round(current_f,2)," Action: ",a, " Reward: ",r)
            
            # Update epsilon
            epsilon = rl.getNewEpsilon(epsilon)
            
            # End episode if delta f is too large
            if rl.endEpisode(area.getDeltaF()):
                break
        
        # Get the discounted rewards
        length = len(r_episode)
        r_episode = np.array(r_episode)
        f_episode = np.array(f_episode)  
        a_episode = np.array(a_episode)
        discounted_r = rl.discountReward(r_episode,gamma)
        
        # Prepare the size of the batch 
        batchs = length//trace
        new_length = batchs*trace
        
        # Prepare the arrays and inputs
        f_episode = f_episode[0:new_length].reshape([new_length,1])
        a_episode = a_episode[0:new_length].reshape([new_length,1])
        discounted_r = discounted_r[0:new_length].reshape([new_length,1])
        state_train = (np.zeros([batchs,h_size]),np.zeros([batchs,h_size]))
        
        session.run(net.upd,feed_dict={net.inputs:f_episode, net.a:a_episode,
                               net.r:discounted_r,net.trainLength:trace,
                               net.batch_size:batchs,net.state_in:state_train})
     
    # Save the model and the data gathered
    saver = tf.train.Saver()
    saver.save(session,"models/mcpg_continuous")
    rl.saveData("cummulative_reward_mcpg_continuous.pickle",cumm_r_list)