# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts implements an RNN as function approximator to 
solve a Reinforcement Learning task by using Q-learning. The environment is
partially observable, so we need to introduce some capability to deal with
non-markovian problems. Double-Q learning is introduced.

""" 

import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architectures

""" Definition of the model using tensorflow"""

# Instantiate the model to initialize the variables
tf.reset_default_graph()
h_s = 100
lstm_M = tf.contrib.rnn.BasicLSTMCell(num_units=h_s,state_is_tuple=True)
lstm_T = tf.contrib.rnn.BasicLSTMCell(num_units=h_s,state_is_tuple=True)

mainNet = architectures.drqn(h_s,lstm_M,'main')
targetNet = architectures.drqn(h_s,lstm_T,'target')
init = tf.global_variables_initializer()

""" Training of the model """

# Parameters of the learning operation
tau = 0.001
epsilon = 0.9
gamma = 0.9
episodes = 50000
steps = 100
batch = 4
n_var = 4
trace_length = 8
cumm_r = 0
cumm_r_list = []

# Variables to tune between the double architecture
trainables = tf.trainable_variables()
targetOps = rl.updateTargetGraph(trainables,tau)

# Buffer to store the experience to replay
buffer = rl.recurrent_experience_buffer()

# Launch the learning
with tf.Session() as session:
    session.run(init)
    
    # Iterate all the episodes
    for i in range(episodes):
        print("\nEPISODE: ",i)
        
        # Store cummulative reward per episode
        cumm_r_list.append(cumm_r)
        cumm_r = 0
        
        # Store the experience from the episode
        episodeBuffer = []
        
        # Instances of the environment
        generator = dn.Node(powerSetPoint=3.15)
        load = dn.Node(powerSetPoint=-3.15+ (-0.25+np.random.rand()/2))
        area = dn.Area(frequencySetPoint=50,M=0.1,D=0.0160)
        area.calculateDeltaF([generator,load])
        
        # Initial state for the LSTM
        state = (np.zeros([1,h_s]),np.zeros([1,h_s]))
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Choose the greedy action and the Q values
            current_f = area.getDeltaF()
            a,new_state = session.run([mainNet.predict,mainNet.rnn_state],
                                      feed_dict={mainNet.inputs:np.array(current_f).reshape(1,1),
                                                mainNet.state_in:state,mainNet.batch_size:1, mainNet.trainLength:1})
            a = a[0]

            # Explore if epsilon parameter agrees
            if np.random.rand() < epsilon:
                a = np.random.randint(0,2)
            
            # Take the action, modify environment and get the reward
            generator = rl.setDiscretePower(a,generator)
            area.calculateDeltaF([generator,load])
            new_f = area.getDeltaF()
            r = rl.getSimpleReward(new_f)
            cumm_r += r
            
            # Store the experience and print some data
            experience = np.array([current_f,new_f,a,r])
            episodeBuffer.append(experience)
            print("Delta f: ",round(current_f,2)," Action: ",a, " Reward: ",r)
            #print("Power generated: ",generator.getPower()," Load consumption: ",load.getPower())
            
            # Update the model each 32 steps with a minibatch of 32
            if ((j % 4) == 0) & (i > 0) & (len(buffer.buffer)>0):
                
                # Sample the miniBatch
                miniBatch = buffer.sample(batch,trace_length,n_var)
                
                #Reset the recurrent layer's hidden state and get states
                state_train = (np.zeros([batch,h_s]),np.zeros([batch,h_s]))
                s = np.reshape(miniBatch[:,0],[32,1])
                s_prime = np.reshape(miniBatch[:,1],[32,1])
                
                # Run the network for current state and next state
                Q_values = session.run(mainNet.Qout,feed_dict={mainNet.inputs: s,mainNet.trainLength:trace_length,
                                                       mainNet.batch_size:batch,mainNet.state_in:state_train})
    
                Q_mainNet = session.run(mainNet.Qout,feed_dict={mainNet.inputs: s_prime,mainNet.trainLength:trace_length,
                                                       mainNet.batch_size:batch,mainNet.state_in:state_train})
    
                Q_targetNet = session.run(targetNet.Qout,feed_dict={targetNet.inputs: s_prime,
                                                       targetNet.trainLength:trace_length,
                                                       targetNet.batch_size:batch, targetNet.state_in:state_train})
                a_max_mainNet = np.argmax(Q_mainNet,axis=1)
                
                # Prepare the target: Q from Q2 and action from Q1
                targetQ = Q_values.copy()
                for k in range(32):
                    a_target = a_max_mainNet[k]
                    a_modify = int(miniBatch[k,2])
                    r_target = miniBatch[k,3]
                    targetQ[k,a_modify] = r_target + gamma*Q_targetNet[k,a_target]
                
                # Update the model
                up,err,lss = session.run([mainNet.upd,mainNet.error,mainNet.loss],
                                         feed_dict={mainNet.inputs:s, mainNet.nextQ:targetQ, 
                                         mainNet.trainLength:trace_length, mainNet.batch_size:batch,
                                         mainNet.state_in:state_train})
    
                # Update the target network towards the main one
                rl.updateTarget(targetOps,session)
                
            # Update epsilon
            epsilon = rl.getNewEpsilon(epsilon)
            
            # Update the state
            state = new_state
            
            # End episode if delta f is too large
            if rl.endEpisode(area.getDeltaF()):
                break
            
        # Append episode to the buffer
        if len(episodeBuffer) >= 8:
            buffer.add(np.array(episodeBuffer))
     
    # Save the model and the data gathered
    saver = tf.train.Saver()
    saver.save(session,"models/ddrqn")
    rl.saveData("cummulative_reward_ddrqn.pickle",cumm_r_list)