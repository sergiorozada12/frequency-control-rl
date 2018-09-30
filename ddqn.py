# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts implements an ANN as function approximator to 
solve a Reinforcement Learning task by using Q-learning. Advanced techniques
such as experience replay and double dqn has been used.

""" 

import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architectures

""" Definition of the model using tensorflow"""

# Instantiate the model to initialize the variables
tf.reset_default_graph()
mainNet = architectures.dqn()
targetNet = architectures.dqn()
init = tf.global_variables_initializer()

""" Training of the model """

# Parameters of the learning operation
tau = 0.001
epsilon = 0.9
gamma = 0.9
episodes = 50000
steps = 100
cumm_r = 0
cumm_r_list = []

# Variables to tune between the double architecture
trainables = tf.trainable_variables()
targetOps = rl.updateTargetGraph(trainables,tau)

# Buffer to store the experience to replay
buffer = rl.experience_buffer()

# Launch the learning
with tf.Session() as session:
    session.run(init)
    
    # Iterate all the episodes
    for i in range(episodes):
        print("\nEPISODE: ",i)
        
        # Store cummulative reward per episode
        cumm_r_list.append(cumm_r)
        cumm_r = 0
        
        # Instances of the environment
        generator = dn.Node(powerSetPoint=3.15)
        load = dn.Node(powerSetPoint=-3.15+ (-0.25+np.random.rand()/2))
        area = dn.Area(frequencySetPoint=50,M=1,D=0)
        area.calculateDeltaF([generator,load])
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Choose the greedy action and the Q values
            current_f = area.getDeltaF()
            a,Q_values = session.run([mainNet.predict,mainNet.Qout],
                                     feed_dict={mainNet.inputs:np.array(current_f).reshape(1,1)})
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
            buffer.add(np.reshape(np.array([current_f,new_f,a,r]),[1,4]))
            print("Delta f: ",round(current_f,2)," Action: ",a, " Reward: ",r)
            
            # Update the model each 32 steps with a minibatch of 32
            if ((j % 4) == 0) & (len(buffer.buffer) > 32):
                miniBatch = buffer.sample(size = 32)
                
                # Run the network for current state and next state
                Q_values = session.run(mainNet.Qout,feed_dict={mainNet.inputs:np.reshape(miniBatch[:,0],[32,1])})
                Q_mainNet = session.run(mainNet.Qout,feed_dict={mainNet.inputs:np.reshape(miniBatch[:,1],[32,1])})
                Q_targetNet = session.run(targetNet.Qout,feed_dict={targetNet.inputs:np.reshape(miniBatch[:,1],[32,1])})
                a_max_mainNet = np.argmax(Q_mainNet,axis=1)
                
                # Prepare the target: Q from Q2 and action from Q1
                targetQ = Q_values.copy()
                for k in range(32):
                    a_target = a_max_mainNet[k]
                    a_modify = int(miniBatch[k,2])
                    r_target = miniBatch[k,3]
                    targetQ[k,a_modify] = r_target + gamma*Q_targetNet[k,a_target]
                
                # Update the model
                session.run(mainNet.upd,feed_dict={mainNet.inputs:np.reshape(miniBatch[:,0],[32,1])
                                                  ,mainNet.nextQ:targetQ})
                
                # Update the target network towards the main one
                rl.updateTarget(targetOps,session)
            
            # Update epsilon
            epsilon = rl.getNewEpsilon(epsilon)
            
            # End episode if delta f is too large
            if rl.endEpisode(area.getDeltaF()):
                break
     
    # Save the model and the data gathered
    saver = tf.train.Saver()
    saver.save(session,"models/ddqn")
    rl.saveData("cummulative_reward_ddqn.pickle",cumm_r_list)