# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts implements multiagent deep deterministic policy gradient algorithm.
This is an actor-critic architecture that aims to produce a continuous deterministic
action for the system to operate in the multiagent case. The critic helps to estimate how good actions
are. The actor is the part of the architecture that produces the action. Training is centralized
by using shared information with the critics. Operation is decentralized just using the actors.

""" 

import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architectures

""" Definition of the model using tensorflow"""

tf.reset_default_graph()
h_size = 100

# Create actor, critic, main and target networks of the first agent
lstm_actor_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor_1 = architectures.actor_maddpg(h_size,lstm_actor_1,'actor_1',0)
critic_1 = architectures.critic_maddpg(h_size,lstm_critic_1,'critic_1',len(tf.trainable_variables()))

lstm_actor_t_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic_t_1 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor_t_1 = architectures.actor_maddpg(h_size,lstm_actor_t_1,'actor_t_1',len(tf.trainable_variables()))
critic_t_1 = architectures.critic_maddpg(h_size,lstm_critic_t_1,'critic_t_1',len(tf.trainable_variables()))

# Create actor, critic, main and target networks of the second agent
lstm_actor_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor_2 = architectures.actor_maddpg(h_size,lstm_actor_2,'actor_2',len(tf.trainable_variables()))
critic_2 = architectures.critic_maddpg(h_size,lstm_critic_2,'critic_2',len(tf.trainable_variables()))

lstm_actor_t_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic_t_2 = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor_t_2 = architectures.actor_maddpg(h_size,lstm_actor_t_2,'actor_t_2',len(tf.trainable_variables()))
critic_t_2 = architectures.critic_maddpg(h_size,lstm_critic_t_2,'critic_t_2',len(tf.trainable_variables()))

# Instantiate the model to initialize the variables
init = tf.global_variables_initializer()

""" Training of the model """

# Parameters of the learning operation
gamma = 0.9
tau = 0.001
epsilon = 0.99
episodes = 50000
steps = 100
cumm_r = 0
trace = 8
batch = 4
n_var = 5
cumm_r_list = []

# Create op holder for the target networks
actor_t_1.createOpHolder(actor_1.network_params,tau)
critic_t_1.createOpHolder(critic_1.network_params,tau)

actor_t_2.createOpHolder(actor_2.network_params,tau)
critic_t_2.createOpHolder(critic_2.network_params,tau)

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
        generator_1 = dn.Node(powerSetPoint=1.5)
        generator_2 = dn.Node(powerSetPoint=1.5)
        load = dn.Node(powerSetPoint=-3.0+ (-0.25+np.random.rand()/2))
        area = dn.Area(frequencySetPoint=50,M=0.1,D=0.0160)
        area.calculateDeltaF([generator_1,generator_2,load])
        
        # Initial state for the LSTM
        state_1 = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        state_2 = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Get the action from the actor and the internal state of the rnn
            current_f = area.getDeltaF()
            
            # First agent
            a_1, new_state_1 = session.run([actor_1.a,actor_1.rnn_state], 
                            feed_dict={actor_1.inputs: np.array(current_f).reshape(1,1),
                            actor_1.state_in: state_1, actor_1.batch_size:1,actor_1.trainLength:1})
            a_1 = a_1[0,0] + epsilon*np.random.normal(0.0,0.2)
            
            # Second agent
            a_2, new_state_2 = session.run([actor_2.a,actor_2.rnn_state], 
                            feed_dict={actor_2.inputs: np.array(current_f).reshape(1,1),
                            actor_2.state_in: state_2, actor_2.batch_size:1,actor_2.trainLength:1})
            a_2 = a_2[0,0] + epsilon*np.random.normal(0.0,0.2)
            
            # Take the action, modify environment and get the reward
            generator_1 = rl.setContinuousPower(a_1,generator_1)
            generator_2 = rl.setContinuousPower(a_2,generator_2)
            area.calculateDeltaF([generator_1,generator_2,load])
            new_f = area.getDeltaF()
            r = rl.getSimpleReward(new_f)
            cumm_r += r

            # Store the experience and print some data
            experience = np.array([current_f,new_f,a_1,a_2,r])
            episodeBuffer.append(experience)
            print("Delta f: ",round(current_f,2)," A1: ",a_1," A2: ",a_2, " Reward: ",r)
            
            # Update the model each 4 steps with a minibatch of 32
            if ((j % 4) == 0) & (i > 0) & (len(buffer.buffer)>0):
                
                # Sample the miniBatch
                miniBatch = buffer.sample(batch,trace,n_var)
                
                # Reset the recurrent layer's hidden state and get states
                state_train = (np.zeros([batch,h_size]),np.zeros([batch,h_size]))
                s = np.reshape(miniBatch[:,0],[32,1])
                s_prime = np.reshape(miniBatch[:,1],[32,1])
                actions_1 = np.reshape(miniBatch[:,2],[32,1])
                actions_2 = np.reshape(miniBatch[:,3],[32,1])
                rewards = np.reshape(miniBatch[:,4],[32,1])

                # Predict the actions of both actors
                a_target_1 = session.run(actor_t_1.a, feed_dict={actor_t_1.inputs: s_prime,
                            actor_t_1.state_in: state_train, actor_t_1.batch_size:batch,
                            actor_t_1.trainLength:trace})
                a_target_2 = session.run(actor_t_2.a, feed_dict={actor_t_2.inputs: s_prime,
                            actor_t_2.state_in: state_train, actor_t_2.batch_size:batch,
                            actor_t_2.trainLength:trace})
                
                # Predict Q of the critics
                Q_target_1 = session.run(critic_t_1.Q,feed_dict={critic_t_1.s: s_prime, 
                                                    critic_t_1.a: a_target_1,critic_t_1.a_o: a_target_2,
                                                    critic_t_1.trainLength:trace,critic_t_1.batch_size:batch,
                                                    critic_t_1.state_in:state_train})   
                Q_target_2 = session.run(critic_t_2.Q,feed_dict={critic_t_2.s: s_prime, 
                                                    critic_t_2.a: a_target_2,critic_t_2.a_o: a_target_1,
                                                    critic_t_2.trainLength:trace,critic_t_2.batch_size:batch,
                                                    critic_t_2.state_in:state_train})
                Q_target_1 = rewards + gamma*Q_target_1
                Q_target_2 = rewards + gamma*Q_target_2

                # Update the critic networks with the new Q's
                session.run(critic_1.upd,feed_dict={critic_1.s: s, critic_1.a: actions_1,
                                       critic_1.a_o: actions_2, critic_1.target_Q: Q_target_1,
                                       critic_1.trainLength:trace, critic_1.batch_size:batch,
                                       critic_1.state_in:state_train})  
                session.run(critic_2.upd,feed_dict={critic_2.s: s, critic_2.a: actions_2,
                                       critic_2.a_o: actions_1, critic_2.target_Q: Q_target_2,
                                       critic_2.trainLength:trace, critic_2.batch_size:batch,
                                       critic_2.state_in:state_train}) 
    
                # Sample the new actions
                new_a_1 = session.run(actor_1.a, feed_dict={actor_1.inputs: s, actor_1.state_in: state_train, 
                                                        actor_1.batch_size:batch,actor_1.trainLength:trace})
                new_a_2 = session.run(actor_2.a, feed_dict={actor_2.inputs: s, actor_2.state_in: state_train, 
                                                        actor_2.batch_size:batch,actor_2.trainLength:trace})
                
                # Calculate the gradients
                gradients_1 = session.run(critic_1.critic_gradients,feed_dict={critic_1.s:s,critic_1.a:new_a_1,
                                                    critic_1.a_o:new_a_2,critic_1.trainLength:trace,
                                                    critic_1.batch_size:batch,critic_1.state_in:state_train})
                gradients_2 = session.run(critic_2.critic_gradients,feed_dict={critic_2.s:s,critic_2.a:new_a_2,
                                                    critic_2.a_o:new_a_1,critic_2.trainLength:trace,
                                                    critic_2.batch_size:batch,critic_2.state_in:state_train})
                gradients_1 = gradients_1[0]
                gradients_2 = gradients_2[0]
                
                # Update the actors
                session.run(actor_1.upd, feed_dict={actor_1.inputs: s, actor_1.state_in: state_train, 
                                              actor_1.critic_gradient:gradients_1, actor_1.batch_size:batch,
                                              actor_1.trainLength:trace})
                session.run(actor_2.upd, feed_dict={actor_2.inputs: s, actor_2.state_in: state_train, 
                                              actor_2.critic_gradient:gradients_2, actor_2.batch_size:batch,
                                              actor_2.trainLength:trace})
    
                # Update target network parameters
                session.run(actor_t_1.update_network_params)
                session.run(critic_t_1.update_network_params)
                session.run(actor_t_2.update_network_params)
                session.run(critic_t_2.update_network_params)
            
            # Update the state
            state_1 = new_state_1
            state_2 = new_state_2
            
            # Update epsilon
            epsilon = rl.getNewEpsilon(epsilon)
            
            # End episode if delta f is too large
            if rl.endEpisode(area.getDeltaF()):
                break
            
        # Append episode to the buffer
        if len(episodeBuffer) >= 8:
            buffer.add(np.array(episodeBuffer))
            
    # Save the model and the data gathered
    saver = tf.train.Saver()
    saver.save(session,"models/maddpg")
    rl.saveData("cummulative_reward_maddpg.pickle",cumm_r_list)