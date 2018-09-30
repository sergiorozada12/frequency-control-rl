# -*- coding: utf-8 -*-
"""
@author: Sergio Rozada Doval

@description: This scripts implements deep deterministic policy gradient algorithm.
This is an actor-critic architecture that aims to produce a continuous deterministic
action for the system to operate. The critic helps to estimate how good actions
are. The actor is the part of the architecture that produces the action.

""" 

import dynamics as dn
import rl
import tensorflow as tf
import numpy as np
import architectures

""" Definition of the model using tensorflow"""

tf.reset_default_graph()
h_size = 100

# Create actor, critic, main and target networks
lstm_actor = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor = architectures.actor_ddpg(h_size,lstm_actor,'actor',0)
critic = architectures.critic_ddpg(h_size,lstm_critic,'critic',len(tf.trainable_variables()))

lstm_actor_t = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
lstm_critic_t = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
actor_t = architectures.actor_ddpg(h_size,lstm_actor_t,'actor_t',len(tf.trainable_variables()))
critic_t = architectures.critic_ddpg(h_size,lstm_critic_t,'critic_t',len(tf.trainable_variables()))

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
n_var = 4
trace = 8
batch = 4
cumm_r_list = []

# Create op holder for the target networks
actor_t.createOpHolder(actor.network_params,tau)
critic_t.createOpHolder(critic.network_params,tau)

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
        state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        
        # Iterate all over the steps
        for j in range(steps):
            
            # Get the action from the actor and the internal state of the rnn
            current_f = area.getDeltaF()
            a, new_state = session.run([actor.a,actor.rnn_state], 
                            feed_dict={actor.inputs: np.array(current_f).reshape(1,1),
                            actor.state_in: state, actor.batch_size:1,actor.trainLength:1})
            a = a[0,0] + epsilon*np.random.normal(0.0,0.2)
            
            # Take the action, modify environment and get the reward
            generator = rl.setContinuousPower(a,generator)
            area.calculateDeltaF([generator,load])
            new_f = area.getDeltaF()
            r = rl.getSimpleReward(new_f)
            cumm_r += r

            # Store the experience and print some data
            experience = np.array([current_f,new_f,a,r])
            episodeBuffer.append(experience)
            print("Delta f: ",round(current_f,2)," Action: ",a, " Reward: ",r)
            
            # Update the model each 4 steps with a minibatch of 32
            if ((j % 4) == 0) & (i > 0) & (len(buffer.buffer)>0):
                
                # Sample the miniBatch
                miniBatch = buffer.sample(batch,trace,n_var)
                
                # Reset the recurrent layer's hidden state and get states
                state_train = (np.zeros([batch,h_size]),np.zeros([batch,h_size]))
                s = np.reshape(miniBatch[:,0],[32,1])
                s_prime = np.reshape(miniBatch[:,1],[32,1])
                actions = np.reshape(miniBatch[:,2],[32,1])
                rewards = np.reshape(miniBatch[:,3],[32,1])

                # Predict target Q with the actor and the critic target networks
                a_target = session.run(actor_t.a, feed_dict={actor_t.inputs: s_prime,
                            actor_t.state_in: state_train, actor_t.batch_size:batch,actor_t.trainLength:trace})
                Q_target = session.run(critic_t.Q,feed_dict={critic_t.s: s_prime, critic_t.a: a_target,
                                                    critic_t.trainLength:trace,critic_t.batch_size:batch,
                                                    critic_t.state_in:state_train})   
                Q_target = rewards + gamma*Q_target

                # Update the critic network with the new Q's
                upd,loss = session.run([critic.upd,critic.loss],feed_dict={critic.s: s, critic.a: actions, 
                                       critic.target_Q: Q_target, critic.trainLength:trace,
                                       critic.batch_size:batch,critic.state_in:state_train})   
    
                # Sample the action gradients and update the actor
                new_a = session.run(actor.a, feed_dict={actor.inputs: s, actor.state_in: state_train, 
                                                        actor.batch_size:batch,actor.trainLength:trace})                
                gradients = session.run(critic.critic_gradients,feed_dict={critic.s:s,critic.a:new_a,
                                                    critic.trainLength:trace,critic.batch_size:batch,
                                                    critic.state_in:state_train})
                gradients = gradients[0]
                upd,grad = session.run([actor.upd,actor.actor_gradients], feed_dict={actor.inputs: s, 
                                       actor.state_in: state_train, actor.critic_gradient:gradients,
                                                  actor.batch_size:batch,actor.trainLength:trace})
    
                # Update target network parameters
                session.run(actor_t.update_network_params)
                session.run(critic_t.update_network_params)
            
            # Update the state
            state = new_state
            
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
    saver.save(session,"models/ddpg")
    rl.saveData("cummulative_reward_ddpg.pickle",cumm_r_list)
            
            
                                       