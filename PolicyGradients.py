from tensorflow import keras
from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
import pandas as pd

class PolicyGradients:
    def __init__(self, inputdims, actiondims, layers, neurons):
        inputdims = inputdims
        self.actiondims = actiondims
        layers = layers
        neurons = neurons
        self.model = self.buildnetwork(inputdims, self.actiondims, layers, neurons)
        
    def buildnetwork(self, inputdims, outputdims, layers, neurons):
        model = keras.Sequential()
        model.add(keras.Input(shape=(inputdims,)))
        for l in range(layers):
            model.add(keras.layers.Dense(neurons, activation = keras.activations.relu))
        model.add(keras.layers.Dense(outputdims, activation = keras.activations.softmax))
        
        optimizer = keras.optimizers.Adamax()
        model.compile(optimizer = optimizer)
        return model

    def choose_action(self, state):
        state = tf.expand_dims(state,0)
        probs = self.model(state)
        action = np.argmax(probs, axis = 1)[0]
        return action
    
    def store_transition(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    def restart_transition(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    def customloss(self, action, prob, delta, batch_train):
        out = k.clip(prob, 1e-8, 1-1e-8)
        if batch_train:
            loglik = action * -k.log(out)
            loglik = k.sum(loglik, axis=1) * delta
            loglik = k.sum(loglik) / len(loglik)
            #print(loglik)
        else:
            loglik = action * -k.log(out)
            loglik = k.sum(loglik, axis=1) * delta
        return loglik
    
    def learn(self, batch_train = True):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        
        actions = np.zeros( [len(action_memory),self.actiondims] )
        actions[ np.arange(len(action_memory)), action_memory ] = 1
        
        if batch_train:
            with tf.GradientTape() as tape:
                probs = self.model(state_memory)
                loss = self.customloss(actions, probs, reward_memory, batch_train)
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        else:
            for i in range(len(state_memory)):
                with tf.GradientTape() as tape:
                    pred = self.model(state_memory[i:i+1])
                    loss = self.customloss(actions[i], pred, reward_memory[i], batch_train)
                grad = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))