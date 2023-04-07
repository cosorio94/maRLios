from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import random
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from tqdm import tqdm
import numpy as np
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
import action_utils 

class DQNSolver(nn.Module):

    def __init__(self, input_shape):
        super(DQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        action_size = 10
        # 6 unique button presses available in joypadspace = 8 - minus start and select and up
        # We then take a vector of 5 being the initial action, and 6 being the second action
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64), # added a new layer can play with the parameters
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, sampled_actions):
        '''
        x - image being passed in as the state
        sampled_actions - np.array with n x 8 
        '''
        conv_out = self.conv(x).view(x.size()[0], -1)

        # append our actions to conv out and create a pseudo batch for each action
        # Convert the array of actions to a PyTorch tensor
        actions_tensor = torch.from_numpy(sampled_actions)

        # Concatenate the PyTorch tensor with the actions tensor along the second dimension
        batched_actions = torch.cat((conv_out.unsqueeze(1).expand(-1, len(sampled_actions), -1), actions_tensor.unsqueeze(0)), dim=2)
        batched_actions = batched_actions.reshape(len(sampled_actions), -1)

        return self.fc(batched_actions)
    

class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 dropout, exploration_max, exploration_min, exploration_decay, double_dq, pretrained, run_id='', n_actions = 32):

        # Define DQN Layers
        self.state_space = state_space

        self.action_space = action_space # this will be a set of actions ie: a subset of TWO_ACTIONS in constants.py
        self.n_actions = n_actions # initial number of actions to sample
        self.cur_action_space = self.subsample_actions(self.n_actions)

        self.double_dq = double_dq
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # should add in mps if available as well

        # this has been altered as we no longer need to pass the number of actions
        self.local_net = DQNSolver(state_space).to(self.device)
        self.target_net = DQNSolver(state_space).to(self.device)
        
        if self.pretrained:
            self.local_net.load_state_dict(torch.load(f"dq1-{run_id}.pt", map_location=torch.device(self.device)))
            self.target_net.load_state_dict(torch.load(f"dq2-{run_id}.pt", map_location=torch.device(self.device)))
                
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0
    
    

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load(f"STATE_MEM-{run_id}.pt")
            self.ACTION_MEM = torch.load(f"ACTION_MEM-{run_id}.pt")
            self.REWARD_MEM = torch.load(f"REWARD_MEM-{run_id}.pt")
            self.STATE2_MEM = torch.load(f"STATE2_MEM-{run_id}.pt")
            self.DONE_MEM = torch.load(f"DONE_MEM-{run_id}.pt")
            with open(f"ending_position-{run_id}.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open(f"num_in_queue-{run_id}.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 10) # this needs to be a matrix of the actual action taken
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
            self.DONE_MEM = torch.zeros(max_memory_size, 1)
            self.ending_position = 0
            self.num_in_queue = 0
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        

    def subsample_actions(n_actions):
        return action_utils.sample_actions(self.action_space, n_actions)


    def remember(self, state, action, reward, state2, done):
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        
    def recall(self):
        # Randomly sample 'batch size' experiences
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        '''
        Returns the action selected by the agent
        '''
        # Epsilon-greedy action
        
        # increment step
        if self.double_dq:
            self.step += 1

        if random.random() < self.exploration_rate:  
            rand_ind = random.randrange(0, self.cur_action_space.shape[0])

            return torch.tensor(self.cur_action_space[rand_ind])
        
        if self.double_dq:
            # Local net is used for the policy

            # Updated for generalization:
            results = self.local_net(state.to(self.device)).unsqueeze(0).unsqueeze(0).cpu()
            act_index = torch.argmax(results)
            action = torch.tensor(self.cur_action_space)
            # pickup from here

            
            return torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
        
        else: # only training the double dq model
            # return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def copy_model(self):
        # Copy local net weights into target net
        
        self.target_net.load_state_dict(self.local_net.state_dict())
    
    def experience_replay(self):
        
        if self.double_dq and self.step % self.copy == 0:
            self.copy_model()

        if self.memory_sample_size > self.num_in_queue:
            return

        STATE, ACTION, REWARD, STATE2, DONE = self.recall()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        if self.double_dq:
            # Double Q-Learning target is Q*(S, A) <- r + γ max_a Q_target(S', a)
            target = REWARD + torch.mul((self.gamma * 
                                        self.target_net(STATE2).max(1).values.unsqueeze(1)), 
                                        1 - DONE)

            current = self.local_net(STATE).gather(1, ACTION.long()) # Local net approximation of Q-value
        else:
            # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
            target = REWARD + torch.mul((self.gamma * 
                                        self.dqn(STATE2).max(1).values.unsqueeze(1)), 
                                        1 - DONE)
                
            current = self.dqn(STATE).gather(1, ACTION.long())
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        self.exploration_rate *= self.exploration_decay
        
        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)