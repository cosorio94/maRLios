import datetime
import torch
import torch.nn as nn
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from tqdm import tqdm
import pickle 
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gym
import numpy as np
import matplotlib.pyplot as plt

from toolkit.gym_env import *
from toolkit.model import *
import argparse
import time
import ast

action_mapping = {
    'SIMPLE_MOVEMENT': SIMPLE_MOVEMENT,
    'RIGHT_ONLY': RIGHT_ONLY,
    'COMPLEX_MOVEMENT': COMPLEX_MOVEMENT
}

def vectorize_action(action, action_space):
    # Given a scalar action, return a one-hot encoded action
    
    return [0 for _ in range(action)] + [1] + [0 for _ in range(action + 1, action_space)]

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    # display(plt.gcf(), clear=True)

def make_env(env, actions=SIMPLE_MOVEMENT):
    env = MaxAndSkipEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return JoypadSpace(env, actions)

def generate_epoch_time_id():
    epoch_time = int(time.time())
    return str(epoch_time)

def save_checkpoint(agent, total_rewards, run_id):
    with open(f"ending_position-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.ending_position, f)
    with open(f"num_in_queue-{run_id}.pkl", "wb") as f:
        pickle.dump(agent.num_in_queue, f)
    with open(f"total_rewards-{run_id}.pkl", "wb") as f:
        pickle.dump(total_rewards, f)
    if agent.double_dq:
        torch.save(agent.local_net.state_dict(), f"dq1-{run_id}.pt")
        torch.save(agent.target_net.state_dict(), f"dq2-{run_id}.pt")
    else:
        torch.save(agent.dqn.state_dict(), f"dq-{run_id}.pt")  
    torch.save(agent.STATE_MEM,  f"STATE_MEM-{run_id}.pt")
    torch.save(agent.ACTION_MEM, f"ACTION_MEM-{run_id}.pt")
    torch.save(agent.REWARD_MEM, f"REWARD_MEM-{run_id}.pt")
    torch.save(agent.STATE2_MEM, f"STATE2_MEM-{run_id}.pt")
    torch.save(agent.DONE_MEM,   f"DONE_MEM-{run_id}.pt")

def main(training_mode=True, pretrained=False, lr=0.00025, gamma=0.90, exploration_decay=0.99, exploration_min=0.02,
        mario_env='SuperMarioBros-1-1-v0', actions=SIMPLE_MOVEMENT, num_episodes=10, run_id=None, exploration_max=1.0):
   
    run_id = run_id or generate_epoch_time_id()
    fh = open(f'progress-{run_id}.txt', 'a')
    env = gym.make(mario_env)
    #env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    #env = make_env(env)  # Wraps the environment so that frames are grayscale 
    #env = SuperMarioBrosEnv()
    env = make_env(env, actions)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n

    #todo: add agent params as a setting/create different agents in diff functions to run 

    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=gamma,
                     lr=lr,
                     dropout=0.,
                     exploration_max=exploration_max,
                     exploration_min=exploration_min,
                     exploration_decay=exploration_decay,
                     double_dq=True,
                     pretrained=pretrained,
                     run_id=run_id)
    
    
    # num_episodes = 10
    env.reset()
    total_rewards = []
    
    for ep_num in tqdm(range(num_episodes), file=fh):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(env, ep_num)
            action = agent.act(state)
            steps += 1
            
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            
            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
            
            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        if training_mode and (ep_num % 300) == 0:
            save_checkpoint(agent, total_rewards, run_id)

        with open(f'total_reward-{run_id}.txt', 'a') as f:
            f.write("Total reward after episode {} is {}\n".format(ep_num + 1, total_rewards[-1]))
            if (ep_num%100 == 0):
                f.write("==================\n")
                f.write("{} current time at episode {}\n".format(datetime.datetime.now(), ep_num+1))
                f.write("==================\n")
            #print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
            num_episodes += 1
    
    if training_mode:
        save_checkpoint(agent, total_rewards, run_id)
    
    env.close()
    fh.close()
    
    # if num_episodes > 500:
    #     plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
    #     plt.plot([0 for _ in range(500)] + 
    #              np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
    #     plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs MaRLios agent on Super Mario bros. gym environment")

    parser.add_argument("--training-mode", type=ast.literal_eval, default=True, help="Training mode (default: True)")
    parser.add_argument("--pretrained", type=ast.literal_eval, default=False, help="Use pretrained model (default: False)")
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate (default: 0.00025)")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor (default: 0.90)")
    parser.add_argument("--exploration-decay", type=float, default=0.99, help="Exploration decay (default: 0.99)")
    parser.add_argument("--exploration-min", type=float, default=0.02, help="Exploration minimum (default: 0.02)")
    parser.add_argument("--exploration-max", type=float, default=0.02, help="Exploration maximum (default: 1.00)")
    parser.add_argument("--mario-env", type=str, default='SuperMarioBros-1-1-v0', help="Mario environment (default: 'SuperMarioBros-1-1-v0')")
    parser.add_argument("--actions", type=str, default='SIMPLE_MOVEMENT', help="Actions (default: 'SIMPLE_MOVEMENT')")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes (default: 10)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: epoch timestring)")

    args = parser.parse_args()
    print('test: ', args)

    if args.actions in action_mapping:
        actions = action_mapping[args.actions]
    else:
        raise ValueError("Invalid actions argument.")

    main(training_mode=args.training_mode,
         pretrained=args.pretrained,
         lr=args.lr,
         gamma=args.gamma,
         exploration_decay=args.exploration_decay,
         exploration_min=args.exploration_min,
         exploration_max=args.exploration_max,
         mario_env=args.mario_env,
         actions=actions,
         num_episodes=args.num_episodes,
         run_id=args.run_id)
