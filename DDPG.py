import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.optim import Adam,SGD
from torchcontrib.optim import SWA

from src.utils import ReplayBuffer, OU_Noise, Experience
from src.nets import Actor,Critic

if torch.cuda.is_available() : 
        device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
else :
    device = "cpu"

def run_DDPG(env = "Pendulum-v0", 
            seed = None, 
            buffer_size = 6000000,

            hidden_layers = (400,300),
            memory_batch_size = 64, 
            lr_Actor = 1e-4,
            lr_Critic = 1e-3,

            nr_of_test_cycles = 50,
            nr_of_episodes = 300,
            episode_time_limit = 3000 ,
            warmup_value= 100,

            gamma = 0.99,
            tau = 0.001,
            noise_stddev = 0.2,

            SWA_freq =  100,
            SWA_lr = 0.05,
            SWA_start = None,

            testing_model = 'actor_target' # we can test either using actor or actor_target
            ):
    '''
    Wrapper around a DDPG training and testing process

        Parameters:
                env (int): A gym environemnt. List here: https://github.com/StanfordVL/Gym/blob/master/docs/environments.md
                seed (int/None): If None - random seed, otherwise deterministic
                buffer_size (int): Size of memory buffer
                hidden_layers tuple(int,int): two int values representing hidden layers of our neural networks (Actor and Critic have same hidden layers by)
                memory_batch_size(int): Specifies number of random N transitions taken from Buffer in training loop
                lr_Actor(float): learning rate Actor
                lr_Critic(float): learning rate Critic
                nr_of_test_cycles(int): how many tests cycles to run (we base our results on averages of those runs)
                nr_of_episodes(int): number of traning cycles
                episode_time_limit(int): Variable specifies how long should the training cycle last for till we give up. Training either terminates based on finished(won/lost) game or going over episode time limit
                warmup_value(int): Variable specifies how many training episodes/cycles to run before starting training. This is used to fill in our memory buffer with data.
                gamma(float): Ornstein–Uhlenbeck noise variable
                tau(float): Ornstein–Uhlenbeck noise variable
                noise_stddev(float): Ornstein–Uhlenbeck noise variable

                SWA_freq(int): Variable defines how often should SWA save weights. (per how many episodes)
                SWA_lr(int): learning rate of SWA 
                SWA_start(None/int): After how many episodes should SWA start saving weights. If None, SWA will by default be set to start after 70% of training has been completed

                testing_model(str): Either 'actor_target' or 'actor'. Defines which model to use for testing






        Returns:
            nets, optimizers, memory

                nets - list of nets in order: [actor, critic, actor_target, critic_target]
                optimizers - list of optimizers in order: [actor_optimizer, actor_optimizer2, critic_optimizer, critic_optimizer2]
                memory - current MemoryBuffer
    '''


    ENV = gym.make(env)
    ACTION_SPACE = ENV.action_space
    NUM_INPUTS = ENV.observation_space.shape[0]
    NUM_ACTIONS = ENV.action_space.shape[-1]

    # Handle inputs
    if seed != None:
        ENV.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
    
    if SWA_start == None: # By default set SWA to start working once we've started reaching convergence. Arbitrary value based on previous runs and notes - research how to make it less arbitrary.
        SWA_start = 0.7 * nr_of_episodes
    
    if testing_model not in ['actor_target', 'actor']:
        raise Exception(f"testing_model variable has to be either `actor_target` or `actor`. You've input: {testing_model}")

    if len(hidden_layers)!= 2:
        raise Exception(f'Currentely tested Neural Nets only handle two layers - hidden_layers variable has to be in form (int,int) but was: {hidden_layers}')

    test_rewards_SWA , rewards_list, mean_test_rewards, episode_list , test_rewards, test_episode_list = [], [], [], [], [], []


    #DDPG ALOGRITHM : 
    #1)RANDOMLY Initialize Critic network and actor network 
    actor = Actor(hidden_layers, NUM_INPUTS, ACTION_SPACE).to(device)
    critic = Critic(hidden_layers, NUM_INPUTS, ACTION_SPACE).to(device)


    #2)Initialize targer actor/critic with same weights as actor/critic
    actor_target = Actor(hidden_layers, NUM_INPUTS, ACTION_SPACE).to(device)
    critic_target = Critic(hidden_layers, NUM_INPUTS, ACTION_SPACE).to(device)

    for target_parameter, parameter in zip (actor_target.parameters(),actor.parameters()):
        target_parameter.data.copy_(parameter.data)
        
    for target_parameter, parameter in zip (critic_target.parameters(),critic.parameters()):
        target_parameter.data.copy_(parameter.data)




    #Optimizers*
    actor_optimizer2 = SGD(actor.parameters(),
                                lr=lr_Actor)  
    critic_optimizer2 = SGD(critic.parameters(),
                                    lr=lr_Critic,
                                    weight_decay=1e-2
                                    )
    #implement SWA  
    actor_optimizer = SWA(actor_optimizer2,swa_lr= SWA_lr) 

    critic_optimizer = SWA(critic_optimizer2,swa_lr= SWA_lr) 



    #3)Initialize replay Buffer R
    memory = ReplayBuffer(buffer_size)

    #4)For loop, episode 1 --> M :
    loop_count = 0
    for episode in range (nr_of_episodes):
        episode_return = 0
        curr_t = 0
        
        
        #4.1)Initialize a random Process OU
        ou_noise = OU_Noise(mu=np.zeros(NUM_ACTIONS), sigma=float(noise_stddev) * np.ones(NUM_ACTIONS))

        #4.2)Receive initial observation of stat
        state = torch.Tensor([env.reset()]).to(device)

        #4.3)for loop , t=1 -->T :
        while (True and curr_t<episode_time_limit) :  #Just for clarity , it finishes if either we run out of moves or time 
            curr_t+=1
            loop_count+=1
            
            #4.3.1)select action chosen by actor and add OUth random process to the choice 
            actor.eval()  # Sets the actor in evaluation mode
            action = actor(state)
            actor.train()  # Sets the actor in training mode
            noise = torch.Tensor(ou_noise.sample_noise()).to(device)
            action = action.data  + noise
            action = action.clamp(ACTION_SPACE.low[0], ACTION_SPACE.high[0]) 
            #^^ to make sure that after adding noise to action we don't end-up with action outside of action space

            
            #4.3.2)Execute selected action and observe : reward,new state
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            episode_return += reward

            
            #4.3.3)Store previous state,action,reward,and new state in Buffer R
            terminal = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, next_state, reward,terminal)
            state = next_state

            #4.3.4)Sample a random minibatch of N transitions from Buffer R
            if len(memory)> warmup_value:
                Experiences = memory.sample(memory_batch_size)
                batch = Experience(*zip(*Experiences))

                states = torch.cat(batch.state).to(device)
                terminals = torch.cat(batch.terminal).to(device)
                actions = torch.cat(batch.action).to(device)
                rewards = torch.cat(batch.reward).to(device)
                next_states = torch.cat(batch.next_state).to(device)


                #4.3.5) calculate batch of Q values dependent on target networks 
                next_actions = actor_target(next_states)
                next_state_action_values = critic_target(next_states, next_actions.detach())
                
                rewards = rewards.unsqueeze(1)
                terminals = terminals.unsqueeze(1)
                Q_values_by_target = rewards + (1.0 - terminals) * gamma * next_state_action_values
                
                
                #4.3.6)Update critic using mean squared error loss between Q_values calc on target vs those by non-target
                critic_optimizer.zero_grad()
                
                Q_values_by_non_target = critic(states, actions)
                critic_loss_function = F.mse_loss(Q_values_by_non_target, Q_values_by_target.detach())
                
                critic_loss_function.backward()
                critic_optimizer.step()

                
                #4.3.7)Update actor using Policy Gradient calculated by non_target networks
                actor_optimizer.zero_grad()
                
                policy_loss = -critic(states, actor(states))
                policy_loss = policy_loss.mean()
                
                policy_loss.backward()
                actor_optimizer.step()

                
                #SWA  own implementation
                if (loop_count==SWA_start) :
                    params1 = actor.named_parameters()
                    nr_of_SWA_updates = 1

                if (loop_count%SWA_freq==0 and loop_count > SWA_start) :
                    params2 = actor.named_parameters()
                    dict_params1 = dict(params1)  
                    nr_of_SWA_updates += 1 
                    SWA_update_factor = (1/nr_of_SWA_updates)

                    for name2, param2 in params2:
                        if name2 in dict_params1:
                            dict_params1[name2].data.copy_(SWA_update_factor*param2.data / + (1-SWA_update_factor)*dict_params1[name2].data)

                    ###hard update :    
                #4.3.8)Update both targer function by tau
                for p_target , p in zip(actor_target.parameters(),actor.parameters()):
                    p_target.data.copy_(p_target.data * (1.0 - tau) +  p.data * tau)
                    
                for p_target , p in zip(critic_target.parameters(),critic.parameters()):
                    p_target.data.copy_(p_target.data * (1.0 - tau) +  p.data * tau)
            if done:
                break
        rewards_list.append(episode_return)
        episode_list.append(episode)


    #########################    TESTING BEFORE SWA WEIGHT SWAP   #####################################

    if testing_model == "actor_target" :
        testing_model = actor_target
    else : 
        testing_model = actor
        
    for episode in range (nr_of_test_cycles) : 
        
        ou_noise = OU_Noise(mu=np.zeros(NUM_ACTIONS), sigma=float(noise_stddev) * np.ones(NUM_ACTIONS))
        test_episode_return = 0
        state = torch.Tensor([env.reset()]).to(device)

        while True : 
            testing_model.eval()  # Sets the actor in evaluation mode
            action = testing_model(state)
            testing_model.train()  # Sets the actor in training mode
            noise = torch.Tensor(ou_noise.sample_noise()).to(device)
            action = action.data  + noise
            action = action.clamp(ACTION_SPACE.low[0], ACTION_SPACE.high[0])

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

            test_episode_return += reward
            next_state = torch.Tensor([next_state]).to(device)
            state = next_state

            if done:
                break

        #append lists so we can graph them later 
        test_episode_list.append(episode)
        test_rewards.append(test_episode_return)

    ##################   Testing after SWA swap    ########################
    actor_optimizer.swap_swa_sgd()
    critic_optimizer.swap_swa_sgd()

    #since we're using batchNorm we've to pass normalization through our model : 
    #opt.bn_update(train_loader = , model=Actor)
    for episode in range (nr_of_test_cycles) : 
        
        test_episode_return = 0
        ou_noise = OU_Noise(mu=np.zeros(NUM_ACTIONS), sigma=float(noise_stddev) * np.ones(NUM_ACTIONS))
        state = torch.Tensor([env.reset()]).to(device)

        while True : 
            actor.eval()  # Sets the actor in evaluation mode
            action = actor(state)
            actor.train()  # Sets the actor in training mode
            noise = torch.Tensor(ou_noise.sample_noise()).to(device)
            action = action.data  + noise
            action = action.clamp(ACTION_SPACE.low[0], ACTION_SPACE.high[0])

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

            test_episode_return += reward
            next_state = torch.Tensor([next_state]).to(device)
            state = next_state

            if done:
                break

        #append lists so we can graph them later 
        test_rewards_SWA.append(test_episode_return)
    env.close()




    #plot the results 
    plt.subplot(2,1,1)
    plt.plot(episode_list,rewards)
    plt.xlabel("episodes")
    plt.ylabel("training rewards")
    plt.title("training rewards , no SWA swaps inbetween")
    plt.legend(["rewards"])

    plt.subplot(2,1,2)
    plt.plot(test_episode_list,test_rewards)
    plt.plot(test_episode_list,test_rewards_SWA)
    plt.xlabel("test episodes")
    plt.ylabel("test rewards")
    plt.legend(["test_rewards","SWA_test_rewards"])

    plt.show()


    # Returns
    nets = [actor, critic, actor_target, critic_target]
    optimizers = [actor_optimizer, actor_optimizer2, critic_optimizer, critic_optimizer2]
    return nets, optimizers, memory