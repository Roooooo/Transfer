import numpy as np
import tensorflow as tf
from network import MLP
from RL_brain import PolicyGradient
import gym

env = gym.make('MountainCar-v0')

env.seed(1)

n_action = env.action_space.n
n_observation = env.observation_space.shape[0]

real_trajectory_set = []

def learn_policy(
    epoches = 1000,
    DISPLAY_REWARD_THRESHOLD = -11000,
    action_decorator = None
):
    
    RENDER = False

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.995,
        # output_graph=True,
    )

    for i_episode in range(epoches):
    
        observation = env.reset()

        while True:
            if RENDER: env.render()

            action = RL.choose_action(observation)

            if action_decorator is not None:
                action = action_decorator(action) 

            observation_, reward, done, info = env.step(action)     # reward = -1 in all cases

            RL.store_transition(observation, action, reward)

            if done:
                # calculate running reward
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()  # train

                if i_episode == 30:
                    plt.plot(vt)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()

                break

        observation = observation_

    return RL

def learnTrans(epoches = 50000, n_sample = 50, sample_limit = None, hidden_units = 50, hidden_units_2 = 40):
    
    X, Y = [], []

    for i_episode in range(n_sample): # sampling
    
        done = False
        observation = env.reset()

        cnt = 0

        while not done:

            action = np.random.randint(4)

            observation_, reward, done, info = env.step(action)

            action = [0 if x != action else 1 for x in range(4)]

            X.append(np.concatenate((observation,action)))
            Y.append(observation_)

            cnt = cnt + 1

            if sample_limit is not None and cnt > limit: break  

    trans = MLP(n_observation+n_action,n_observation,
        n_hidden_units=hidden_units,
        n_hidden_units_2=hidden_units_2,
        optimizer = tf.train.AdamOptimizer,
        activation = tf.nn.relu
    )

    trans.ep_obs = X
    trans.ep_as = Y

    print len(X)

    trans.learn(iters = epoches)

    return trans

def get_explore_reward(explore_policy, policy, lamda = 0.5, n_sample = 10):
    reward = 0
    for iter in xrange(n_sample):
        observation = env.reset()
        done = False
        
        trajectory = [observation]

        while not done:
            action = explore_policy.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            trajectory.push(observation_)

            observation = observation_

        reward = reward + trajectory_dist(trajectory, traj_set) * lamda + prob(policy, trajectory)
    
    return reward / 10

def prob(policy, trajectory):
    pass

def get_dist(src, dst):
    # TODO: Consider if src and dst have different length
    return np.sum(np.linalg.norm(src-dst,axis = 1))

def trajectory_dist(trajectory, traj_set):
    min_dist = None
    for elem in traj_set:
        dist = get_dist(trajectory, elem)
        if min_dist is None:
            min_dist = dist
        else:
            min_dist = min(min_dist, dist)
    return min_dist
    
def learn_explore_policy(action_decorator = None):
    
    

if __name__ == '__main__':
    
    # Learning transition

    trans = learnTrans(
        epoches = 10000,
        n_sample = 15,
        sample_limit = 100,
    )

    # should be initialized to zero
    residue = MLP(
        n_observation+n_action,n_observation,
        n_hidden_units=150,
        n_hidden_units_2=150,
        optimizer = tf.train.AdamOptimizer,
        activation = tf.nn.relu
    )

    while true:
        
        #   RL - get best policy on new transition
        # decorator should be a function in the form: (s, a) -> a'
        policy = learn_policy(action_decorator = trans.predict)

        #   try policy on real env
        observation = env.reset()
        running_reward = 0
        done = False

        real_trajectory_set.push([observation])

        while not done:
            
            action = policy.choose_action(observation)
            action = real_env_action_decorator(action)

            observation_, reward, done, info = env.step(action)

            running_reward = running_reward * 0.99 + reward

            real_trajectory_set[-1].push(action)
            real_trajectory_set[-1].push(observation_)
            

    #   if policy is good enough : break
        if running_reward >= reward_lower_bound: break

    #   RL - get exploration Policy
    
    #   TODO: learn_explore_policy() and get_explore_reward()
    #         save trajectory
        explore_policy = learn_explore_policy()

    #   Update transition residue
    #   Sampling
        observation = env.reset()
        done = False

        while not done:
            action = explore_policy.choose_action(observation)
            action = real_env_action_decorator(action)

            observation_, reward, done, info = env.step(action)

            # save residue
        
        residue.train()