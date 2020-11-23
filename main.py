import os
import gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
pos_list = [-1.2, -1.1, -1.0, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0.0, .1, .2, .3, .4, .5, .6]
speed_list = [-.07, -.06, -0.05, -.04, -.03, -.02, -.01, 0.0, .01, .02, .03, .04, .05, .06, .07]
env = gym.make('MountainCar-v0')
env.reset()

# Some code to generate a new folder for each run
path = 'images'
folders = os.listdir(path)
fmax = 0
for folder in folders:
    if folder[0] != '.':
        f = int(folder)
        if f > fmax:
            fmax = f
fmax += 1
os.makedirs('images/' + str(fmax))


def discretize(state):
    position = round(state[0], 1)
    speed = round(state[1], 2)
    state_adj = (position, speed)
    return state_adj


def q_init():
    # Create a uniform distribution between -1 and 1 for the Q table
    Q_in = np.random.uniform(low=-1, high=1, size=(15,  19, env.action_space.n))
    Q = {}
    for i, p in enumerate(pos_list):
        for j, s in enumerate(speed_list):
            # We create a dict with as key the state, and as value the reward
            state = (p, s)
            Q[state] = Q_in[j][i]
    return Q


def qlearning(env, learning, discount, epsilon, min_eps, episodes):
    # Uniform distributed State Action pair to reward
    Q = q_init()

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # The amount of epsilon decay
    reduction = (epsilon - min_eps) / episodes

    for i_episode in range(episodes):
        done = False
        state = env.reset()
        state_d = discretize(state)

        total_reward = 0

        while not done:
            # Display the last 20 episodes
            if i_episode > episodes - 20:
                env.render()

            # Epsilon greedy between Q value or random
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_d])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Do the action and record the state and reward
            next_state, reward, done, info = env.step(action)
            next_state_d = discretize(next_state)

            # If the goal is reached or 200 timesteps have been reached
            if done and next_state_d[0] >= 0.5:
                Q[state_d][action] = reward

            # The simple Bellmann update
            else:
                delta = learning * (reward + discount * np.max(Q[next_state_d]) - Q[state_d][action])
                Q[state_d][action] += delta

            # Record the total reward over all timesteps (max = -200)
            total_reward += reward
            # Update the current state to the next state
            state_d = next_state_d

        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction

        # Track rewards
        reward_list.append(total_reward)

        # Some printing and image generation
        if (i_episode + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            print('Episode {} Average Reward: {}'.format(i_episode + 1, ave_reward))
            heat_display(Q, episode=i_episode+1, save=True)

    return ave_reward_list, Q


# To display the value function (and action mapping?)
# qora indicates which type to save (value=0, action=1, both=2)
# save will save the image(s) to a new folder
def heat_display(Q, episode, qora=0, save=False):
    r_img = np.zeros((15, 19))
    a_img = np.zeros((15, 19))
    for i, state in enumerate(Q.keys()):
        row = i // 15
        col = i % 15
        r_img[col][row] = np.max(Q[state])
        a_img[col][row] = np.argmax(Q[state])

    if qora == 0 or qora == 2:
        plt.imshow(r_img, cmap='viridis')
        plt.colorbar()
        plt.xticks(ticks=list(range(len(pos_list))), labels=pos_list)
        plt.yticks(ticks=list(range(len(speed_list))), labels=speed_list)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.title("Value function after {e} episodes".format(e=episode))
        if not save:
            plt.show()
        else:
            plt.savefig('images/{}/r_img_{}'.format(fmax, episode))
            plt.close()

    if qora == 1 or qora == 2:
        plt.imshow(a_img, cmap='seismic')
        plt.colorbar()
        plt.xticks(ticks=list(range(len(pos_list))), labels=pos_list)
        plt.yticks(ticks=list(range(len(speed_list))), labels=speed_list)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.title("Action function after {e} episodes".format(e=episode))
        if not save:
            plt.show()
        else:
            plt.savefig('images/{}/a_img_{}'.format(fmax, episode))
            plt.close()


if __name__ == '__main__':
    # Run Q-learning algorithm

    num_episodes = 5000
    rewards, Q = qlearning(env=env,
                           learning=0.2,
                           discount=0.9,
                           epsilon=0.8,
                           min_eps=0,
                           episodes=num_episodes)

    # Plot Value function and action mapping
    heat_display(Q, num_episodes, 2)

    # Plot Rewards
    plt.plot(100 * (np.arange(len(rewards)) + 1), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.savefig('rewards.jpg')
    plt.close()




