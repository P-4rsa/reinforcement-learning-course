import gym
import numpy as np 
import time
env = gym.make('FrozenLake-v1', render_mode='human')

# Initialize a random policy (for each state, each action has equal probability)
random_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
# تعریف یک سیاست تصادفی دوم (random_policy_2) به صورتی که اعداد به صورت تصادفی باشند
# سپس هر ردیف (هر حالت) را نرمالایز می‌کنیم تا مجموع احتمالات برابر با 1 شود.
random_policy_2 = np.random.rand(env.observation_space.n, env.action_space.n)
for i in range(env.observation_space.n):
    random_policy_2[i] /= np.sum(random_policy_2[i])
gamma = 0.9
print(random_policy_2)
def my_policy_evaluation(policy):
    prev_value = np.zeros(env.observation_space.n)  # initialize value function
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        value = np.zeros(env.observation_space.n)
        for state in range(env.observation_space.n):
            for action in range(env.action_space.n):
                for prob, next_state, reward, _ in env.P[state][action]:
                    # Use the provided 'policy' instead of the global 'random_policy'
                    value[state] += prob * policy[state][action] * (reward + gamma * prev_value[next_state])
        delta = np.max(np.abs(value - prev_value))
        if delta < 0.0001:
            break
        prev_value = np.copy(value)
    return value, iteration

def policy_improvement(v):
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    q = np.zeros([env.observation_space.n, env.action_space.n])
    
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            for prob, next_state, reward, _ in env.P[state][action]:
                q[state][action] += prob * (reward + gamma * v[next_state])
    
    max_index = np.argmax(q, axis=1)
    for i, index in enumerate(max_index):
        policy[i][index] = 1  # Greedy policy: select the action with highest Q-value
    
    return policy

def policy_iteration(policy): 
    iteration = 0
    while True:
        iteration += 1
        v, _ = my_policy_evaluation(policy)   # Evaluate the current policy
        new_policy = policy_improvement(v)       # Improve the policy based on v
        if (new_policy == policy).all():         # If the policy did not change, we're done
            break
        policy = new_policy                      # Update policy for next iteration
    return policy,iteration
start = time.time()
final_policy, iteration = policy_iteration(random_policy)
end = time.time()

start = time.time()
final_policy_2, iteration_2 = policy_iteration(random_policy_2)
end = time.time()

if (final_policy == final_policy_2).all()==True:
    print('Both policies are the same.')
# print(final_policy)
# print(f'Policy iteration converged in {iteration} iterations.')
# print(f'time{end-start:.2f}s')


