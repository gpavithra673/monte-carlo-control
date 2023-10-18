# MONTE CARLO CONTROL ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## MONTE CARLO CONTROL ALGORITHM
Include the steps involved in the Monte Carlo control algorithm

## MONTE CARLO CONTROL FUNCTION
```
from tqdm import tqdm
def mc_control(env, gamma = 1.0, init_alpha = 0.5, min_alpha = 0.01,
               alpha_decay_ratio = 0.5, init_epsilon = 1.0, min_epsilon = 0.1,
               epsilon_decay_ratio = 0.9, n_episodes = 3000, max_steps = 200,
               first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n

  discounts = np.logspace(0,max_steps, num=max_steps,
                          base=gamma, endpoint = False)

  alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

  epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

  pi_track = []

  Q = np.zeros((nS, nA), dtype = np.float64)
  Q_track = np.zeros((n_episodes, nS, nA), dtype = np.float64)

  select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon_decay_ratio else np.random.randint(len(Q[state]))

  for e in tqdm(range(n_episodes), leave = False):
    trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)
    visited = np.zeros((nS, nA), dtype = np.bool)
    for t, (state, action, reward, _, _) in enumerate(trajectory):
      if visited[state][action] and first_visit:
        continue
      visited[state][action] = True
      n_steps = len(trajectory[t:])
      G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
      Q[state][action] += alphas[e] * (G - Q[state][action])
    Q_track[e] = Q
    pi_track.append(np.argmax(Q, axis = 1))
  V = np.max(Q, axis = 1)
  pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis = 1))}[s]
  return Q, V, pi
```
## OUTPUT:


## RESULT:
