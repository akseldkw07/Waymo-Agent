import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. Device Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. Custom City Environment (Multi-Agent)
# ---------------------------------------------------------

class CityTaxiEnv(gym.Env):
    """
    A 4x5 Grid City Environment for Taxi Service (100 Cars).

    Nodes: 0-19 (Row-major order)

    Action Space (Discrete 5):
    0: Pick Up Passenger
    1: Move Up, 2: Move Down, 3: Move Left, 4: Move Right

    Observation Space (Box 21):
    [My_Taxi_Loc, Demand_Node_0, ..., Demand_Node_19]
    """

    def __init__(self, num_taxis=100):
        super(CityTaxiEnv, self).__init__()

        self.rows = 4
        self.cols = 5
        self.num_nodes = self.rows * self.cols
        self.num_taxis = num_taxis

        # Action space: 0=Pickup, 1-4=Move
        self.action_space = spaces.Discrete(5)

        # Observation: My Position (1) + Demand Map (20)
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(1 + self.num_nodes,),
            dtype=np.float32
        )

        self.max_steps = 72 # 24 hours / 20 min steps
        self.current_step = 0
        self.taxi_locs = np.zeros(self.num_taxis, dtype=int)
        self.demand = np.zeros(self.num_nodes)

    def _get_coords(self, node):
        return node // self.cols, node % self.cols

    def _get_node(self, r, c):
        return r * self.cols + c

    def _generate_demand(self):
        """Generates Poisson demand for the step."""
        demand = np.zeros(self.num_nodes)
        # Upper 10 nodes (High Demand)
        for i in range(10):
            demand[i] = np.random.poisson(lam=0.8)
        # Lower 10 nodes (Low Demand)
        for i in range(10, 20):
            demand[i] = np.random.poisson(lam=0.2)
        return demand

    def _get_observation_batch(self):
        """Returns (num_taxis, 21) array - each taxi sees its loc + global demand."""
        # Vectorized optimization for 100 cars
        # Shape: (Num_Taxis, 1)
        locs = self.taxi_locs.reshape(-1, 1)
        # Shape: (Num_Taxis, 20) - Repeat demand for every car
        demands = np.tile(self.demand, (self.num_taxis, 1))
        # Concatenate: (Num_Taxis, 21)
        return np.concatenate((locs, demands), axis=1).astype(np.float32)

    def reset(self):
        self.current_step = 0
        # Start all taxis at random nodes
        self.taxi_locs = np.random.randint(0, self.num_nodes, size=self.num_taxis)
        self.demand = self._generate_demand()
        return self._get_observation_batch()

    def step(self, actions):
        """
        actions: list or array of shape (num_taxis,)
        """
        rewards = np.zeros(self.num_taxis)
        done = False

        # Track for viz
        picked_up_map = np.zeros(self.num_nodes)

        # Process each taxi's action
        # Note: We iterate sequentially. This creates a race condition where
        # lower-index taxis get first dibs on passengers at the same node.
        for i in range(self.num_taxis):
            action = actions[i]
            loc = self.taxi_locs[i]
            r, c = self._get_coords(loc)

            if action == 0: # PICK UP
                if self.demand[loc] > 0:
                    rewards[i] = 10.0
                    self.demand[loc] -= 1 # Consume demand
                    picked_up_map[loc] += 1 # Record pickup for viz

                    # Teleport to random destination
                    self.taxi_locs[i] = np.random.randint(0, self.num_nodes)
                else:
                    rewards[i] = -1.0 # Failed pickup penalty
                    # Stays in place (wasted step)

            else: # MOVE
                new_r, new_c = r, c
                if action == 1: new_r -= 1
                elif action == 2: new_r += 1
                elif action == 3: new_c -= 1
                elif action == 4: new_c += 1

                # Boundary checks
                if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
                    self.taxi_locs[i] = self._get_node(new_r, new_c)
                else:
                    rewards[i] = -0.5 # Wall penalty

        # Generate NEW demand (accumulate)
        new_demand = self._generate_demand()
        self.demand += new_demand
        self.demand = np.clip(self.demand, 0, 50) # Cap higher for 100 cars

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        next_obs = self._get_observation_batch()

        # Return info for visualization
        info = {
            'generated_demand': new_demand,
            'picked_up_map': picked_up_map
        }

        return next_obs, rewards, done, info

# ---------------------------------------------------------
# 2. PPO Classes (CUDA Enabled)
# ---------------------------------------------------------

class Policy(object):
    def __init__(self, obssize, actsize, lr, device):
        self.device = device
        self.actsize = actsize

        # Increased network size for 100 cars complexity
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, actsize)
        ).to(self.device) # Move model to GPU

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_prob(self, states):
        # states shape: [batch_size, obssize]
        # Move input to GPU
        states = torch.FloatTensor(states).to(self.device)
        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        # Move output back to CPU for numpy interaction
        return prob.cpu().data.numpy()

    def _to_one_hot(self, y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype, device=self.device)
        return zeros.scatter(scatter_dim, y_tensor, 1)

    def train(self, states, actions, Qs):
        # Move inputs to GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        Qs = torch.FloatTensor(Qs).to(self.device)

        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)

        action_onehot = self._to_one_hot(actions, self.actsize)
        prob_selected = torch.sum(prob * action_onehot, axis=-1)
        prob_selected += 1e-8

        log_prob_selected = torch.log(prob_selected)
        loss = -torch.mean(Qs * log_prob_selected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()


class ValueFunction(object):
    def __init__(self, obssize, lr, device):
        self.device = device
        # Increased network size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(self.device) # Move model to GPU

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_values(self, states):
        # Move to GPU
        states = torch.FloatTensor(states).to(self.device)
        return self.model(states).cpu().data.numpy()

    def train(self, states, targets):
        # Move to GPU
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)

        v_preds = self.model(states)
        loss = torch.nn.functional.mse_loss(v_preds, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().data.numpy()

def discounted_rewards(r, gamma):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_sum = 0
    for i in reversed(range(0, len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)

# ---------------------------------------------------------
# 3. Training & Evaluation (Multi-Agent Loop)
# ---------------------------------------------------------

def evaluate(policy, env, episodes):
    """Evaluates the fleet performance."""
    total_score = 0
    for episode in range(episodes):
        obs_batch = env.reset() # (100, 21)
        done = False
        episode_reward = 0

        while not done:
            # Predict for all 100 cars at once
            p = policy.compute_prob(obs_batch) # (100, 5)

            # Sample actions for each car
            actions = []
            for i in range(env.num_taxis):
                # Normalize just in case
                p_i = p[i] / np.sum(p[i])
                act = np.random.choice(np.arange(env.action_space.n), p=p_i)
                actions.append(act)

            obs_batch, rewards, done, _ = env.step(actions)
            episode_reward += np.sum(rewards)

        total_score += episode_reward

    return total_score / episodes

def run_baseline(env, episodes=50):
    """
    Baseline Strategy:
    - If demand exists at current location: Pick up (Action 0)
    - Else: Move Randomly (Action 1-4)
    """
    print(f"\nRunning Baseline (Random Move + Greedy Pickup) for {episodes} episodes...")
    total_score = 0

    for episode in range(episodes):
        obs_batch = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = []
            # We use a copy of demand to simulate greedy allocation within the step
            current_demand_snapshot = env.demand.copy()

            for i in range(env.num_taxis):
                loc = env.taxi_locs[i]

                # Check demand
                if current_demand_snapshot[loc] > 0:
                    action = 0 # Pickup
                    current_demand_snapshot[loc] -= 1
                else:
                    action = np.random.randint(1, 5)

                actions.append(action)

            obs_batch, rewards, done, _ = env.step(actions)
            episode_reward += np.sum(rewards)

        total_score += episode_reward

    avg_score = total_score / episodes
    print(f"Baseline Average Score: {avg_score:.2f}")
    return avg_score

def visualize_last_day(policy, env):
    """
    Runs one episode and visualizes:
    1. Taxi presence heatmap
    2. Demand generated heatmap
    3. Demand matched heatmap
    4. Profit growth curve
    """
    print("\nVisualizing last day (1 Episode)...")

    # Reset Environment
    obs_batch = env.reset()
    done = False

    # Aggregation Statistics
    taxi_visits = np.zeros(env.num_nodes)
    demand_generated = np.zeros(env.num_nodes)
    demand_matched = np.zeros(env.num_nodes)

    cumulative_profit = []
    current_profit = 0

    steps = 0

    while not done:
        # Track initial taxi locations for this step
        for loc in env.taxi_locs:
            taxi_visits[loc] += 1

        # Get actions
        p = policy.compute_prob(obs_batch)
        actions = []
        for i in range(env.num_taxis):
            p_i = p[i] / np.sum(p[i])
            act = np.random.choice(np.arange(env.action_space.n), p=p_i)
            actions.append(act)

        # Step
        obs_batch, rewards, done, info = env.step(actions)

        # Accumulate Stats
        step_reward = np.sum(rewards)
        current_profit += step_reward
        cumulative_profit.append(current_profit)

        demand_generated += info['generated_demand']
        demand_matched += info['picked_up_map']

        steps += 1

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reshape arrays to grid
    visit_grid = taxi_visits.reshape(env.rows, env.cols)
    gen_grid = demand_generated.reshape(env.rows, env.cols)
    match_grid = demand_matched.reshape(env.rows, env.cols)

    # Helper for heatmap
    def plot_heatmap(ax, data, title, cmap='Blues'):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(np.arange(env.cols))
        ax.set_yticks(np.arange(env.rows))

        # Loop over data dimensions and create text annotations.
        for i in range(env.rows):
            for j in range(env.cols):
                text = ax.text(j, i, int(data[i, j]),
                               ha="center", va="center", color="black" if data[i, j] < data.max()/2 else "white")
        plt.colorbar(im, ax=ax)

    # 1. Taxi Visits
    plot_heatmap(axes[0, 0], visit_grid, "Total Taxi Visits (Presence)", cmap='Purples')

    # 2. Demand Generated
    plot_heatmap(axes[0, 1], gen_grid, "Total Demand Generated", cmap='Reds')

    # 3. Demand Matched
    plot_heatmap(axes[1, 0], match_grid, "Total Demand Matched (Pickups)", cmap='Greens')

    # 4. Profit Curve
    axes[1, 1].plot(cumulative_profit, color='blue', linewidth=2)
    axes[1, 1].set_title("Cumulative Profit Over Day")
    axes[1, 1].set_xlabel("Time Step (20 mins)")
    axes[1, 1].set_ylabel("Profit ($)")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def main():

    # Hyperparameters
    alpha = 1e-3
    beta = 1e-3
    numtrajs = 5  # Number of episodes per update
    iterations = 500  # Increased to 500
    gamma = 0.997

    num_taxis = 10 # Increased to 100

    # Initialize Env
    env = CityTaxiEnv(num_taxis=num_taxis)
    obssize = env.observation_space.shape[0]
    actsize = env.action_space.n

    print(f"Observation Size: {obssize}, Action Size: {actsize}, Num Taxis: {num_taxis}")

    # --- Run Baseline Comparison First ---
    baseline_score = run_baseline(env, episodes=50)

    # Initialize Networks (Shared Policy) with Device
    actor = Policy(obssize, actsize, alpha, device)
    baseline = ValueFunction(obssize, beta, device)

    print("\nStarting PPO Training...")

    for ite in range(iterations):
        # Master buffers for the update step
        OBS_BUFFER = []
        ACTS_BUFFER = []
        VAL_BUFFER = []

        total_batch_reward = 0

        # Run episodes
        for num in range(numtrajs):
            # Per-car storage for this episode
            car_obss = [[] for _ in range(num_taxis)]
            car_acts = [[] for _ in range(num_taxis)]
            car_rews = [[] for _ in range(num_taxis)]

            obs_batch = env.reset() # (100, 21)
            done = False

            while not done:
                # 1. Store current state for each car
                for i in range(num_taxis):
                    car_obss[i].append(obs_batch[i])

                # 2. Get probabilities for all cars (Batch Inference)
                probs = actor.compute_prob(obs_batch) # Shape (100, 5)

                # 3. Sample actions
                actions = []
                for i in range(num_taxis):
                    a = np.random.choice(actsize, p=probs[i].flatten())
                    actions.append(a)
                    car_acts[i].append(a)

                # 4. Step Environment
                next_obs_batch, rewards, done, _ = env.step(actions)

                # 5. Store rewards
                for i in range(num_taxis):
                    car_rews[i].append(rewards[i])

                obs_batch = next_obs_batch

            # End of Episode: Process Rewards per Car
            ep_total = 0
            for i in range(num_taxis):
                ep_total += np.sum(car_rews[i])

                # Calculate discounted returns for this car's trajectory
                dis_r = discounted_rewards(car_rews[i], gamma)

                # Add to master buffers
                OBS_BUFFER.extend(car_obss[i])
                ACTS_BUFFER.extend(car_acts[i])
                VAL_BUFFER.extend(dis_r)

            total_batch_reward += ep_total

        # --- PPO UPDATE ---

        # Logging
        avg_ep_reward = total_batch_reward / numtrajs

        if ite % 10 == 0:
            print(f"Iteration {ite}, Avg Fleet Reward: {avg_ep_reward:.2f}")

        # Convert buffers to arrays
        obs_train = np.array(OBS_BUFFER)
        val_train = np.array(VAL_BUFFER)
        acts_train = np.array(ACTS_BUFFER)

        # 1. Train Baseline
        baseline_loss = baseline.train(obs_train, val_train)

        # 2. Train Actor
        v_preds = baseline.compute_values(obs_train).flatten()
        advantages = val_train - v_preds

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / advantages.std()

        actor_loss = actor.train(obs_train, acts_train, advantages)


    print("\nTraining Complete. Evaluating PPO...")
    test_score = evaluate(actor, env, episodes=50)
    print(f"Final PPO Test Score: {test_score:.2f}")
    print(f"Baseline Score: {baseline_score:.2f}")

    if test_score > baseline_score:
        print("Result: PPO outperformed the Baseline!")
    else:
        print("Result: PPO failed to beat the Baseline.")

    # --- VISUALIZATION ---
    visualize_last_day(actor, env)

if __name__ == "__main__":
    main()