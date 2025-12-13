import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------
# 0. Device Setup
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 1. Custom City Environment (Dynamic Spinning Bonus)
# ---------------------------------------------------------


class CityTaxiEnv(gym.Env):
    """
    A 4x5 Grid City Environment for Taxi Service (100 Cars).
    Dynamic Features: Ultra-High Bonus Node spins clockwise every 4 steps.
    """

    def __init__(self, num_taxis=100):
        super().__init__()

        self.rows = 4
        self.cols = 5
        self.num_nodes = self.rows * self.cols
        self.num_taxis = num_taxis

        # Clockwise perimeter path
        self.perimeter_path = [0, 1, 2, 3, 4, 9, 14, 19, 18, 17, 16, 15, 10, 5]  # Top  # Right  # Bottom  # Left
        self.special_path_idx = 0
        self.steps_per_move = 4

        self.action_space = spaces.Discrete(5)
        # Observation: My Pos(1) + Demand(20) + Next Ultra Pos(1)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1 + self.num_nodes + 1,), dtype=np.float32)

        self.max_steps = 72
        self.current_step = 0
        self.taxi_locs = np.zeros(self.num_taxis, dtype=int)
        self.demand = np.zeros(self.num_nodes)

    def _get_coords(self, node):
        return node // self.cols, node % self.cols

    def _get_node(self, r, c):
        return r * self.cols + c

    def _get_neighbors(self, node):
        r, c = self._get_coords(node)
        neighbors = []
        if r > 0:
            neighbors.append(self._get_node(r - 1, c))
        if r < self.rows - 1:
            neighbors.append(self._get_node(r + 1, c))
        if c > 0:
            neighbors.append(self._get_node(r, c - 1))
        if c < self.cols - 1:
            neighbors.append(self._get_node(r, c + 1))
        return neighbors

    def _generate_demand(self):
        demand = np.zeros(self.num_nodes)
        ultra_node = self.perimeter_path[self.special_path_idx]
        high_nodes = self._get_neighbors(ultra_node)

        for i in range(self.num_nodes):
            if i == ultra_node:
                demand[i] = np.random.poisson(lam=3.0)
            elif i in high_nodes:
                demand[i] = np.random.poisson(lam=1.5)
            else:
                demand[i] = np.random.poisson(lam=0.2)
        return demand

    def _get_next_special_node(self):
        future_step = self.current_step + 1
        moves_count = future_step // self.steps_per_move
        next_idx = moves_count % len(self.perimeter_path)
        return self.perimeter_path[next_idx]

    def _get_observation_batch(self):
        locs = self.taxi_locs.reshape(-1, 1)
        demands = np.tile(self.demand, (self.num_taxis, 1))
        next_special = self._get_next_special_node()
        next_specials = np.full((self.num_taxis, 1), next_special)
        return np.concatenate((locs, demands, next_specials), axis=1).astype(np.float32)

    def reset(self):
        self.current_step = 0
        self.special_path_idx = 0
        self.taxi_locs = np.random.randint(0, self.num_nodes, size=self.num_taxis)
        self.demand = self._generate_demand()
        return self._get_observation_batch()

    def step(self, actions):
        rewards = np.zeros(self.num_taxis)
        done = False

        current_ultra_node = self.perimeter_path[self.special_path_idx]
        current_high_nodes = self._get_neighbors(current_ultra_node)

        for i in range(self.num_taxis):
            action = actions[i]
            loc = self.taxi_locs[i]
            r, c = self._get_coords(loc)

            if action == 0:  # PICK UP
                if self.demand[loc] > 0:
                    if loc == current_ultra_node:
                        rewards[i] = 100.0
                    elif loc in current_high_nodes:
                        rewards[i] = 50.0
                    else:
                        rewards[i] = 10.0
                    self.demand[loc] -= 1

                    # Random dropoff
                    dr, dc = np.random.choice([-1, 0, 1], 2)
                    self.taxi_locs[i] = self._get_node(
                        np.clip(r + dr, 0, self.rows - 1), np.clip(c + dc, 0, self.cols - 1)
                    )
                else:
                    rewards[i] = -1.0

            else:  # MOVE
                nr, nc = r, c
                if action == 1:
                    nr -= 1
                elif action == 2:
                    nr += 1
                elif action == 3:
                    nc -= 1
                elif action == 4:
                    nc += 1

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    self.taxi_locs[i] = self._get_node(nr, nc)
                else:
                    rewards[i] = -0.5

        self.current_step += 1
        if self.current_step % self.steps_per_move == 0:
            self.special_path_idx = (self.special_path_idx + 1) % len(self.perimeter_path)
        if self.current_step >= self.max_steps:
            done = True

        new_demand = self._generate_demand()
        self.demand = new_demand
        self.demand = np.clip(self.demand, 0, 50)

        return self._get_observation_batch(), rewards, done, {}


# ---------------------------------------------------------
# 2. PPO Classes
# ---------------------------------------------------------


class Policy:
    def __init__(self, obssize, actsize, lr, device, clip_eps=0.2, use_clip=True):
        self.device = device
        self.actsize = actsize
        self.clip_eps = clip_eps
        self.use_clip = use_clip
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, actsize),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_prob(self, states):
        states = torch.FloatTensor(states).to(self.device)
        logits = self.model(states)
        return torch.nn.functional.softmax(logits, dim=-1).cpu().data.numpy()

    def train(self, states, actions, Qs, old_prob_selected=None):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        Qs = torch.FloatTensor(Qs).to(self.device)
        logits = self.model(states)
        prob = torch.nn.functional.softmax(logits, dim=-1)
        prob_selected = prob.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8
        if self.use_clip:
            if old_prob_selected is None:
                raise ValueError("old_prob_selected required for PPO-Clip training.")
            old_prob_selected = torch.FloatTensor(old_prob_selected).to(self.device)
            ratio = prob_selected / (old_prob_selected + 1e-8)
            unclipped = ratio * Qs
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * Qs
            loss = -torch.mean(torch.min(unclipped, clipped))
        else:
            loss = -torch.mean(Qs * torch.log(prob_selected + 1e-8))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().data.numpy()


class ValueFunction:
    def __init__(self, obssize, lr, device):
        self.device = device
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obssize, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_values(self, states):
        return self.model(torch.FloatTensor(states).to(self.device)).cpu().data.numpy()

    def train(self, states, targets):
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        loss = torch.nn.functional.mse_loss(self.model(states), targets.unsqueeze(1))
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
# 3. Training & NEW Dynamic Visualization
# ---------------------------------------------------------


def evaluate(policy, env, episodes):
    total_score = 0
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            p = policy.compute_prob(obs)
            actions = [np.random.choice(env.action_space.n, p=p[i] / np.sum(p[i])) for i in range(env.num_taxis)]
            obs, rewards, done, _ = env.step(actions)
            ep_reward += np.sum(rewards)
        total_score += ep_reward
    return total_score / episodes


def run_baseline(env, episodes=50):
    print(f"\nRunning Baseline...")
    total_score = 0
    for _ in range(episodes):
        env.reset()
        done = False
        ep_reward = 0
        while not done:
            actions = [np.random.randint(0, 5) for _ in range(env.num_taxis)]
            _, rewards, done, _ = env.step(actions)
            ep_reward += np.sum(rewards)
        total_score += ep_reward
    return total_score / episodes


# --- NEW: Capture data step-by-step for animation ---
def capture_last_episode_data(policy, env):
    print("\nCapturing data for final episode animation...")
    obs = env.reset()
    done = False

    history = []

    while not done:
        # Record state at the beginning of the step
        current_ultra = env.perimeter_path[env.special_path_idx]
        history.append({"step": env.current_step, "taxi_locs": env.taxi_locs.copy(), "ultra_node": current_ultra})

        p = policy.compute_prob(obs)
        actions = [np.random.choice(env.action_space.n, p=p[i] / np.sum(p[i])) for i in range(env.num_taxis)]
        obs, _, done, _ = env.step(actions)

    # Append final state
    history.append(
        {
            "step": env.current_step,
            "taxi_locs": env.taxi_locs.copy(),
            "ultra_node": env.perimeter_path[env.special_path_idx],
        }
    )
    return history


# --- NEW: Shared training helper for different policy objectives ---
def train_agent(env, actor, baseline, iterations, numtrajs, gamma, label="Policy", log_interval=20):
    print(f"\nTraining {label}...")
    num_taxis = env.num_taxis
    training_curve = []

    for ite in range(iterations):
        OBS, ACTS, VALS = [], [], []
        OLD_PROBS = [] if actor.use_clip else None
        total_batch_reward = 0

        for _ in range(numtrajs):
            car_obss = [[] for _ in range(num_taxis)]
            car_acts = [[] for _ in range(num_taxis)]
            car_rews = [[] for _ in range(num_taxis)]
            car_old_probs = [[] for _ in range(num_taxis)] if actor.use_clip else None
            obs = env.reset()
            done = False

            while not done:
                for i in range(num_taxis):
                    car_obss[i].append(obs[i])
                probs = actor.compute_prob(obs)
                actions = [
                    np.random.choice(env.action_space.n, p=probs[i] / np.sum(probs[i])) for i in range(num_taxis)
                ]
                for i in range(num_taxis):
                    car_acts[i].append(actions[i])
                    if actor.use_clip:
                        car_old_probs[i].append(probs[i][actions[i]])

                obs, rewards, done, _ = env.step(actions)
                for i in range(num_taxis):
                    car_rews[i].append(rewards[i])

            ep_total = 0
            for i in range(num_taxis):
                ep_total += np.sum(car_rews[i])
                OBS.extend(car_obss[i])
                ACTS.extend(car_acts[i])
                VALS.extend(discounted_rewards(car_rews[i], gamma))
                if actor.use_clip:
                    OLD_PROBS.extend(car_old_probs[i])
            total_batch_reward += ep_total

        avg_reward = total_batch_reward / numtrajs
        training_curve.append(avg_reward)
        if ite % log_interval == 0:
            print(f"[{label}] Iteration {ite}, Avg Fleet Reward: {avg_reward:.2f}")

        obs_train = np.array(OBS)
        val_train = np.array(VALS)
        acts_train = np.array(ACTS)
        old_prob_train = np.array(OLD_PROBS) if actor.use_clip else None
        baseline.train(obs_train, val_train)
        v_preds = baseline.compute_values(obs_train).flatten()
        adv = val_train - v_preds
        if adv.std() > 1e-8:
            adv = (adv - adv.mean()) / adv.std()
        actor.train(obs_train, acts_train, adv, old_prob_train)

    return training_curve


# --- NEW: Comparison plotting helper ---
def plot_policy_comparison(training_curves, eval_scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training curves
    for label, curve in training_curves.items():
        axes[0].plot(curve, label=label, linewidth=2)
    axes[0].set_title("Average Fleet Reward per Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Avg Fleet Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Evaluation bars
    labels = list(eval_scores.keys())
    values = [eval_scores[l] for l in labels]
    axes[1].bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    for idx, value in enumerate(values):
        axes[1].text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Evaluation Reward (20 episodes)")
    axes[1].set_ylabel("Average Reward")
    axes[1].grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.show()


# --- NEW: Animation Function ---
def animate_dynamics(env, history_data):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Setup Static Grid lines
    for r in range(env.rows + 1):
        ax.plot([-0.5, env.cols - 0.5], [-r, -r], color="#DDDDDD", linewidth=1, zorder=0)
    for c in range(env.cols + 1):
        ax.plot([c - 0.5, c - 0.5], [0.5, -env.rows + 0.5], color="#DDDDDD", linewidth=1, zorder=0)

    # 2. Initialize Node Circles and Text objects
    node_circles = []
    node_texts = []
    for r in range(env.rows):
        for c in range(env.cols):
            # Circle Patch
            circle = patches.Circle((c, -r), 0.4, color="blue", alpha=0.7, zorder=1)
            ax.add_patch(circle)
            node_circles.append(circle)
            # Text Object for Car Count
            text = ax.text(c, -r, "0", ha="center", va="center", color="white", fontweight="bold", zorder=2)
            node_texts.append(text)

    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-env.rows + 0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    title_text = ax.set_title("Step: 0", fontsize=14)

    # 3. Update function for animation
    def update(frame_idx):
        data = history_data[frame_idx]
        step_num = data["step"]
        taxi_locs = data["taxi_locs"]
        ultra_node = data["ultra_node"]
        high_nodes = env._get_neighbors(ultra_node)

        # Count cars at each node
        car_counts = np.bincount(taxi_locs, minlength=env.num_nodes)

        # Update visuals for each node
        for i in range(env.num_nodes):
            # Determine Color based on Bonus Status
            if i == ultra_node:
                color = "#8B0000"  # Dark Red (Ultra)
            elif i in high_nodes:
                color = "#FF4500"  # OrangeRed (High)
            else:
                color = "#4682B4"  # SteelBlue (Standard)

            node_circles[i].set_color(color)

            # Update Car Count Text
            node_texts[i].set_text(str(car_counts[i]))

        title_text.set_text(f"Step: {step_num} | Ultra-High Node spins clockwise")
        return node_circles + node_texts + [title_text]

    # Create Animation
    ani = FuncAnimation(fig, update, frames=len(history_data), interval=200, blit=False)

    print("Displaying animation...")
    plt.show()


def main():
    # Hyperparameters
    alpha = 1e-3
    beta = 1e-3
    numtrajs = 4
    iterations = 600
    gamma = 0.99
    num_taxis = 100

    # Shared observation/action sizes
    tmp_env = CityTaxiEnv(num_taxis=num_taxis)
    obssize = tmp_env.observation_space.shape[0]
    actsize = tmp_env.action_space.n

    baseline_score = run_baseline(CityTaxiEnv(num_taxis=num_taxis), episodes=20)

    # Train PPO-Clip
    clip_actor = Policy(obssize, actsize, alpha, device, clip_eps=0.2, use_clip=True)
    clip_baseline = ValueFunction(obssize, beta, device)
    clip_curve = train_agent(
        CityTaxiEnv(num_taxis=num_taxis), clip_actor, clip_baseline, iterations, numtrajs, gamma, label="PPO-Clip"
    )

    # Train vanilla policy gradient
    vanilla_actor = Policy(obssize, actsize, alpha, device, use_clip=False)
    vanilla_baseline = ValueFunction(obssize, beta, device)
    vanilla_curve = train_agent(
        CityTaxiEnv(num_taxis=num_taxis),
        vanilla_actor,
        vanilla_baseline,
        iterations,
        numtrajs,
        gamma,
        label="Vanilla PG",
    )

    clip_score = evaluate(clip_actor, CityTaxiEnv(num_taxis=num_taxis), episodes=20)
    vanilla_score = evaluate(vanilla_actor, CityTaxiEnv(num_taxis=num_taxis), episodes=20)

    print("-" * 30)
    print(f"Baseline Score:       {baseline_score:.2f}")
    print(f"PPO-Clip Test Score:  {clip_score:.2f}")
    print(f"Vanilla PG Score:     {vanilla_score:.2f}")
    print("-" * 30)

    plot_policy_comparison(
        {"PPO-Clip": clip_curve, "Vanilla PG": vanilla_curve}, {"PPO-Clip": clip_score, "Vanilla PG": vanilla_score}
    )

    # Optional: animate PPO-Clip behavior for qualitative view
    history_data = capture_last_episode_data(clip_actor, CityTaxiEnv(num_taxis=num_taxis))
    animate_dynamics(CityTaxiEnv(num_taxis=num_taxis), history_data)


if __name__ == "__main__":
    main()
