import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from dotenv import load_dotenv

from vlm_interactions import get_vlm_response
load_dotenv()

from entropy_tracker import EntropyTracker

# Parameters (configurable)
GRID_SIZE = 20
NUM_AGENTS = 3
START_POS = (0, 0)  # (x, y), ensured to be passable
START_TIMES = [0, 0, 0]  # Start time for each agent
BUDGETS = [20, 20, 20]  # Budget for each agent
MAX_TIME = 50  # Cap at 50 steps to limit VLM calls (50 calls max)
MOVE_COST = 1  # Cost per move
OUTPUT_DIR = "simulation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PERLIN_SCALE = 5.0  # Controls feature size
REWARD_SEED = 42
OBSTACLE_SEED = 43
OBSTACLE_THRESHOLD = 0.7
OBSERVATION_NOISE = 0.01  # Noise in observations for GP

# Directions mapping (for VLM recommendations)
DIRECTIONS = {
    "stay": (0, 0),
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0)
}

VLM_MODE = "conversation"

# Perlin noise functions
def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, x):
    return a + x * (b - a)

def gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin(x, y, seed=0):
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    u = fade(xf)
    v = fade(yf)
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

# Add this function after your Perlin noise functions (around line 70)
def create_structured_uncertainty_map(grid_size, obstacle_mask):
    """
    Create a variance map with structured uncertainty zones that provide
    clear coordination targets for the VLM
    """
    variance_map = np.zeros((grid_size, grid_size))
    
    # Create distinct uncertainty zones
    for i in range(grid_size):
        for j in range(grid_size):
            if obstacle_mask[j, i]:
                variance_map[j, i] = 0.0  # Obstacles have no uncertainty for now
            else:
                # Zone 1: High uncertainty in top-right quadrant
                if i > grid_size//2 and j > grid_size//2:
                    variance_map[j, i] = 0.9
                # Zone 2: Medium uncertainty in bottom-left quadrant  
                elif i < grid_size//2 and j < grid_size//2:
                    variance_map[j, i] = 0.6
                # Zone 3: Low uncertainty in center (already "known")
                elif grid_size//4 < i < 3*grid_size//4 and grid_size//4 < j < 3*grid_size//4:
                    variance_map[j, i] = 0.2
                # Default: medium uncertainty
                else:
                    variance_map[j, i] = 0.5
    
    return variance_map

# Generate reward map (ground truth, not sent to VLM)
lin_x = np.linspace(0, PERLIN_SCALE, GRID_SIZE, endpoint=False)
lin_y = np.linspace(0, PERLIN_SCALE, GRID_SIZE, endpoint=False)
x, y = np.meshgrid(lin_x, lin_y)
reward_map = perlin(x, y, seed=REWARD_SEED)
reward_map = (reward_map + 1) / 2  # Normalize to [0,1]
reward_map = np.clip(reward_map, 0, 1)

# Generate obstacle map
obstacle_map = perlin(x, y, seed=OBSTACLE_SEED)
obstacle_map = (obstacle_map + 1) / 2
obstacle_mask = obstacle_map > OBSTACLE_THRESHOLD
reward_map[obstacle_mask] = 0

# Ensure start position is passable
if obstacle_mask[START_POS[1], START_POS[0]]:
    for r in range(1, GRID_SIZE):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = START_POS[0] + dx, START_POS[1] + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and not obstacle_mask[ny, nx]:
                    START_POS = (nx, ny)
                    break
            else:
                continue
            break
        else:
            continue
        break
    else:
        raise ValueError("No passable cells found near start position")

# Gaussian Process setup
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=OBSERVATION_NOISE**2, n_restarts_optimizer=10)

# Observations list: list of (pos_x, pos_y, observed_reward)
observations = []

# Initialize belief maps (from GP predictions)
belief_mean = np.zeros((GRID_SIZE, GRID_SIZE))  # Initial mean=0 (or prior)
initial_uncertainty = create_structured_uncertainty_map(GRID_SIZE, obstacle_mask)
belief_variance = initial_uncertainty.copy()
belief_mean[obstacle_mask] = -2.0
belief_variance[obstacle_mask] = 0.0

# Grid points for prediction
grid_points = np.array([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if not obstacle_mask[j, i]])

# Function to update GP and belief maps
def update_gp_and_belief():
    global belief_mean, belief_variance
    if observations:
        X_obs = np.array([[obs[0], obs[1]] for obs in observations])
        y_obs = np.array([obs[2] for obs in observations])
        gp.fit(X_obs, y_obs)
        mean_pred, std_pred = gp.predict(grid_points, return_std=True)
        for idx, (i, j) in enumerate(grid_points):
            belief_mean[j, i] = mean_pred[idx]
            # Only update variance for observed areas, preserve structured uncertainty elsewhere
            if (i, j) in [(obs[0], obs[1]) for obs in observations]:
                belief_variance[j, i] = std_pred[idx]**2
            else:
                # Keep the original structured uncertainty for unobserved areas
                belief_variance[j, i] = initial_uncertainty[j, i]
    else:
        belief_mean[~obstacle_mask] = 0.0
        # Use structured uncertainty instead of uniform
        belief_variance[~obstacle_mask] = initial_uncertainty[~obstacle_mask]

# Function to observe at position (add to observations)
def observe(pos):
    x, y = pos
    if not obstacle_mask[y, x]:
        obs = reward_map[y, x] + np.random.normal(0, OBSERVATION_NOISE)
        observations.append((x, y, obs))
    # Observe neighbors
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and not obstacle_mask[ny, nx]:
                obs = reward_map[ny, nx] + np.random.normal(0, OBSERVATION_NOISE)
                if (nx, ny, obs) not in observations:  # Avoid duplicates
                    observations.append((nx, ny, obs))

# Initialize agents
agents = [
    {
        "id": i,
        "pos": list(START_POS),
        "budget": BUDGETS[i],
        "start_time": START_TIMES[i],
        "active": False
    }
    for i in range(NUM_AGENTS)
]

# Initialize entropy tracker
entropy_tracker = EntropyTracker()

# Function to compute entropy (approximated via variance)
def compute_entropy():
    entropy = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Adaptive minimum based on observation noise
    min_variance = max(1e-6, OBSERVATION_NOISE**2)
    adjusted_variance = np.maximum(belief_variance[~obstacle_mask], min_variance)
    entropy[~obstacle_mask] = 0.5 * np.log(2 * np.pi * np.e * adjusted_variance)
    
    return entropy

# Set up interactive plotting
plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle("Simulation Viewer")

# Ground truth plot (left)
im_gt = axs[0].imshow(reward_map, cmap='viridis', origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
axs[0].imshow(obstacle_mask, cmap='binary', alpha=0.5, origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
axs[0].set_title("Ground Truth Reward Map (Black: Impassable)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].grid(True)
scatter_gt = axs[0].scatter([], [], c='r', s=100, label='Agents')

# Belief mean plot (middle)
im_belief = axs[1].imshow(belief_mean, cmap='viridis', origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE], vmin=-2, vmax=1)
axs[1].imshow(obstacle_mask, cmap='binary', alpha=0.5, origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
axs[1].set_title("Belief Mean (Black: Impassable)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].grid(True)
scatter_belief = axs[1].scatter([], [], c='r', s=100, label='Agents')

# Belief variance plot (right)
im_variance = axs[2].imshow(belief_variance, cmap='hot', origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE], vmin=0, vmax=1.0)
axs[2].imshow(obstacle_mask, cmap='binary', alpha=0.5, origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE])
axs[2].set_title("Belief Variance (Uncertainty)")
axs[2].set_xlabel("X")
axs[2].set_ylabel("Y")
axs[2].grid(True)
scatter_variance = axs[2].scatter([], [], c='r', s=100, label='Agents')

# Colorbars
fig.colorbar(im_gt, ax=axs[0], orientation='vertical', fraction=0.02, pad=0.04)
fig.colorbar(im_belief, ax=axs[1], orientation='vertical', fraction=0.02, pad=0.04)
fig.colorbar(im_variance, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.04)

# Function to update plots
def update_plots(t):
    active_agents = [a for a in agents if a["active"]]
    if active_agents:
        positions = np.array([a["pos"] for a in active_agents])
        scatter_gt.set_offsets(positions)
        scatter_belief.set_offsets(positions)
        scatter_variance.set_offsets(positions)
        for txt in axs[0].texts:
            txt.remove()
        for txt in axs[1].texts:
            txt.remove()
        for txt in axs[2].texts:
            txt.remove()
        for a in active_agents:
            axs[0].text(a["pos"][0] + 0.1, a["pos"][1] + 0.1, f'A{a["id"]}', color='white')
            axs[1].text(a["pos"][0] + 0.1, a["pos"][1] + 0.1, f'A{a["id"]}', color='white')
            axs[2].text(a["pos"][0] + 0.1, a["pos"][1] + 0.1, f'A{a["id"]}', color='white')
    else:
        scatter_gt.set_offsets(np.empty((0, 2)))
        scatter_belief.set_offsets(np.empty((0, 2)))
        scatter_variance.set_offsets(np.empty((0, 2)))

    im_belief.set_array(belief_mean)
    im_variance.set_array(belief_variance)
    fig.suptitle(f"Simulation Viewer - Time Step: {t}")
    fig.canvas.draw()
    fig.canvas.flush_events()

def generate_vlm_image(t, belief_variance, agents, obstacle_mask):
    """
    Generate VLM image showing ONLY the belief variance (uncertainty) map
    This is what the VLM actually needs to make orchestration decisions
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a copy of variance for visualization (to avoid modifying the original)
    viz_variance = belief_variance.copy()
    
    # Set obstacle areas to a special value for visualization
    viz_variance[obstacle_mask] = -1
    
    # Show belief variance (uncertainty) with fixed range
    im = ax.imshow(viz_variance, cmap='hot', origin='lower', 
                   extent=[0, GRID_SIZE, 0, GRID_SIZE], vmin=0, vmax=1)
    
    # Overlay obstacles with a different approach
    # Create a binary mask for obstacles and overlay it
    obstacle_overlay = np.ma.masked_where(~obstacle_mask, np.ones_like(obstacle_mask))
    ax.imshow(obstacle_overlay, cmap='gray', alpha=0.8, origin='lower', 
              extent=[0, GRID_SIZE, 0, GRID_SIZE])
    
    # Add grid for better spatial understanding
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 5))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 5))
    
    # Plot agents with clear visibility
    for agent in agents:
        if agent["active"]:
            # Large, clear agent markers
            ax.plot(agent["pos"][0] + 0.5, agent["pos"][1] + 0.5, 'o', 
                   markersize=20, color='red', markeredgecolor='white', markeredgewidth=3)
            # Clear agent labels
            ax.text(agent["pos"][0] + 0.5, agent["pos"][1] + 0.5, f'A{agent["id"]}', 
                   color='white', fontsize=14, fontweight='bold', ha='center', va='center')
    
    # Clear title explaining what this map shows
    ax.set_title(f"Belief Variance (Uncertainty) - Step {t}", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # Add colorbar with clear labels
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Uncertainty Level', fontweight='bold')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', 'Low-Med', 'Medium', 'Med-High', 'High'])
    
    plt.tight_layout()
    return fig

# Simulation loop
vlm_call_count = 0
for t in range(MAX_TIME):
    print(f"Step {t}")
    # Activate agents
    for agent in agents:
        if t >= agent["start_time"]:
            agent["active"] = True

    # Update GP and belief
    update_gp_and_belief()

    # Update plots
    update_plots(t)

    # Compute entropy and high-entropy cells
    entropy_map = compute_entropy()
    total_entropy = np.sum(entropy_map[~obstacle_mask])
    high_entropy_cells = [
        [int(i), int(j)]
        for i, j in zip(*np.where((entropy_map > np.percentile(entropy_map[~obstacle_mask], 75)) & (~obstacle_mask)))
    ]
    
    # Update entropy tracking
    entropy_tracker.update(total_entropy, t, len(observations))
    
    # Print current entropy status
    if len(entropy_tracker.entropy_history) > 1:
        current_reduction = entropy_tracker.entropy_reduction_history[-1]
        print(f"Step {t}: Total entropy = {total_entropy:.4f}, Reduction = {current_reduction:.4f}")
    else:
        print(f"Step {t}: Total entropy = {total_entropy:.4f}")

    # Generate visualization of belief mean (for VLM input)
    fig = generate_vlm_image(t, belief_variance, agents, obstacle_mask)
    image_path = os.path.join(OUTPUT_DIR, f"belief_state_{t}.png")
    plt.savefig(image_path)
    plt.close(fig)

    # Generate JSON state (for VLM input)
    state = {
        "time": t,
        "grid_size": GRID_SIZE,
        "belief_mean": belief_mean.tolist(),
        "belief_variance": belief_variance.tolist(),
        "total_entropy": float(total_entropy),
        "high_entropy_cells": high_entropy_cells,
        "agents": [
            {
                "id": agent["id"],
                "pos": agent["pos"],
                "budget": agent["budget"],
                "active": agent["active"]
            }
            for agent in agents
        ]
    }

    # Automate VLM call (cap at 10)
    if vlm_call_count >= MAX_TIME:
        print("VLM call limit reached. Ending simulation.")
        break
    moves = get_vlm_response(VLM_MODE, NUM_AGENTS, GRID_SIZE, state, image_path)
    vlm_call_count += 1
    print(f"VLM moves at step {t}: {moves}")

    # Apply moves
    for agent in agents:
        if not agent["active"] or agent["budget"] < MOVE_COST:
            continue
        agent_id_str = str(agent["id"])
        if agent_id_str in moves:
            direction = moves[agent_id_str]
            if direction in DIRECTIONS:
                dx, dy = DIRECTIONS[direction]
                new_x = min(max(0, agent["pos"][0] + dx), GRID_SIZE - 1)
                new_y = min(max(0, agent["pos"][1] + dy), GRID_SIZE - 1)
                if not obstacle_mask[new_y, new_x]:
                    agent["pos"] = [new_x, new_y]
                    agent["budget"] -= MOVE_COST
                    # Observe after move
                    observe(agent["pos"])
                else:
                    print(f"Agent {agent['id']} cannot move to ({new_x}, {new_y}) - impassable. Staying put.")
            else:
                print(f"Invalid direction '{direction}' for agent {agent['id']}. Staying put.")
        else:
            print(f"No move for agent {agent['id']}. Staying put.")

    # Break if all budgets depleted
    if all(agent["budget"] < MOVE_COST for agent in agents if agent["active"]):
        print("All agents out of budget. Simulation ended.")
        break

# Turn off interactive mode and show final plot
# plt.ioff()
# plt.show()
print("Simulation completed.")

# Final entropy analysis
print("\n" + "="*50)
print("ENTROPY ANALYSIS RESULTS")
print("="*50)

summary = entropy_tracker.get_summary_stats()
print(f"Initial Entropy: {summary['initial_entropy']:.4f}")
print(f"Final Entropy: {summary['final_entropy']:.4f}")
print(f"Total Entropy Reduction: {summary['total_entropy_reduction']:.4f}")
print(f"Average Reduction per Step: {summary['avg_reduction_per_step']:.4f}")
print(f"Efficiency per Observation: {summary['efficiency_per_observation']:.4f}")
print(f"Total Observations: {summary['total_observations']}")
print(f"Convergence Step: {summary['convergence_step'] if summary['convergence_step'] else 'Not reached'}")

# Plot entropy trajectory
entropy_plot_path = os.path.join(OUTPUT_DIR, "entropy_analysis.png")
entropy_tracker.plot_entropy_trajectory(save_path=entropy_plot_path)

# Save entropy data to file
entropy_data = {
    'step_times': entropy_tracker.step_times,
    'entropy_history': entropy_tracker.entropy_history,
    'entropy_reduction_history': entropy_tracker.entropy_reduction_history,
    'entropy_rate_history': entropy_tracker.entropy_rate_history,
    'total_observations_history': entropy_tracker.total_observations_history,
    'summary_stats': summary
}

entropy_file_path = os.path.join(OUTPUT_DIR, "entropy_data.json")
with open(entropy_file_path, 'w') as f:
    json.dump(entropy_data, f, indent=2)

print(f"\nEntropy analysis saved to: {entropy_plot_path}")
print(f"Entropy data saved to: {entropy_file_path}")