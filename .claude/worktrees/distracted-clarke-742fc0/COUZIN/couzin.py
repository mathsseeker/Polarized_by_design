"""
Couzin collective motion model (Couzin et al., 2002).

Agents follow three zonal rules:
  1. Zone of Repulsion  (r < r_r)   → move away from neighbours
  2. Zone of Orientation (r_r < r < r_o) → align with neighbours
  3. Zone of Attraction  (r_o < r < r_a) → move toward neighbours
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ── Agent ────────────────────────────────────────────────────────────────
class Agent:
    """Single individual in the Couzin model."""

    def __init__(self, position: np.ndarray, direction: np.ndarray, speed: float):
        """
        Parameters
        ----------
        position : (2,) or (3,) array  – spatial coordinates
        direction : same shape as position – unit vector of heading
        speed : scalar cruise speed
        """
        self.position = np.asarray(position, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)  # ensure unit vector
        self.speed = speed

    def velocity(self) -> np.ndarray:
        return self.speed * self.direction


# ── Model ────────────────────────────────────────────────────────────────
class CouzinModel:
    """
    Manages an ensemble of agents and runs the zonal interaction rules.

    Parameters
    ----------
    n_agents : int           – number of agents
    r_r : float              – repulsion zone radius
    r_o : float              – orientation zone radius  (must be > r_r)
    r_a : float              – attraction zone radius   (must be > r_o)
    blind_angle : float      – half-angle of blind zone behind agent (radians)
    speed : float            – common cruise speed
    turning_rate : float     – max turning angle per step (radians)
    noise : float            – angular noise std (radians)
    dt : float               – time step
    domain_size : float|None – side length of periodic square domain (None = unbounded)
    dim : int                – spatial dimensions (2 or 3)
    seed : int|None          – RNG seed for reproducibility
    """

    def __init__(
        self,
        n_agents: int = 50,
        r_r: float = 1.0,
        r_o: float = 5.0,
        r_a: float = 10.0,
        blind_angle: float = np.pi / 4,
        speed: float = 1.0,
        turning_rate: float = np.pi / 4,
        noise: float = 0.0,
        dt: float = 0.1,
        domain_size: float | None = None,
        dim: int = 2,
        seed: int | None = None,
    ):
        self.n_agents = n_agents
        self.r_r = r_r
        self.r_o = r_o
        self.r_a = r_a
        self.blind_angle = blind_angle
        self.speed = speed
        self.turning_rate = turning_rate
        self.noise = noise
        self.dt = dt
        self.domain_size = domain_size
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        self.agents = self._init_agents()

    # ── initialisation ───────────────────────────────────────────────────
    def _init_agents(self) -> list[Agent]:
        """Create agents with random positions and headings."""
        agents = []
        for _ in range(self.n_agents):
            # random position
            if self.domain_size is not None:
                pos = self.rng.uniform(0, self.domain_size, size=self.dim)
            else:
                pos = self.rng.uniform(-10, 10, size=self.dim)

            # random unit direction
            direction = self.rng.normal(size=self.dim)
            direction /= np.linalg.norm(direction)

            agents.append(Agent(pos, direction, self.speed))
        return agents

    # ── geometry helpers ─────────────────────────────────────────────────
    def _displacement(self, a: Agent, b: Agent) -> np.ndarray:
        """Vector from a to b, respecting periodic boundaries if set."""
        d = b.position - a.position
        if self.domain_size is not None:
            d = d - self.domain_size * np.round(d / self.domain_size)
        return d

    def _is_visible(self, agent: Agent, displacement: np.ndarray) -> bool:
        """Return True if displacement falls outside agent's blind zone."""
        dist = np.linalg.norm(displacement)
        if dist < 1e-12:
            return False
        cos_angle = np.dot(agent.direction, displacement) / dist
        # blind zone is behind the agent: angle > (pi - blind_angle)
        # visible if angle < (pi - blind_angle), i.e. cos > cos(pi - blind_angle)
        return cos_angle > np.cos(np.pi - self.blind_angle)

    # ── zonal rules ──────────────────────────────────────────────────────
    def _desired_direction(self, idx: int) -> np.ndarray:
        """
        Compute desired heading for agent `idx` by applying priority rules:
          1. If any neighbour in ZoR  → repulsion  (highest priority)
          2. Else combine ZoO alignment + ZoA attraction
          3. If no neighbours in any zone → keep current heading
        """
        agent = self.agents[idx]
        repulsion_vec = np.zeros(self.dim)
        orientation_vec = np.zeros(self.dim)
        attraction_vec = np.zeros(self.dim)
        n_rep = 0
        n_ori = 0
        n_att = 0

        for j, other in enumerate(self.agents):
            if j == idx:
                continue
            disp = self._displacement(agent, other)
            dist = np.linalg.norm(disp)

            if dist < 1e-12 or dist > self.r_a:
                continue
            if not self._is_visible(agent, disp):
                continue

            if dist < self.r_r:
                # repulsion: move AWAY
                repulsion_vec -= disp / dist
                n_rep += 1
            elif dist < self.r_o:
                # orientation: align
                orientation_vec += other.direction
                n_ori += 1
            else:
                # attraction: move TOWARD
                attraction_vec += disp / dist
                n_att += 1

        # Priority: repulsion > (orientation + attraction) > keep heading
        if n_rep > 0:
            desired = repulsion_vec
        elif n_ori > 0 or n_att > 0:
            desired = np.zeros(self.dim)
            if n_ori > 0:
                desired += orientation_vec / n_ori
            if n_att > 0:
                desired += attraction_vec / n_att
        else:
            desired = agent.direction.copy()

        norm = np.linalg.norm(desired)
        if norm > 1e-12:
            desired /= norm
        else:
            desired = agent.direction.copy()

        return desired

    # ── update ───────────────────────────────────────────────────────────
    def step(self) -> None:
        """Advance the simulation by one time step dt."""
        desired_dirs = [self._desired_direction(i) for i in range(self.n_agents)]

        for i, agent in enumerate(self.agents):
            desired = desired_dirs[i]

            # add angular noise
            if self.noise > 0:
                noise_angle = self.rng.normal(0, self.noise)
                if self.dim == 2:
                    c, s = np.cos(noise_angle), np.sin(noise_angle)
                    rot = np.array([[c, -s], [s, c]])
                    desired = rot @ desired
                # (3D noise omitted for simplicity)

            # limit turning rate
            cos_theta = np.clip(np.dot(agent.direction, desired), -1.0, 1.0)
            angle = np.arccos(cos_theta)
            if angle > self.turning_rate:
                # rotate current direction toward desired by turning_rate
                axis = desired - agent.direction * cos_theta
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-12:
                    axis /= axis_norm
                    agent.direction = (
                        agent.direction * np.cos(self.turning_rate)
                        + axis * np.sin(self.turning_rate)
                    )
                # else: opposite direction, pick arbitrary perpendicular
            else:
                agent.direction = desired

            # normalise
            agent.direction /= np.linalg.norm(agent.direction)

            # move
            agent.position += agent.speed * agent.direction * self.dt

            # periodic wrap
            if self.domain_size is not None:
                agent.position %= self.domain_size

    def run(self, n_steps: int) -> np.ndarray:
        """
        Run for n_steps and return position history.

        Returns
        -------
        positions : (n_steps+1, n_agents, dim) array
        """
        history = np.zeros((n_steps + 1, self.n_agents, self.dim))
        history[0] = np.array([a.position for a in self.agents])

        for t in range(n_steps):
            self.step()
            history[t + 1] = np.array([a.position for a in self.agents])

        return history

    # ── metrics ──────────────────────────────────────────────────────────
    def polarisation(self) -> float:
        """Order parameter: |mean direction| ∈ [0, 1]."""
        dirs = np.array([a.direction for a in self.agents])
        return float(np.linalg.norm(dirs.mean(axis=0)))

    def mean_position(self) -> np.ndarray:
        """Centre of mass of all agents."""
        return np.mean([a.position for a in self.agents], axis=0)


# ── Run & Animate ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create the model with 40 agents in a 50×50 periodic box
    model = CouzinModel(
        n_agents=40,
        r_r=1.0,
        r_o=5.0,
        r_a=10.0,
        speed=1.0,
        turning_rate=np.pi / 6,
        noise=0.1,
        dt=0.2,
        domain_size=50.0,
        dim=2,
        seed=42,
    )

    # Run simulation
    n_steps = 500
    history = model.run(n_steps)

    # ── Animate ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, model.domain_size)
    ax.set_ylim(0, model.domain_size)
    ax.set_aspect("equal")
    ax.set_title("Couzin Model")

    quiver = ax.quiver(
        history[0, :, 0], history[0, :, 1],
        [a.direction[0] for a in model.agents],
        [a.direction[1] for a in model.agents],
        scale=30, color="steelblue",
    )

    def update(frame):
        quiver.set_offsets(history[frame])
        ax.set_title(f"Couzin Model  —  step {frame}/{n_steps}")
        return (quiver,)

    anim = FuncAnimation(fig, update, frames=n_steps + 1, interval=30, blit=True)
    plt.show()
