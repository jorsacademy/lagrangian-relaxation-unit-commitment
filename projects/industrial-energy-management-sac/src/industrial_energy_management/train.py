from __future__ import annotations

import argparse
from pathlib import Path

from .environment import IndustrialEnergyEnv


def train_sac(total_timesteps: int, output: str, seed: int) -> None:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    env = Monitor(IndustrialEnergyEnv())
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC for industrial energy management.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", default="models/sac_industrial_energy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_sac(args.timesteps, args.output, args.seed)


if __name__ == "__main__":
    main()
