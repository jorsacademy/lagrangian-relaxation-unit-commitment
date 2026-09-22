from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .baselines import IdleBatteryPolicy, PeakShavingPolicy, RuleBasedTariffPolicy
from .environment import IndustrialEnergyEnv


@dataclass
class EvaluationResult:
    policy: str
    mean_total_cost: float
    std_total_cost: float
    mean_peak_kw: float
    mean_energy_cost: float
    mean_degradation_cost: float


def run_policy(policy, episodes: int = 20, seed: int = 100) -> EvaluationResult:
    totals: list[float] = []
    peaks: list[float] = []
    energy_costs: list[float] = []
    degradation_costs: list[float] = []

    for episode in range(episodes):
        env = IndustrialEnergyEnv()
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = policy(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        totals.append(info["total_cost"])
        peaks.append(info["max_grid_import_kw"])
        energy_costs.append(info["energy_cost"])
        degradation_costs.append(info["degradation_cost"])
        env.close()

    return EvaluationResult(
        policy=policy.__class__.__name__,
        mean_total_cost=float(np.mean(totals)),
        std_total_cost=float(np.std(totals)),
        mean_peak_kw=float(np.mean(peaks)),
        mean_energy_cost=float(np.mean(energy_costs)),
        mean_degradation_cost=float(np.mean(degradation_costs)),
    )


def evaluate_sac(model_path: str, episodes: int = 20, seed: int = 100) -> EvaluationResult:
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install RL dependencies with: pip install -e '.[rl]'") from exc

    model = SAC.load(model_path)

    class SACPolicy:
        def __call__(self, observation):
            action, _ = model.predict(observation, deterministic=True)
            return action

    result = run_policy(SACPolicy(), episodes=episodes, seed=seed)
    result.policy = "SAC"
    return result


def print_result(result: EvaluationResult) -> None:
    print(
        f"{result.policy:24s} | total cost={result.mean_total_cost:8.2f} ± {result.std_total_cost:6.2f} "
        f"| peak={result.mean_peak_kw:7.2f} kW | energy={result.mean_energy_cost:8.2f} "
        f"| battery wear={result.mean_degradation_cost:6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate industrial energy-management controllers.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--model", default=None, help="Optional Stable-Baselines3 SAC model path.")
    args = parser.parse_args()

    policies = [IdleBatteryPolicy(), RuleBasedTariffPolicy(), PeakShavingPolicy()]
    for policy in policies:
        print_result(run_policy(policy, episodes=args.episodes, seed=args.seed))
    if args.model:
        print_result(evaluate_sac(args.model, episodes=args.episodes, seed=args.seed))


if __name__ == "__main__":
    main()
