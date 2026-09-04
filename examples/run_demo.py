from lagrangian_uc import run_benchmark

result = run_benchmark(iterations=120)
print(f"exact objective: {result['exact']['objective']:.3f}")
print(f"best LR lower bound: {result['lagrangian']['best_lower_bound']:.3f}")
print(f"best recovered upper bound: {result['lagrangian']['best_upper_bound']:.3f}")
