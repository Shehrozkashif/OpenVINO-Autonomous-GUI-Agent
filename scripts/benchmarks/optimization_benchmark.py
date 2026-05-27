# scripts/benchmarks/optimization_benchmark.py
"""
Measure KV cache impact and overall latency.
Run before and after Phase 9 to quantify improvement.
Target: 40% latency reduction.
"""
import time
from core.pipeline.optimized_pipeline import OptimizedLLMPipeline


def benchmark_kv_cache(model_path: str, n_turns: int = 5):
    pipe = OptimizedLLMPipeline(model_path)
    prompts = [f"Plan desktop automation step {i+1}" for i in range(n_turns)]

    # Without KV cache (reset each time)
    times_no_cache = []
    for p in prompts:
        pipe.reset()
        start = time.time()
        pipe.generate([{"role": "user", "content": p}], max_tokens=50)
        times_no_cache.append(time.time() - start)

    # With KV cache (single session)
    times_cached = []
    pipe.reset()
    for p in prompts:
        start = time.time()
        pipe.generate([{"role": "user", "content": p}], max_tokens=50)
        times_cached.append(time.time() - start)

    avg_no = sum(times_no_cache) / len(times_no_cache) * 1000
    avg_cached = sum(times_cached) / len(times_cached) * 1000
    reduction = (1 - avg_cached / avg_no) * 100

    print(f"Without KV cache: {avg_no:.0f}ms avg")
    print(f"With KV cache:    {avg_cached:.0f}ms avg")
    print(f"Improvement:      {reduction:.1f}% (target: 40%)")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov"
    benchmark_kv_cache(model_path)
