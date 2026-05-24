# scripts/benchmarks/latency_benchmark.py
"""Run after OVMS is up to establish performance baseline. Target: < 2000ms per VLM call."""
import time
from core.capture.screenshot import ScreenCapture
from core.pipeline.ovms_client import OVMSClient


def benchmark_vlm(n_trials: int = 5):
    cap = ScreenCapture()
    client = OVMSClient()
    latencies = []

    for i in range(n_trials):
        b64 = cap.capture_as_base64()
        start = time.time()
        result = client.query_vlm(
            "What application is currently open? Output the app name only.",
            b64, max_tokens=20
        )
        latencies.append((time.time() - start) * 1000)
        print(f"  Trial {i+1}: {latencies[-1]:.0f}ms → {result.content}")

    avg = sum(latencies) / len(latencies)
    print(f"\nVLM avg: {avg:.0f}ms | Target: < 2000ms")
    return avg


if __name__ == "__main__":
    benchmark_vlm()
