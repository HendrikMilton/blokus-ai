from dataclasses import dataclass, field


@dataclass
class FeatureMetrics:
    """Track latency metrics for a feature."""

    inference_count: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.inference_count += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        # Keep only last 1000 latencies to prevent memory growth
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

    @property
    def avg_latency_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
