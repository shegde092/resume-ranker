import time
import psutil
from prometheus_client import Summary, Gauge, Counter

# Latency tracking
STAGE_LATENCY = Summary(
    'stage_latency_seconds', 
    'Time spent in each pipeline stage',
    ['stage']
)

# RAM tracking
RAM_USAGE = Gauge(
    'ram_usage_bytes',
    'Current RAM usage of the process'
)

# CPU tracking
CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU utilization'
)

# Throughput
CROSS_ENCODER_THROUGHPUT = Counter(
    'cross_encoder_processed_blocks',
    'Number of blocks processed by cross encoder'
)

CROSS_ENCODER_BATCH_LATENCY = Summary(
    'cross_encoder_batch_latency_seconds',
    'Time spent computing a single batch'
)

CROSS_ENCODER_FAILURES = Counter(
    'cross_encoder_failures_total',
    'Number of inference failures in cross encoder'
)

# Quality tracking
RETRIEVAL_MISS_RATE = Gauge(
    'retrieval_miss_rate',
    'Degradation of shortlist quality / retrieval miss rate'
)

SHORTLIST_SIZE = Gauge(
    "retrieval_shortlist_size",
    "Number of candidates after retrieval"
)

FUSION_CANDIDATES = Gauge(
    "fusion_candidate_count",
    "Number of candidates entering fusion stage"
)

# Confidence tracking
CONFIDENCE_DISTRIBUTION = Counter(
    'ranking_confidence_total',
    'Distribution of ranking confidence',
    ['level'] # 'high', 'medium', 'low'
)

def update_system_usage():
    """Update the Prometheus RAM and CPU usage gauges."""
    process = psutil.Process()
    RAM_USAGE.set(process.memory_info().rss)
    CPU_USAGE.set(psutil.cpu_percent())

class LatencyTimer:
    """Context manager for timing pipeline stages"""
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        STAGE_LATENCY.labels(stage=self.stage_name).observe(duration)
        update_system_usage()
