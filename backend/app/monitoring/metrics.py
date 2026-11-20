"""Minimal Prometheus metrics (text exposition format 0.0.4), no extra dependency.

Process-level counters/histograms are recorded by the gateway middleware; model
and job gauges are read from the database at scrape time, so they are correct
even with several API workers.
"""

import threading
from collections import defaultdict

BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _labels(d: dict) -> str:
    if not d:
        return ""
    esc = lambda v: str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in sorted(d.items())) + "}"


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self.hist: dict[str, dict[tuple, list]] = defaultdict(dict)
        self.help: dict[str, tuple[str, str]] = {}

    def inc(self, name: str, labels: dict, value: float = 1.0, help_: str = ""):
        with self._lock:
            self.help.setdefault(name, ("counter", help_))
            self.counters[name][tuple(sorted(labels.items()))] += value

    def observe(self, name: str, labels: dict, value: float, help_: str = ""):
        with self._lock:
            self.help.setdefault(name, ("histogram", help_))
            key = tuple(sorted(labels.items()))
            h = self.hist[name].setdefault(key, [[0] * len(BUCKETS), 0.0, 0])
            for i, b in enumerate(BUCKETS):
                if value <= b:
                    h[0][i] += 1
            h[1] += value
            h[2] += 1

    def render(self) -> list[str]:
        lines = []
        with self._lock:
            for name, series in self.counters.items():
                kind, help_ = self.help[name]
                lines += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]
                lines += [f"{name}{_labels(dict(k))} {v}" for k, v in series.items()]
            for name, series in self.hist.items():
                _, help_ = self.help[name]
                lines += [f"# HELP {name} {help_}", f"# TYPE {name} histogram"]
                for k, (buckets, total, count) in series.items():
                    base = dict(k)
                    for b, c in zip(BUCKETS, buckets):
                        lines.append(f"{name}_bucket{_labels({**base, 'le': b})} {c}")
                    lines.append(f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {count}")
                    lines.append(f"{name}_sum{_labels(base)} {total}")
                    lines.append(f"{name}_count{_labels(base)} {count}")
        return lines


REGISTRY = Registry()


def gauge_lines(name: str, help_: str, samples: list[tuple[dict, float]]) -> list[str]:
    lines = [f"# HELP {name} {help_}", f"# TYPE {name} gauge"]
    lines += [f"{name}{_labels(labels)} {value}" for labels, value in samples]
    return lines
