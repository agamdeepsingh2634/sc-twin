"""
Disruption library. Each disruption is a dict:
    {type, start, duration, magnitude, label}

Types
  supplier_outage   : primary supplier S1 ships nothing (fire, shutdown, export ban)
  port_blockage     : shipments S1 -> factory dispatched in the window take
                      `magnitude` extra days (Suez Canal 2021, port strikes)
  demand_surge      : store demand x magnitude (COVID panic buying)
  factory_capacity  : factory capacity x magnitude (strike, lockdown, breakdown)
  dc_outage         : one DC cannot ship (flood, fire); target = DC name

Scenario templates are sampled randomly in Monte Carlo runs so that results
cover a distribution of start dates and severities, not one lucky case.
"""
import numpy as np

SCENARIOS = {
    "No disruption": [],
    "Supplier outage": [
        dict(type="supplier_outage", duration=(14, 21, 35), magnitude=(0, 0, 0)),
    ],
    "Port blockage": [
        dict(type="port_blockage", duration=(6, 10, 18), magnitude=(10, 14, 21)),
    ],
    "Demand surge": [
        dict(type="demand_surge", duration=(7, 14, 21), magnitude=(1.5, 1.8, 2.2)),
    ],
    "Factory capacity loss": [
        dict(type="factory_capacity", duration=(20, 30, 45), magnitude=(0.4, 0.5, 0.6)),
    ],
    "DC flood (North)": [
        dict(type="dc_outage", target="DC_North", duration=(5, 8, 12), magnitude=(0, 0, 0)),
    ],
    "Compound: outage + surge": [
        dict(type="supplier_outage", duration=(14, 21, 35), magnitude=(0, 0, 0)),
        dict(type="demand_surge", duration=(7, 14, 21), magnitude=(1.5, 1.8, 2.2), offset=5),
    ],
}


def sample_scenario(name: str, rng: np.random.Generator, start_window=(100, 200),
                    severity: float = 1.0) -> list:
    """Draw one concrete disruption list from a scenario template using
    triangular distributions (min, mode, max) for duration and magnitude."""
    template = SCENARIOS[name]
    if not template:
        return []
    start = int(rng.integers(*start_window))
    events = []
    for t in template:
        lo, mode, hi = t["duration"]
        dur = int(round(rng.triangular(lo, mode, hi))) if hi > lo else int(mode)
        dur = max(1, int(round(dur * severity)))      # sensitivity: scale duration
        mlo, mmode, mhi = t["magnitude"]
        mag = float(rng.triangular(mlo, mmode, mhi)) if mhi > mlo else float(mmode)
        events.append(dict(type=t["type"], target=t.get("target"),
                           start=start + t.get("offset", 0), duration=dur,
                           magnitude=mag, label=name))
    return events


class DisruptionState:
    """Answers 'what is disrupted on day t?' for the simulation engine."""

    def __init__(self, events: list):
        self.events = events

    def _active(self, t, typ, target=None):
        for e in self.events:
            if e["type"] == typ and e["start"] <= t < e["start"] + e["duration"]:
                if target is None or e.get("target") in (None, target):
                    return e
        return None

    def s1_capacity_factor(self, t):
        return 0.0 if self._active(t, "supplier_outage") else 1.0

    def s1_extra_delay(self, t):
        e = self._active(t, "port_blockage")
        return int(round(e["magnitude"])) if e else 0

    def demand_factor(self, t):
        e = self._active(t, "demand_surge")
        return e["magnitude"] if e else 1.0

    def factory_capacity_factor(self, t):
        e = self._active(t, "factory_capacity")
        return e["magnitude"] if e else 1.0

    def dc_down(self, t, dc):
        return self._active(t, "dc_outage", dc) is not None

    def s1_disrupted(self, t):
        """Visible to the planner -> triggers dual sourcing."""
        return self.s1_capacity_factor(t) == 0.0 or self.s1_extra_delay(t) > 0

    @property
    def first_start(self):
        return min((e["start"] for e in self.events), default=None)

    @property
    def last_end(self):
        return max((e["start"] + e["duration"] for e in self.events), default=None)
