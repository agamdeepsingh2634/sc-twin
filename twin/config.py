"""
Network, cost and policy configuration for the FMCG supply-chain digital twin.

Units: 1 unit = 1 case of a packaged FMCG product (e.g. a 12-pack of cooking oil
or biscuits). Money is in Indian Rupees (INR). Time step = 1 day.

Network (Walmart-style retail chain for one FMCG product family):

    S1 (primary raw-material supplier, imported via port, cheap, slow)  ┐
                                                                        ├─> Factory (RM -> FG) ─┬─> DC_North ─> Store N1, N2, N3
    S2 (backup domestic supplier, fast, +premium, limited capacity)     ┘                       └─> DC_West  ─> Store W1, W2, W3
"""
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
@dataclass
class NetworkConfig:
    # stores and which DC serves them
    stores: dict = field(default_factory=lambda: {
        "N1": "DC_North", "N2": "DC_North", "N3": "DC_North",
        "W1": "DC_West", "W2": "DC_West", "W3": "DC_West",
    })
    # mean daily demand per store (cases/day) before seasonality
    store_base_demand: dict = field(default_factory=lambda: {
        "N1": 50, "N2": 40, "N3": 30, "W1": 45, "W2": 40, "W3": 35,
    })
    dcs: tuple = ("DC_North", "DC_West")

    # lead times (days)
    lt_dc_to_store: int = 1
    lt_factory_to_dc: int = 3
    lt_production: int = 2
    lt_s1_to_factory: int = 10    # imported raw material, via port
    lt_s2_to_factory: int = 4     # domestic backup supplier

    # capacities as a multiple of total mean demand
    factory_capacity_mult: float = 1.25   # -> 80% utilisation in normal times
    s1_capacity_mult: float = 1.60
    s2_capacity_mult: float = 0.60        # backup cannot fully replace primary

    # demand noise: negative-binomial dispersion (lower = noisier)
    demand_dispersion: float = 25.0

    @property
    def total_base_demand(self) -> float:
        return float(sum(self.store_base_demand.values()))

    def dc_base_demand(self, dc: str) -> float:
        return float(sum(m for s, m in self.store_base_demand.items()
                         if self.stores[s] == dc))


# ----------------------------------------------------------------------------
# Costs (INR)
# ----------------------------------------------------------------------------
@dataclass
class CostConfig:
    # value of one case at each echelon; holding cost = 25% per year of value
    holding_rate_per_year: float = 0.25
    value_rm: float = 400.0
    value_fg: float = 800.0
    value_dc: float = 850.0
    value_store: float = 900.0

    lost_sale_penalty: float = 250.0     # lost margin + goodwill per case
    backup_premium: float = 120.0        # extra INR per case bought from S2

    order_cost_store: float = 500.0      # fixed cost per replenishment order
    order_cost_dc: float = 2000.0
    setup_cost_production: float = 1000.0   # FMCG line, low changeover
    order_cost_rm: float = 5000.0

    def h(self, value: float) -> float:
        """Holding cost per case per day."""
        return value * self.holding_rate_per_year / 365.0


# ----------------------------------------------------------------------------
# Inventory policy: (s, S) at every echelon, expressed in DAYS OF COVER of the
# mean demand flowing through that node. s = reorder point, S = order-up-to.
# ----------------------------------------------------------------------------
@dataclass
class Policy:
    name: str
    store: tuple = (3.0, 6.0)
    dc: tuple = (6.0, 11.0)
    fg: tuple = (5.0, 9.0)
    rm: tuple = (14.0, 24.0)
    dual_sourcing: bool = False   # switch RM orders to S2 when S1 is disrupted

    def as_dict(self) -> dict:
        return {
            "policy": self.name,
            "store_s": self.store[0], "store_S": self.store[1],
            "dc_s": self.dc[0], "dc_S": self.dc[1],
            "fg_s": self.fg[0], "fg_S": self.fg[1],
            "rm_s": self.rm[0], "rm_S": self.rm[1],
            "dual_sourcing": self.dual_sourcing,
        }


# Three reference strategies for the baseline study
LEAN = Policy("Lean (JIT)", store=(2.5, 5.0), dc=(5.0, 9.0), fg=(4.0, 7.0),
              rm=(12.0, 20.0), dual_sourcing=False)
BUFFERED = Policy("Buffered", store=(4.0, 8.0), dc=(9.0, 16.0), fg=(8.0, 14.0),
                  rm=(25.0, 40.0), dual_sourcing=False)
DUAL = Policy("Lean + Dual sourcing", store=(2.5, 5.0), dc=(5.0, 9.0),
              fg=(4.0, 7.0), rm=(12.0, 20.0), dual_sourcing=True)

REFERENCE_POLICIES = [LEAN, BUFFERED, DUAL]

# Base-stock: order EVERY day up to S (s just below S) - classic textbook policy
BASESTOCK = Policy("Base-stock", store=(5.99, 6.0), dc=(11.99, 12.0),
                   fg=(9.99, 10.0), rm=(29.99, 30.0), dual_sourcing=False)
