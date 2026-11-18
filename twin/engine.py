"""
Time-stepped (daily) discrete simulation engine for the supply-chain twin.

Daily sequence of events
  1. Receive shipments / production that arrive today
  2. Stores serve customer demand (unmet demand = lost sale)
  3. Upstream nodes ship against their order backlogs (FIFO):
       DCs -> stores, factory FG -> DCs, suppliers -> factory RM
  4. Factory releases production (limited by capacity and raw material)
  5. Every node reviews inventory position and places (s, S) orders
  6. Costs are booked

Inventory position (IP) = on hand + on order - backlog owed downstream.
"""
from collections import defaultdict, deque
import numpy as np

from .config import NetworkConfig, CostConfig, Policy
from .demand import generate_store_demand
from . import m5
from .disruptions import DisruptionState


class SupplyChainTwin:
    def __init__(self, policy: Policy, events: list, seed: int,
                 horizon: int = 365, net: NetworkConfig = None,
                 cost: CostConfig = None, demand: dict = None, rm_agent=None):
        self.p = policy
        self.net = net or NetworkConfig()
        self.c = cost or CostConfig()
        self.T = horizon
        self.rng = np.random.default_rng(seed)
        self.dis = DisruptionState(events)
        self.rm_agent = rm_agent          # optional learning agent for RM sourcing
        if demand is not None:
            self.demand = demand
        elif m5.enabled():                 # real Walmart M5 demand window
            self.demand = m5.demand_for(seed, self.net.store_base_demand, horizon)
        else:                              # synthetic M5-style demand
            self.demand = generate_store_demand(
                self.net.store_base_demand, horizon, self.rng,
                dispersion=self.net.demand_dispersion)

        n = self.net
        D = n.total_base_demand
        # mean daily throughput per node -> converts days-of-cover to units
        self.mu = {s: m for s, m in n.store_base_demand.items()}
        for dc in n.dcs:
            self.mu[dc] = n.dc_base_demand(dc)
        self.mu["FG"] = D
        self.mu["RM"] = D

        self.cap_factory = n.factory_capacity_mult * D
        self.cap_s1 = n.s1_capacity_mult * D
        self.cap_s2 = n.s2_capacity_mult * D

        # state ---------------------------------------------------------
        self.on_hand = {}
        for s in n.stores:
            self.on_hand[s] = self._S(s)
        for dc in n.dcs:
            self.on_hand[dc] = self._S(dc)
        self.on_hand["FG"] = self._S("FG")
        self.on_hand["RM"] = self._S("RM")
        self.on_order = defaultdict(float)
        self.backlog = {dc: deque() for dc in n.dcs}     # store orders at DC
        self.backlog["FG"] = deque()                     # DC orders at factory
        self.backlog["S1"] = deque()                     # RM orders at S1
        self.backlog["S2"] = deque()
        self.arrivals = defaultdict(list)                # day -> [(node, qty)]

        # logs ------------------------------------------------------------
        self.log = {k: np.zeros(self.T) for k in (
            "demand", "sales", "lost", "inv_store", "inv_dc", "inv_fg", "inv_rm",
            "orders_dc", "orders_fg", "prod", "rm_s1", "rm_s2",
            "c_hold", "c_order", "c_lost", "c_premium")}

    # helpers -------------------------------------------------------------
    def _cover(self, node):
        if node in self.net.stores:
            return self.p.store
        if node in self.net.dcs:
            return self.p.dc
        return self.p.fg if node == "FG" else self.p.rm

    def _s(self, node):
        return self._cover(node)[0] * self.mu[node]

    def _S(self, node):
        return self._cover(node)[1] * self.mu[node]

    def _owed(self, node):
        return sum(q for _, q in self.backlog[node]) if node in self.backlog else 0.0

    def _ip(self, node):
        return self.on_hand[node] + self.on_order[node] - self._owed(node)

    def _ship(self, src, dest, qty, lt, t):
        if qty <= 0:
            return
        self.arrivals[t + lt].append((dest, qty))

    def _serve_backlog(self, node, available, lead_time_fn, t):
        """Ship FIFO from `available` units against node's backlog.
        Returns units shipped."""
        shipped = 0.0
        q = self.backlog[node]
        while q and available - shipped > 1e-9:
            dest, need = q[0]
            give = min(need, available - shipped)
            self._ship(node, dest, give, lead_time_fn(dest), t)
            shipped += give
            if give >= need - 1e-9:
                q.popleft()
            else:
                q[0] = (dest, need - give)
        return shipped

    # main loop -----------------------------------------------------------
    def run(self):
        n, c, L = self.net, self.c, self.log
        for t in range(self.T):
            # 1. arrivals
            for node, qty in self.arrivals.pop(t, []):
                self.on_hand[node] += qty
                self.on_order[node] -= qty

            # 2. customer demand at stores (lost sales)
            f = self.dis.demand_factor(t)
            for s in n.stores:
                d = self.demand[s][t] * f
                sold = min(self.on_hand[s], d)
                self.on_hand[s] -= sold
                L["demand"][t] += d
                L["sales"][t] += sold
                L["lost"][t] += d - sold

            # 3a. DCs ship to stores
            for dc in n.dcs:
                if not self.dis.dc_down(t, dc):
                    sh = self._serve_backlog(dc, self.on_hand[dc],
                                             lambda _: n.lt_dc_to_store, t)
                    self.on_hand[dc] -= sh
            # 3b. factory ships finished goods to DCs
            sh = self._serve_backlog("FG", self.on_hand["FG"],
                                     lambda _: n.lt_factory_to_dc, t)
            self.on_hand["FG"] -= sh
            # 3c. suppliers ship raw material
            lt1 = n.lt_s1_to_factory + self.dis.s1_extra_delay(t)
            s1 = self._serve_backlog("S1", self.cap_s1 * self.dis.s1_capacity_factor(t),
                                     lambda _: lt1, t)
            s2 = self._serve_backlog("S2", self.cap_s2, lambda _: n.lt_s2_to_factory, t)
            L["rm_s1"][t], L["rm_s2"][t] = s1, s2
            L["c_premium"][t] = s2 * c.backup_premium

            # 4. production release
            ip_fg = self._ip("FG")
            prod = 0.0
            if ip_fg <= self._s("FG"):
                cap = self.cap_factory * self.dis.factory_capacity_factor(t)
                prod = max(0.0, min(self._S("FG") - ip_fg, cap, self.on_hand["RM"]))
                if prod > 0:
                    self.on_hand["RM"] -= prod
                    self.on_order["FG"] += prod
                    self.arrivals[t + n.lt_production].append(("FG", prod))
                    L["c_order"][t] += c.setup_cost_production
            L["prod"][t] = prod

            # 5. ordering decisions
            for s in n.stores:                         # stores -> DC
                ip = self._ip(s)
                if ip <= self._s(s):
                    q = self._S(s) - ip
                    self.backlog[n.stores[s]].append((s, q))
                    self.on_order[s] += q
                    L["orders_dc"][t] += q
                    L["c_order"][t] += c.order_cost_store
            for dc in n.dcs:                           # DCs -> factory
                ip = self._ip(dc)
                if ip <= self._s(dc):
                    q = self._S(dc) - ip
                    self.backlog["FG"].append((dc, q))
                    self.on_order[dc] += q
                    L["orders_fg"][t] += q
                    L["c_order"][t] += c.order_cost_dc
            use_backup = self.p.dual_sourcing and self.dis.s1_disrupted(t)
            if use_backup and self.backlog["S1"]:      # re-route open orders
                self.backlog["S2"].extend(self.backlog["S1"])
                self.backlog["S1"].clear()
            ip_rm = self._ip("RM")                     # factory -> suppliers
            if self.rm_agent is not None:              # RL agent decides sourcing
                q1, q2 = self.rm_agent.act(t, self)
                for src, q in (("S1", q1), ("S2", q2)):
                    if q > 0:
                        self.backlog[src].append(("RM", q))
                        self.on_order["RM"] += q
                        L["c_order"][t] += c.order_cost_rm
            elif ip_rm <= self._s("RM"):
                q = self._S("RM") - ip_rm
                self.backlog["S2" if use_backup else "S1"].append(("RM", q))
                self.on_order["RM"] += q
                L["c_order"][t] += c.order_cost_rm

            # 6. holding & lost-sales costs
            inv_store = sum(self.on_hand[s] for s in n.stores)
            inv_dc = sum(self.on_hand[dc] for dc in n.dcs)
            L["inv_store"][t], L["inv_dc"][t] = inv_store, inv_dc
            L["inv_fg"][t], L["inv_rm"][t] = self.on_hand["FG"], self.on_hand["RM"]
            L["c_hold"][t] = (inv_store * c.h(c.value_store) + inv_dc * c.h(c.value_dc)
                              + self.on_hand["FG"] * c.h(c.value_fg)
                              + self.on_hand["RM"] * c.h(c.value_rm))
            L["c_lost"][t] = L["lost"][t] * c.lost_sale_penalty
            if self.rm_agent is not None:
                self.rm_agent.feedback(-(L["c_hold"][t] + L["c_order"][t] + L["c_lost"][t]
                                         + L["c_premium"][t]))
        return L
