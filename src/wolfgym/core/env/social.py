"""Social layer: scale-free graph, message propagation, cascade tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

import networkx as nx
import numpy as np


@dataclass
class Message:
    sender_id: str
    asset: str
    sentiment: float            # +1 bullish, -1 bearish
    intensity: float            # >0
    is_harmful: bool = False    # came from a harmful source
    is_bot: bool = False
    day: int = 0
    # Optional metadata used by the final bounded-rational network signaling
    # game. Legacy scenarios can ignore these fields.
    message_id: str = ""
    kind: str = "post"          # post / reshare / challenge
    root_sender_id: str = ""
    confidence: float = 0.5
    social_proof: float = 0.0
    # Public language is part of simulator state.  It is generated before the
    # message enters the social layer and is therefore copied and replayed with
    # the rest of the state during counterfactual continuations.
    text: str = ""

    def __post_init__(self) -> None:
        self.text = " ".join(str(self.text).split()).strip()[:280]
        if not self.text:
            self.text = default_message_text(
                sentiment=self.sentiment,
                kind=self.kind,
                confidence=self.confidence,
                day=self.day,
            )


def default_message_text(
    *,
    sentiment: float,
    kind: str,
    confidence: float,
    day: int = 0,
) -> str:
    """Render deterministic public language from observable message content.

    The renderer deliberately receives no role, intent, scenario, or harmful
    status.  LLM-driven agents may provide their own ``Message.text``; rule
    agents use this stable fallback so every public post has language evidence.
    """
    direction = (
        "positive" if sentiment > 0.05
        else "negative" if sentiment < -0.05
        else "mixed"
    )
    strong = float(confidence) >= 0.65
    templates = {
        ("post", "positive", True): (
            "Buying interest looks strong and momentum is building.",
            "The latest activity reinforces a strongly positive outlook.",
            "Demand appears firm, and the recent trend still points upward.",
        ),
        ("post", "positive", False): (
            "Recent activity looks encouraging, though I am still watching it closely.",
            "The short-term picture looks positive, but the evidence is not decisive yet.",
            "There are some constructive signals in the latest market activity.",
        ),
        ("post", "negative", True): (
            "Selling pressure looks strong and the outlook is deteriorating.",
            "The latest activity reinforces a strongly negative outlook.",
            "Supply appears heavy, and the recent trend still points downward.",
        ),
        ("post", "negative", False): (
            "Recent activity looks weak, so I am staying cautious.",
            "The short-term picture looks negative, but the evidence is not decisive yet.",
            "There are some concerning signals in the latest market activity.",
        ),
        ("reshare", "positive", True): (
            "Sharing this because the positive momentum looks convincing.",
            "This positive view is consistent with the strong activity I am seeing.",
        ),
        ("reshare", "positive", False): (
            "Sharing this positive view for others to consider.",
            "This cautiously positive take seems worth passing along.",
        ),
        ("reshare", "negative", True): (
            "Sharing this because the downside risk looks convincing.",
            "This cautious view is consistent with the weak activity I am seeing.",
        ),
        ("reshare", "negative", False): (
            "Sharing this cautious view for others to consider.",
            "This mildly negative take seems worth passing along.",
        ),
        ("challenge", "positive", True): (
            "I disagree with the pessimistic view; the recent evidence looks stronger.",
            "The negative interpretation seems overstated given the latest activity.",
        ),
        ("challenge", "positive", False): (
            "I am not convinced by the pessimistic view yet.",
            "The recent evidence does not fully support that negative conclusion.",
        ),
        ("challenge", "negative", True): (
            "I disagree with the optimism; the recent evidence looks weaker.",
            "The positive interpretation seems overstated given the latest activity.",
        ),
        ("challenge", "negative", False): (
            "I am not convinced by the optimistic view yet.",
            "The recent evidence does not fully support that positive conclusion.",
        ),
    }
    options = templates.get((str(kind).casefold(), direction, strong), (
        "Sharing an update based on the latest public market activity.",
        "The latest public signals do not support a clear directional view.",
    ))
    # Never use sender or asset identity to choose wording: raw simulator ids
    # can be correlated with role or scenario. Day and message semantics are
    # already public and therefore safe deterministic variation keys.
    identity = f"{int(day)}|{kind}|{direction}|{int(strong)}"
    index = int.from_bytes(hashlib.blake2s(identity.encode(), digest_size=2).digest(), "big")
    return options[index % len(options)]


@dataclass
class SocialState:
    # per-agent exposure to messages this step (sentiment-weighted)
    exposure: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    harmful_exposure: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # per-asset aggregate signal
    msg_volume: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sentiment_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    cascade_size: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    history: dict[str, list[dict[str, float]]] = field(default_factory=lambda: defaultdict(list))


class SocialGraph:
    def __init__(self, n_agents: int, mean_degree: int, rng: np.random.Generator,
                 graph_type: str = "scale_free"):
        m = max(1, mean_degree // 2)
        if graph_type == "scale_free" and n_agents > m + 1:
            seed = int(rng.integers(0, 2**31 - 1))
            self.g = nx.barabasi_albert_graph(n_agents, m, seed=seed)
        else:
            seed = int(rng.integers(0, 2**31 - 1))
            p = mean_degree / max(n_agents - 1, 1)
            self.g = nx.fast_gnp_random_graph(n_agents, p, seed=seed)
        self.agent_nodes: list[int] = list(self.g.nodes())
        self.id_to_node: dict[str, int] = {}
        self.node_to_id: dict[int, str] = {}
        self.rng = rng

    def assign_ids(self, agent_ids: list[str]) -> None:
        if len(agent_ids) != len(self.agent_nodes):
            raise ValueError("agent_ids length does not match graph nodes")
        for aid, node in zip(agent_ids, self.agent_nodes):
            self.id_to_node[aid] = node
            self.node_to_id[node] = aid

    def degree(self, agent_id: str) -> int:
        return self.g.degree[self.id_to_node[agent_id]]

    def degrees(self) -> dict[str, int]:
        return {self.node_to_id[n]: int(d) for n, d in self.g.degree()}

    def top_degree_ids(self, k: int) -> list[str]:
        order = sorted(self.g.degree, key=lambda x: x[1], reverse=True)[:k]
        return [self.node_to_id[n] for n, _ in order]

    def neighbours(self, agent_id: str) -> list[str]:
        node = self.id_to_node[agent_id]
        return [self.node_to_id[n] for n in self.g.neighbors(node)]


class SocialEnv:
    """Daily message propagation over a static graph.

    Agents post messages targeting an asset with a sentiment. Each message
    reaches direct neighbours with probability ``p_expose``; bots reshare with
    probability ``p_reshare``. We track cumulative exposure per (agent, asset)
    used by retail belief updates and by WolfGuard signals.
    """

    def __init__(self, graph: SocialGraph, scenario, rng: np.random.Generator):
        self.graph = graph
        self.set_rng(rng)
        self.feedback_strength = float(scenario.social.get("feedback_strength", 0.6))
        self.message_decay = float(scenario.social.get("message_decay", 0.5))
        self.bot_ratio = float(scenario.social.get("bot_ratio", 0.05))
        self.p_expose = float(scenario.social.get("p_expose", 0.6))
        self.p_reshare = float(scenario.social.get("p_reshare", 0.25))
        self.state = SocialState()
        # decayed running aggregates
        self._decayed_msg_vol: dict[str, float] = defaultdict(float)
        self._decayed_sent: dict[str, float] = defaultdict(float)
        self._decayed_harm_share: dict[str, float] = defaultdict(float)
        self.message_events: list[dict[str, object]] = []

    def set_rng(self, rng: np.random.Generator) -> None:
        """Install a continuation stream and derive an event-level RNG key.

        Social propagation must not depend on how many earlier messages a
        counterfactual action removed.  We therefore draw one root key here
        and derive every exposure decision from stable event identifiers.
        """
        self.rng = rng
        self._event_randomness_key = int(
            rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        )

    @staticmethod
    def _message_id(day: int, message: Message) -> str:
        """Return a stable public event id independent of list position."""
        if message.message_id:
            return message.message_id
        identity = "|".join((
            str(int(day)),
            message.sender_id,
            message.asset,
            message.kind,
            message.root_sender_id,
            format(float(message.sentiment), ".17g"),
            format(float(message.confidence), ".17g"),
            message.text,
        ))
        digest = hashlib.blake2s(identity.encode(), digest_size=10).hexdigest()
        return f"m{int(day):03d}_{digest}"

    def _event_uniform(self, message_id: str, recipient_id: str, decision: str) -> float:
        """Deterministic U[0, 1) draw for one social propagation event."""
        identity = (
            f"{self._event_randomness_key}|{message_id}|"
            f"{recipient_id}|{decision}"
        )
        value = int.from_bytes(
            hashlib.blake2s(identity.encode(), digest_size=8).digest(), "big"
        )
        return value / float(1 << 64)

    def _event_choice(
        self,
        values: list[str],
        message_id: str,
        recipient_id: str,
        decision: str,
    ) -> str:
        if not values:
            raise ValueError("event choice requires at least one value")
        draw = self._event_uniform(message_id, recipient_id, decision)
        return values[min(int(draw * len(values)), len(values) - 1)]

    def step(self, day: int, messages: list[Message],
             market_returns: dict[str, float]) -> None:
        # decay yesterday's exposure
        for aid in list(self.state.exposure.keys()):
            for ast in list(self.state.exposure[aid].keys()):
                self.state.exposure[aid][ast] *= self.message_decay
                self.state.harmful_exposure[aid][ast] *= self.message_decay
        for ast in list(self._decayed_msg_vol.keys()):
            self._decayed_msg_vol[ast] *= self.message_decay
            self._decayed_sent[ast] *= self.message_decay
            self._decayed_harm_share[ast] *= self.message_decay

        # market feedback: amplify intensity on assets with rising prices
        for m in messages:
            message_id = self._message_id(day, m)
            r = market_returns.get(m.asset, 0.0)
            feedback = 1.0 + self.feedback_strength * max(r, 0.0) * (5.0 if m.sentiment > 0 else 1.0)
            eff_intensity = m.intensity * feedback

            # base exposure: direct neighbours
            sender_node = self.graph.id_to_node.get(m.sender_id)
            if sender_node is None:
                continue
            neighbours = list(self.graph.g.neighbors(sender_node))
            reached: set[str] = set()

            for nb in neighbours:
                aid = self.graph.node_to_id[nb]
                if self._event_uniform(message_id, aid, "direct_exposure") < self.p_expose:
                    reached.add(aid)
                    self.state.exposure[aid][m.asset] += eff_intensity * m.sentiment
                    if m.is_harmful:
                        self.state.harmful_exposure[aid][m.asset] += eff_intensity
                    # bots reshare to neighbours of neighbours (weak)
                    if (
                        m.is_bot
                        and self._event_uniform(message_id, aid, "bot_reshare")
                        < self.p_reshare
                    ):
                        for nb2 in self.graph.g.neighbors(nb):
                            aid2 = self.graph.node_to_id[nb2]
                            reached.add(aid2)
                            self.state.exposure[aid2][m.asset] += 0.5 * eff_intensity * m.sentiment
                            if m.is_harmful:
                                self.state.harmful_exposure[aid2][m.asset] += 0.5 * eff_intensity

            self._decayed_msg_vol[m.asset] += eff_intensity
            self._decayed_sent[m.asset] += eff_intensity * m.sentiment
            if m.is_harmful:
                self._decayed_harm_share[m.asset] += eff_intensity

            self.message_events.append({
                "day": int(day),
                "sender_id": m.sender_id,
                "message_id": message_id,
                "asset": m.asset,
                "kind": m.kind,
                "text": m.text,
                "sentiment": float(m.sentiment),
                "intensity": float(eff_intensity),
                "confidence": float(m.confidence),
                "audience_size": int(len(reached)),
                # Trusted simulator diagnostics only. ObservationBuilder uses
                # an explicit public-field allowlist and never reads these.
                "is_harmful_source": int(m.is_harmful),
                "is_bot": int(m.is_bot),
            })

            # cascade tracking: any agent currently exposed to this asset
            for aid in self.state.exposure:
                if abs(self.state.exposure[aid].get(m.asset, 0.0)) > 0.1:
                    self.state.cascade_size[m.asset].add(aid)

        # snapshot per-asset aggregate
        for ast, vol in self._decayed_msg_vol.items():
            sent = self._decayed_sent[ast] / max(vol, 1e-9)
            harm_share = self._decayed_harm_share[ast] / max(vol, 1e-9)
            self.state.history[ast].append({
                "day": float(day),
                "msg_volume": float(vol),
                "sentiment": float(sent),
                "harmful_msg_share": float(harm_share),
                "cascade_size": float(len(self.state.cascade_size[ast])),
            })

    # -------- views

    def asset_signal(self, asset: str) -> dict[str, float]:
        vol = self._decayed_msg_vol.get(asset, 0.0)
        sent = self._decayed_sent.get(asset, 0.0) / max(vol, 1e-9)
        harm = self._decayed_harm_share.get(asset, 0.0) / max(vol, 1e-9)
        return {
            "msg_volume": vol,
            "sentiment": sent,
            "harmful_msg_share": harm,
            "cascade_size": float(len(self.state.cascade_size.get(asset, set()))),
        }

    def agent_signal(self, agent_id: str, asset: str) -> tuple[float, float]:
        """Return (signed exposure, harmful exposure magnitude)."""
        return (
            self.state.exposure.get(agent_id, {}).get(asset, 0.0),
            self.state.harmful_exposure.get(agent_id, {}).get(asset, 0.0),
        )
