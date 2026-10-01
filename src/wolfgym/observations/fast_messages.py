"""Opt-in message observation fast path for the frozen, append-only market.

No simulator mutation or RNG use. The historical builder remains unchanged.
Caches belong to a builder/episode, not a global simulator-id lookup.
"""
from bisect import bisect_left, insort
import weakref

import numpy as np

from wolfgym.observations.builder import ObservationBuilder, _IDENTIFIER_TOKEN
from wolfgym.observations.schema import PublicMessageObservation


class FastMessageObservationBuilder(ObservationBuilder):
    def __init__(self, config=None, *, cache_ids=True, index_events=True):
        super().__init__(config)
        self.cache_ids = cache_ids
        self.index_events = index_events
        self.clear_cache()

    def clear_cache(self):
        self._simulator = None
        self._identity_maps = None
        self._public_ids = None
        self._events = None
        self._indexed_count = 0
        self._last_event = None
        self._events_by_day = {}
        self._days = []
        self._last_day = -1

    def _bind(self, simulator):
        maps = (simulator._public_entity_ids, simulator._public_asset_ids,
                simulator.graph.id_to_node, simulator.market.assets)
        same_world = self._simulator is not None and self._simulator() is simulator
        same_maps = self._identity_maps is not None and all(a is b for a, b in zip(maps, self._identity_maps))
        if not same_world or not same_maps or simulator._next_day < self._last_day:
            self.clear_cache()
            self._simulator = weakref.ref(simulator)
            self._identity_maps = maps
        self._last_day = simulator._next_day

    def _identifiers(self, simulator):
        if not self.cache_ids or self._public_ids is None:
            ids = {key: simulator.public_entity_id(key) for key in simulator.graph.id_to_node}
            ids.update({key: simulator.public_asset_id(key) for key in simulator.market.assets})
            if self.cache_ids:
                self._public_ids = ids
            return ids
        return self._public_ids

    def _recent_events(self, simulator, first_day):
        events = getattr(simulator.social, 'message_events', ())
        if not self.index_events:
            return events
        # set_state/fork/reset replace objects; also handle cleared/truncated logs.
        # The frozen simulator only appends completed event dictionaries. Direct
        # edits to past event days or identity maps require clear_cache().
        prefix_changed = (self._indexed_count > 0 and len(events) >= self._indexed_count
                          and events[self._indexed_count - 1] is not self._last_event)
        if events is not self._events or len(events) < self._indexed_count or prefix_changed:
            self._events = events
            self._indexed_count = 0
            self._events_by_day = {}
            self._days = []
        for event in events[self._indexed_count:]:
            day = int(event.get('day', -1))
            if day not in self._events_by_day:
                self._events_by_day[day] = []
                insort(self._days, day)
            self._events_by_day[day].append(event)
        self._indexed_count = len(events)
        self._last_event = events[-1] if events else None
        # Keep insertion order within each day: the original stable sort relies
        # on that order when all four ordering fields are equal.
        return (event for day in self._days[bisect_left(self._days, first_day):]
                for event in self._events_by_day[day])

    def _build_messages(self, simulator):
        if self.config.max_messages == 0:
            return ()
        self._bind(simulator)
        # get_state().next_day is exactly _next_day but copies the entire graph.
        first_day = simulator._next_day - self.config.window_days
        public_ids = self._identifiers(simulator)
        candidates = []
        for event in self._recent_events(simulator, first_day):
            day = int(event.get('day', -1))
            text = str(event.get('text', '')).strip()
            sender = str(event.get('sender_id', ''))
            asset = str(event.get('asset', ''))
            if day < first_day or not text or sender not in simulator.graph.id_to_node:
                continue
            if asset not in simulator.market.assets:
                continue
            text = _IDENTIFIER_TOKEN.sub(lambda match: public_ids.get(match.group(0), match.group(0)), text)
            reach = float(np.clip(float(event.get('audience_size', 0.0)) / max(simulator.n_society, 1), 0.0, 1.0))
            public_sender = simulator.public_entity_id(sender)
            public_asset = simulator.public_asset_id(asset)
            message = PublicMessageObservation(day=day, sender_id=public_sender, asset_id=public_asset,
                kind=str(event.get('kind', 'post')), text=text, audience_reach=reach)
            order = (-day, -reach, public_sender, str(event.get('message_id', '')))
            candidates.append((order, message))
        candidates.sort(key=lambda item: item[0])
        return tuple(message for _, message in candidates[:self.config.max_messages])
