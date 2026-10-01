"""One public-evidence controller, fixed before independent v4.1 confirmation."""
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType


class PublicCampaignFreeze:
    def __init__(self, alarm_threshold):
        self.alarm_threshold = alarm_threshold
        self.first_alarm = None
        self.target = None

    def __call__(self, observation, score):
        if observation.messages is None:
            raise ValueError("campaign targeting requires the existing public message view")
        if self.first_alarm is None and score > self.alarm_threshold:
            self.first_alarm = observation.day
            reach = {}
            valid = {a.asset_id for a in observation.assets}
            for message in observation.messages:
                if message.asset_id in valid:
                    reach[message.asset_id] = reach.get(message.asset_id, 0.) + message.audience_reach
            self.target = (max(reach, key=lambda asset: (reach[asset], asset)) if reach else
                           max(observation.assets, key=lambda a: (a.return_3d, a.volume_zscore, a.asset_id)).asset_id)
        if (self.first_alarm is not None and observation.day - self.first_alarm in {0, 5, 10}
                and observation.remaining_budget >= 4):
            return RegulatoryAction(RegulatoryActionType.FREEZE, self.target)
        return RegulatoryAction(RegulatoryActionType.WAIT)
