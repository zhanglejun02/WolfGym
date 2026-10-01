"""Public campaign control with fresh targeting on each one-day action."""
from wolfgym.evaluation.prevention_policy import PublicCampaignFreeze
from wolfgym.regulation import RegulatoryAction, RegulatoryActionType


class TrackingCampaignFreeze:
    def __init__(self, alarm_threshold, delay=0):
        if delay not in (0, 4):
            raise ValueError('only the two predeclared development candidates are allowed')
        self.alarm_threshold = alarm_threshold
        self.delay = delay
        self.first_alarm = None

    def __call__(self, observation, score):
        if observation.messages is None:
            raise ValueError('tracking requires the existing public message view')
        if self.first_alarm is None and score > self.alarm_threshold:
            self.first_alarm = observation.day
        if (self.first_alarm is None or observation.day - self.first_alarm - self.delay not in (0, 5, 10)
                or observation.remaining_budget < 4):
            return RegulatoryAction(RegulatoryActionType.WAIT)
        # A fresh instance reuses exactly the validated public targeting rule.
        # The synthetic trigger only requests that rule at this scheduled action;
        # it is not a new detector output or a hidden-information input.
        targeter = PublicCampaignFreeze(self.alarm_threshold)
        return targeter(observation, 1.)
