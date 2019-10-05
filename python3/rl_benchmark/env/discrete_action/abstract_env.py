class AbstractEnvironment(object):
    """
    Abstract Class For Environment Classes
    """
    def get_state(self, setting = None):
        """
        Get Current State
        Args:
        setting : dictionary
        setting = setting for states
        """
        raise NotImplementedError

    def apply_action(self, action, num_repeat):
        """
        Apply Actions To The Environment And Get Reward 
        """
        raise NotImplementedError

    def new_episode(self):
        """
        Start A New Episode
        """
        raise NotImplementedError

    def episode_end(self):
        """
        Check If The Episode Ends
        """
        raise NotImplementedError

    def action_set(self):
        """
        Get Actions Set
        """
        raise NotImplementedError

    def available_action(self):
        """
        Get Indices of Available Actions For Current State
        """
        raise NotImplementedError

    def episode_total_score(self):
        """
        Get Total Score For Last Episode
        """
        raise NotImplementedError

    def close(self):
        """
        Close The Environment
        """
        raise NotImplementedError

