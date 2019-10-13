import numpy as np

def linear_decay(current_cnt, begin_cnt, end_cnt, begin_prob, end_prob):
    """
    Linear Decay Probability
    Args:
    current_cnt : int
    current_cnt = current counter (e.g. iteration or episode)
    begin_cnt : int
    begin_cnt = begin counter
    end_cnt : int or None
    end_cnt = end counter
    begin_prob : float
    begin_prob = begin probability
    end_prob : float or None
    end_prob = end probability
    Returns:
    current_prob : float
    current_prob = current probability
    """
    if (end_cnt is None or end_prob is None or end_cnt == begin_cnt):
        return begin_prob
    ratio = (current_cnt - begin_cnt) / (end_cnt - begin_cnt)
    ratio = min(max(ratio, 0), 1)
    current_prob = begin_prob + (end_prob - begin_prob) * ratio
    return current_prob

