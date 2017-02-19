import numpy as np

def period_with_error(low, best, high, latex=False):
    """
    Convert a given period range to an interval with correct significant digits.
    If either `low` or `high` is less than 0 or `high` > 2*`best`, return best in "x.xxxx:" format.

    Parameters
    ----------
    low:
        lower limit.
    best:
        best value.
    high:
        upper limit.
    latex: bool, optional
        if True, return a string ready for LaTeX math mode.

    Returns
    -------
    result: str
    """
    lowdiff = best - low
    highdiff = high - best

    if low <= 0. or high <= 0. or highdiff > best:
        if latex:
            return r"$"+format(best, 'f')[:4]+r":$"
        else:
            return format(best, 'f')[:4]+":"

    loworder =  int(np.floor(np.log10( lowdiff)))
    highorder = int(np.floor(np.log10(highdiff)))

    lowestorder = -min(loworder, highorder)
    p_round = round(best,     lowestorder)
    l_round = round(lowdiff,  lowestorder)
    h_round = round(highdiff, lowestorder)

    percstr = "%0."+str(lowestorder)+"f"
    p_str = percstr % p_round
    l_str = percstr % l_round
    h_str = percstr % h_round

    if l_str == h_str:
        if latex:
            return r"$"+p_str+r" \pm "+l_str+r"$"
        else:
            return p_str+" +- "+l_str
    else:
        if latex:
            return r"$"+p_str+r"^{+"+h_str+r"}_{-"+l_str+"}$"
        else:
            return p_str+" +"+h_str+" -"+l_str

class Star(object):
    def __init__(self, ascc, ra, dec, spectype, B, V):
        self.ascc = int(ascc)
        self.ra = ra
        self.dec = dec
        self.spectype = spectype
        self.B = B
        self.V = V
    def __repr__(self):
        return "Star ASCC {0} at coordinates ({1:.2f}; {2:.2f}); spectral type {3}; B = {4} and V = {5}".format(self.ascc, self.ra, self.dec, self.spectype, self.B, self.V)