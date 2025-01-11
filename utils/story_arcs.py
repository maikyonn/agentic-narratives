import numpy as np
import matplotlib.pyplot as plt

def warp_domain(t, warp_strength=0.0):
    """
    Warps the domain array t by adding small random (or sinusoidal) distortions.
    The stronger the warp_strength, the more it skews the curve.
    
    This particular implementation:
      1) Adds a small sinusoidal component: warp_strength * 0.2 * sin(2*pi * t)
      2) Adds a small uniform random offset: warp_strength * uniform(-0.05, 0.05)
      3) Clamps the result to [0, 1]

    Returns a new array t_warped of the same length as t.
    """
    # Make a copy so as not to alter original t in-place
    t_warped = t.copy()
    
    # Add a small sinusoidal distortion
    t_warped += warp_strength * 0.2 * np.sin(2 * np.pi * t_warped)
    
    # Add a small uniform random offset
    random_offsets = warp_strength * np.random.uniform(-0.05, 0.05, size=len(t_warped))
    t_warped += random_offsets
    
    # Clamp to [0, 1]
    t_warped = np.clip(t_warped, 0, 1)
    
    # We do NOT sort t_warped here, so the curve can fold back a bit on itself.
    # If you'd like to preserve the strictly increasing order, you could sort;
    # that, however, changes the indexing of points.
    
    return t_warped

def rags_to_riches(n=20, warp_strength=0.0):
    """
    Starts near -1 and ends near +1 with a smooth rise.
    By default, uses -cos(pi*t).
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = -np.cos(np.pi * t2)
    return y

def riches_to_rags(n=20, warp_strength=0.0):
    """
    Starts near +1 and ends near -1 with a smooth decline.
    By default, uses cos(pi*t).
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = np.cos(np.pi * t2)
    return y

def man_in_a_hole(n=20, warp_strength=0.0):
    """
    One full cosine wave: cos(2*pi * t).
    Starts at +1, dips to -1 around t=0.5, then returns to +1.
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = np.cos(2 * np.pi * t2)
    return y

def double_man_in_a_hole(n=20, warp_strength=0.0):
    """
    Two full cosine waves: cos(4*pi * t).
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = np.cos(4 * np.pi * t2)
    return y

def icarus(n=20, warp_strength=0.0):
    """
    Piecewise linear:
      - t in [0..0.6] -> -1 -> +1
      - t in [0.6..1] -> +1 -> -1
    Then warp the domain if warp_strength>0 to skew it.
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = []
    for x in t2:
        if x <= 0.6:
            # -1 -> +1 from 0..0.6
            val = -1 + (x / 0.6) * 2
        else:
            # +1 -> -1 from 0.6..1
            # slope = (-2) / 0.4 = -5
            val = 1 - ( (x - 0.6) * 5 )
        y.append(val)
    return y

def cinderella(n=20, warp_strength=0.0):
    """
    Piecewise linear:
      - t in [0..0.3]: -1 -> +1
      - t in [0.3..0.6]: +1 -> -1
      - t in [0.6..1]: -1 -> +0.8
    Then warp the domain if warp_strength>0.
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = []
    for x in t2:
        if x <= 0.3:
            val = -1 + (x / 0.3) * 2
        elif x <= 0.6:
            val = 1 - ((x - 0.3) / 0.3) * 2
        else:
            val = -1 + ((x - 0.6) / 0.4) * 1.8
        y.append(val)
    return y

def oedipus(n=20, warp_strength=0.0):
    """
    Piecewise linear:
      - fall (+1 to -1),
      - rise (-1 to +1),
      - final fall (+1 to -1).
    Then warp the domain if warp_strength>0.
    """
    t = np.linspace(0, 1, n)
    t2 = warp_domain(t, warp_strength=warp_strength)
    y = []
    for x in t2:
        if x <= 0.3:
            val = 1 - (x / 0.3) * 2
        elif x <= 0.6:
            val = -1 + ((x - 0.3) / 0.3) * 2
        else:
            val = 1 - ((x - 0.6) / 0.4) * 2
        y.append(val)
    return y