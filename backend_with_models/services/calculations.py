import math

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a, b, c.
    Points are tuples: (x, y, z).
    """
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c

    # Calculate vectors
    ab = (ax - bx, ay - by, az - bz)
    bc = (cx - bx, cy - by, cz - bz)

    # Dot product and magnitudes
    dot_product = sum(ab[i] * bc[i] for i in range(3))
    magnitude_ab = math.sqrt(sum(ab[i] ** 2 for i in range(3)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(3)))

    # Prevent division by zero
    if magnitude_ab * magnitude_bc == 0:
        return 0

    # Calculate angle in degrees
    angle = math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))
    return angle