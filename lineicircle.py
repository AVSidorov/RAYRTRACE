import numpy as np


def lineicircle(x1, y1, x2, y2, r, c, circle_sign):
    """
    calculate intersection point x,y of line (based on two points) and circle
    with radii r and center c (y=circle_sign*2*(r*r - (x-c).^2).^(1/2)
    circle_sign - upper or lower part of circle, 2 for making really chord length)
    """
    success = 1
    #   constructing line
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1

    if ((4 * c - a * b) ** 2 - (a * a + 4) * (b * b + 4 * c * c - 4 * r * r)) < 0:
        success = 0
        xint = 0
        yint = 0
        return xint, yint, success
    if a * circle_sign < 0:
        xint = (4 * c - a * b - ((4 * c - a * b) ** 2 - (a * a + 4) * (b * b + 4 * c * c - 4 * r * r)) ** (1 / 2)) / (
                    a * a + 4)
    else:
        xint = (4 * c - a * b + ((4 * c - a * b) ** 2 - (a * a + 4) * (b * b + 4 * c * c - 4 * r * r)) ** (1 / 2)) / (
                    a * a + 4)

    yint = a * xint + b
    if np.sign(yint * circle_sign * 2 * (r * r - (xint - c) ** 2) ** (1 / 2)) == -1:
        success = 0
        xint = 0
        yint = 0
    return xint, yint, success
