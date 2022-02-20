
def chordlength(radius, center, point):
    """
    calculate vertical chord length of circle (radius, center) at x=point
    """
    global SMALLVALUE
    SMALLVALUE = 1E-4

    smallvalue = SMALLVALUE * radius
    if center - radius + smallvalue < point < center + radius - smallvalue:
        chordlen = 2 * (radius ** 2 - (center - point) ** 2) ** (1/2)
    else:
        chordlen = 0
    return chordlen
