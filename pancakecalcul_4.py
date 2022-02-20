import numpy as np
#from tensorflow.python.distribute.device_util import current

from data_class import pdata, cake, point
from lineicircle import lineicircle
from chordlength import chordlength
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot


def pancakecalcul_4(x, fx):
    """
    Return: pancakes, success
    calculates density pancakes according to phase fx in x points
    called from phasecontrol
    x           - array of elements in ascending order;
    fx          - array same length with x, zeros on each ends (or smallvalue*max(fx))

    denspancake - array of records: thick, radius, center
    success     - indicator: 1 if ok, 0 - otherwise
    each time next pancake calculates by edge point (it can
    be negative or positive)
    """
    global SMALLVALUE, DENSCOEFF, SMALLX, SMALL_FX
    SMALLVALUE = 1E-4
    LAMBDA = 0.22  # 2.15E-1; wavelength of the interferometer
    ELECTRONCHARGE = 4.8027E-10
    LIGHTVELOS = 2.9979E10
    ELECTRONMASS = 9.1083E-28
    OMEGA = 2 * np.pi * LIGHTVELOS / LAMBDA
    DENSCOEFF = (ELECTRONCHARGE) ** 2 / OMEGA / LIGHTVELOS / ELECTRONMASS
    SMALLX = max(x) * SMALLVALUE
    SMALL_FX = max(fx) * SMALLVALUE


    isplotting = 1  #1 - for plot, 0 to supress plotting
    success = 1

    ind = np.argsort(x)
    tempx = np.sort(x)
    #tempfx = fx
    tempfx = np.zeros_like(fx)

    i = 0
    for arg in ind:
        tempfx[i] = fx[arg]
        i += 1
    del i

    points = point(tempx, tempfx)

    maxfx = max(abs(points.fx))
    if (abs(points.fx[0]) > SMALL_FX) or (abs(points.fx[-1]) > SMALL_FX):
        'one of edges is non zero'
        success = 0
        return None, success
    current_points_number = len(points.x)
    current_pancake_number = -1 #   mb = 0
    pancakes = cake()

    while success == 1 and current_points_number >= 1:
        current_pancake_number = current_pancake_number + 1
        pancakes, success, left_bound, right_bound, bound_flag = make_cake(points, pancakes, current_pancake_number)

        #Here some non informative comments from .m code...

        if success == 1:
            if left_bound == len(points) - 1 and right_bound == 0:  # mb right_bound == 1
                current_pancake_number = current_pancake_number - 1
                #pancakes = pancakes(1:current_pancake_number)
                #pancakes.pop()
                new_points_number = 0
            else:
                points, success = apply_pancake(points, pancakes, current_pancake_number, left_bound,
                                                right_bound, bound_flag)
                if success == 1:
                    new_points_number = right_bound - left_bound + 1  # reindex for python ?
                    if new_points_number >= current_points_number:
                        'new nonzero segment not smaller than previous'
                        print(current_pancake_number)
                        success = 0
            current_points_number = new_points_number

    if isplotting == 1:
        plotpancake(x, fx, pancakes)

    return pancakes, success


def make_cake(points, pancakes, current_pancake_number):
    """
    make one cake with according to first or last nonzero point
    bound_flag = 'l' or 'r' according to which bound will be zero
    """

    success = 1

    left_bound = boundary_search(points, 'l')
    right_bound = boundary_search(points, 'r')
    bound_flag = []

    if left_bound == len(points.x)-1 and right_bound == 0:    #   mb right_bound == 1
        #pancakes.radius[current_pancake_number-1] = None
        #bound_flag = []
        return pancakes, success, left_bound, right_bound, bound_flag
    if left_bound > right_bound:
        success = 0
        return pancakes, success, left_bound, right_bound, bound_flag

    current_pancake = cake((points.x[right_bound + 1] - points.x[left_bound - 1]) / 2,
                           (points.x[right_bound + 1] + points.x[left_bound - 1]) / 2, 0)

    left_thick = points.fx[left_bound] / chordlength(current_pancake.radius, current_pancake.center,
                                                     points.x[left_bound])
    right_thick = points.fx[right_bound] / chordlength(current_pancake.radius, current_pancake.center,
                                                       points.x[right_bound])
    current_pancake.thick = left_thick
    bound_flag = 'l'
    if abs(right_thick) < abs(left_thick):
        current_pancake.thick = right_thick
        bound_flag = 'r'
    current_pancake.thick = current_pancake.thick / DENSCOEFF
    pancakes += current_pancake
    a = 1
    #pancakes.radius.append(current_pancake.radius)
    return pancakes, success, left_bound, right_bound, bound_flag


def boundary_search(points, flag):
    """
    searching bound flag='l' or 'r' for left and right bound correspondingly
    """
    tmp_array = (abs(points.fx) > SMALL_FX)
    bound = []

    if flag == 'l':
        start = 0
        step = 1
    elif flag == 'r':
        start = len(points.x)-1
        step = -1
    else:
        return bound
    finish = start + step * (len(points.x) - 1)
    ind = start
    while len(bound) == 0 and ind != finish-step:
        new_ind = ind + step
        if tmp_array[new_ind] > tmp_array[ind]:
            bound.append(new_ind)
        ind = new_ind
    if len(bound) == 0:
        bound.append(finish)
    return bound[0]


def apply_pancake(points, pancakes, current_pancake_number,
                  left_bound, right_bound, bound_flag):

    r = pancakes.radius[current_pancake_number]
    c = pancakes.center[current_pancake_number]
    thick = pancakes.thick[current_pancake_number]

    #   another edge checking
    if abs(right_bound - left_bound) > 1:
        if bound_flag == 'l':
            edge_point_index = right_bound + 1
            neighbour_point_index = right_bound
        else:
            edge_point_index = left_bound - 1
            neighbour_point_index = left_bound

        edge_point = point(points.x[edge_point_index], points.fx[edge_point_index])
        neighbour_point = point(points.x[neighbour_point_index], points.fx[neighbour_point_index])
        points.x[edge_point_index], points.fx[edge_point_index] = check_neighbour_point(edge_point, neighbour_point, r,
                                                                                        c, thick)
    success = 1
    counter = 0

    for ind in range(len(points)):
        if (points.x[ind] > c - r) and (points.x[ind] < c + r):
            counter = counter + 1
            points.fx[ind] = points.fx[ind] - thick * DENSCOEFF * chordlength(r, c, points.x[ind])

    if counter == 0:
        success = 0

    return points, success


def check_neighbour_point(edge_point, neighbour_point, r, c, thick):

    xint, yint, success = lineicircle(edge_point.x, edge_point.fx / DENSCOEFF / abs(thick),
                                      neighbour_point.x, neighbour_point.fx / DENSCOEFF / abs(thick),
                                      r, c, np.sign(thick))

    if success and np.sign((xint - edge_point.x) * (xint - neighbour_point.x)) == -1:
        edge_point.x = xint
        edge_point.fx = yint * DENSCOEFF * abs(thick)
    return edge_point.x, edge_point.fx


def plotpancake(x, fx, denspancake):

    plt.figure(1)
    plot(x, fx, 'b.')
    yin = np.array(fx)
    yout = list()
    for i in range(len(denspancake.radius)):
        for j in range(len(x)):
            yout.append(denspancake.thick[i] * DENSCOEFF *
                        chordlength(denspancake.radius[i], denspancake.center[i], x[j]))
            if yout[j] != 0:
                yout[j] = yin[j] - yout[j]
        plt.figure(2)
        plot(x, yout, 'r-')

        ##if i == len(denspancake.radius)-10:
        ##    plt.figure(2)
        ##    plot(x, yout, 'r-')

        yin = np.array(yout)
        yout.clear()
    plt.show()
