import numpy as np
from svg.path import *
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
import xml.etree.ElementTree as ET
import re
import sys

# %%

'''
Draws Fourier Epicycle animation for input svg line art.

Usage:
    $ Epicycle_Draw_Script in N, out

Arguments:
    > in: input svg file
    > out: name of output file (optional)
    > N: int number of points to sample along Bezier curves (default 100)
'''

# get input file name
filename = sys.argv[1]

# get output file name
if len(sys.argv) > 2:
    out = sys.argv[2]
else:
    out='.'.join([filename[:-4], 'mp4'])


# no. points to sample along each bezier curve
if len(sys.argv) > 3:
    N = sys.argv[3]
else:
    N = 100


# list of bezier curves
cur = list()

# import svg data
tree = ET.parse(filename)
root = tree.getroot()

# get dimensions for centering
h = float(re.sub('[^0-9]', '', root.attrib["height"])) # remove unit and convert to float
w = float(re.sub('[^0-9]', '', root.attrib["width"]))

# extract paths
for child in root.findall(".//{http://www.w3.org/2000/svg}g"): # find g tags
    for p in child.findall(".//{http://www.w3.org/2000/svg}path"): # get paths
        tp = parse_path(p.attrib["d"]) # parse paths
        dc = list() # instantiate list of discrete points for sampling

        # discretise path
        for i in range(0, N+1):
            dc.append(tp.point(i/N) - w/2 - (h/2)*1j) # discretise and recenter

        # convert discrete curve to np array and append to cur
        cur.append(np.array(dc).conj())

# %%

# get params for DFT and epicycle drawing
def compute_epi(Z, N):
    DFT = np.fft.fft(Z, n=N)/N # DFT of data
    k = np.arange(0, N) # circle frequencies
    r = np.abs(DFT) # circle radii
    phase = np.angle(DFT) # initial circle phases

    # sort by descending radius
    idx = r.argsort(kind='stable') # indices if r sorted ascending
    r = r[idx][::-1] # sort r and reverse
    k = k[idx][::-1] # sort k and reverse
    phase = phase[idx][::-1] #sort and reverse
    DFT = DFT[idx][::-1] # sort and reverse

    return { 'DFT': DFT, 'radii': r, 'frequency': k, 'phase': phase }


# draw epicycles and line
def draw_epi(k, A, time, phase, wavex, wavey):

    # compute coords
    x = 0
    y = 0
    N = k.shape[0]
    centres = np.zeros((N, 2))
    lines = np.zeros((N, 4))

    for i in range(0, N):
        # store previous coords for new circle centre
        px = x
        py = y

        # get new joint point coords
        x += A[i] * np.cos(k[i]*time + phase[i])
        y += A[i] * np.sin(k[i]*time + phase[i])

        # circle centres
        centres[i, 0] = px
        centres[i, 1] = py

        # radii lines
        lines[i, :] = np.array([px, x, py, y])

    ax.clear()

    circles = list()

    # update resultant waveform
    wavex.append(x)
    wavey.append(y)

    # plot
    for i in range(N):
        ax.add_artist(Circle((centres[i, 0], centres[i, 1]), radius=A[i], color='k', fill=False))

        # joining lines
        plt.plot(lines[i, 0:2], lines[i, 2:4], color='k')

        # pointer
        plt.plot(x, y, 'or')

        # plot resultant waveform
        if len(wavex) > 0:
            plt.plot(wavex, wavey, '-b')

    return [wavex, wavey]


# %%

# matplotlib animation lib method
comp = compute_epi(cur[0], N)

ts = 2*np.pi/comp['DFT'].shape[0]

# draw result
time = 0

wavex = list()
wavey = list()

fig = plt.figure()
ax = plt.axes()

xp = list() # previous x
yp = list()

def drawnext(i):



    global wavex, wavey, comp, xp, yp

    if i%N==0:
        comp = compute_epi(cur[int(np.floor(i/(N)))], N)
        print(f'{i*100/(len(child.findall(".//{http://www.w3.org/2000/svg}path"))*(N))}% Complete!')
        xp.append(wavex.copy())
        yp.append(wavey.copy())
        wavex = list()
        wavey=list()


    a = draw_epi(comp['frequency'], comp['radii'], i*2*np.pi/N, comp['phase'], wavex, wavey)

    for i, prev in enumerate(xp):
        plt.plot(prev, yp[i], '-b')

    wavex = a[0]
    wavey = a[1]

def init_func():
    global h, w
    ax.axis([-w/2, w/2, -h/2, h/2])
    plt.axis('scaled')
    plt.xlabel('Dalgarnitude')
    plt.ylabel('Paulness')
    plt.title('Fourier Series Drawing of Dr. Paul Andrew Dalgarno')


ani = animation.FuncAnimation(
    fig, drawnext, init_func=init_func, interval=100/2.4, frames=range(len(child.findall(".//{http://www.w3.org/2000/svg}path"))*(N)), repeat=False)

plt.show()

ani.save(out)
