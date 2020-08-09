import numpy as np
from svg.path import *
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

%matplotlib qt5

# %%

global wavex, wavey
# head shape curve
cur = parse_path("m 94.494047,208.55357 c 10.362693,-2.93893 18.903243,-7.88223 25.702383,-14.74107 8.33601,-9.53107 15.46412,-19.81706 18.14285,-32.88393 1.75796,3.22836 3.11941,3.28461 4.53572,3.77976 4.76742,-2.33283 7.92752,-6.96182 10.20536,-12.85119 2.45906,-6.00773 1.80505,-11.73245 0.37797,-17.3869 2.76964,-9.61765 -6.96626,-8.72973 -6.42559,-5.66965 -3.05793,1.66602 -3.93669,6.13384 -4.53572,10.96131 -5.4594,-6.58769 -2.40832,-16.82274 -3.40178,-25.3244 C 134.68552,77.012113 130.27579,80.593446 125.86607,75.505952 117.7868,66.938707 106.79962,75.645663 97.13988,77.193413 90.840277,73.886957 84.540675,72.687434 78.241072,73.61607 66.812191,78.547959 62.847348,87.211868 56.696427,94.782737 c -1.691528,7.937503 -0.715052,15.875003 0,23.812503 -0.334175,6.84737 -1.226644,13.13645 -2.267855,19.27678 0.06727,6.18749 2.266901,8.82104 3.401785,13.22917 -1.300622,6.67813 1.357057,15.05268 2.267857,22.67857 9.90308,20.18336 20.940088,34.13011 34.395833,34.77381 z"
)

# discretise curve

# N = 120 # no. points
N = 100

plist = list() # list of points

for i in range(0, N+1):
    plist.append(cur.point(i/N))

parr = np.array(plist)

parr -= 100 + 140*1j

parr = parr.conj()

plt.plot(np.real(parr), np.imag(parr))

# %%

# hair
cur1 = parse_path("m 138.33928,160.92857 c -0.19555,5.22535 -1.83242,11.71877 0.70769,12.192 10.52332,6.91743 10.44506,-7.80199 15.43481,-11.62622 4.3654,-11.75986 15.87527,-29.77113 13.09621,-35.27958 -2.46859,-14.16196 -3.61272,-19.05278 -5.07812,-26.19241 C 159.20355,90.489746 159.46859,83.727082 152.6109,71.424519 143.86775,65.899001 141.48591,57.801273 133.36749,55.388345 115.91201,49.762198 108.13035,50.065168 101.56242,51.112033 86.526787,50.183427 74.87661,54.967778 62.54107,58.595579 48.801618,76.695016 47.646943,85.874741 48.910324,93.34062 c 1.747439,10.50303 -2.653522,17.56015 2.939963,28.33057 0.607172,1.67625 1.800919,3.93907 4.009043,7.21627"
)

plist1 = list() # list of points

for i in range(0, N+1):
    plist1.append(cur1.point(i/N))

parr1 = np.array(plist1)

parr1 -= 100 + 140*1j

parr1 = parr1.conj()

plt.plot(np.real(parr1), np.imag(parr1))

# %%

# brow-nose
cur2 = parse_path("m 86.060787,122.60663 c -0.658478,-1.37487 1.435197,-2.44394 -2.138156,-4.14268 -9.24983,-0.89176 -16.719111,-0.48858 -22.049737,1.46999 -0.557348,0.70131 -0.689284,1.61533 -0.534538,2.67269 8.716727,-0.86666 15.428379,-0.39661 22.049734,0.13364 5.579252,1.11501 2.739512,5.38715 3.207235,8.41899 -1.158168,3.64453 -2.316335,5.26578 -3.474503,10.95805 -0.524451,2.92315 -1.291865,6.25123 -0.80181,7.48354 -2.498857,-0.1635 -3.411272,2.44927 -4.543581,4.67722 -0.222199,2.86485 0.810973,3.63741 1.603619,4.81085 1.648167,-0.4009 2.22724,-0.97999 1.737251,-1.73725 2.47652,-0.39942 3.55352,3.32534 7.617182,3.87541 l 4.276313,-3.47451 c 3.913167,-1.297 4.391334,0.41163 5.612659,1.46999 3.466845,-0.25907 2.539395,-1.49466 3.073595,-2.40543 -0.1221,-2.25822 0.51226,-5.17751 -2.93996,-7.88445 -2.475414,-8.41506 -2.057534,-13.41371 -4.009043,-19.24341 0.657207,-2.21922 0.488099,-4.79257 3.474503,-6.01356 6.90299,-2.06572 16.23685,-1.16037 26.32605,0.66817 1.04872,-1.1798 1.07951,-2.69891 -1.06908,-4.94448 -8.59717,-3.20854 -17.19434,-1.2381 -25.791508,-1.60362 -1.457905,1.19807 -2.481601,2.45818 -1.469982,4.00904"
)

plist2 = list() # list of points

for i in range(0, N+1):
    plist2.append(cur2.point(i/N))

parr2 = np.array(plist2)

parr2 -= 100 + 140*1j

parr2 = parr2.conj()

plt.plot(np.real(parr2), np.imag(parr2))

# %%

# lips
cur3 = parse_path("m 76.572718,174.72419 c 6.21583,-4.68084 8.721598,-1.94156 12.294398,-1.33635 6.204403,-0.60018 11.414814,0.45628 16.570704,1.60362 -3.64944,2.48515 -11.413445,5.65706 -11.759851,5.61266 -13.998842,0.59297 -16.879525,-2.23502 -17.105251,-5.87993 z"
)

plist3 = list() # list of points

for i in range(0, N+1):
    plist3.append(cur3.point(i/N))

parr3 = np.array(plist3)

parr3 -= 100 + 140*1j

parr3 = parr3.conj()

plt.plot(np.real(parr3), np.imag(parr3))

# %%

# lip line
cur9 = parse_path("m 76.572718,174.72419 c 4.488552,0.14806 6.367555,-0.42611 13.29666,1.5368 -0.287399,-0.86208 9.133566,-1.05288 15.568442,-1.26953"
)

plist9 = list() # list of points

for i in range(0, N+1):
    plist9.append(cur9.point(i/N))

parr9 = np.array(plist9)

parr9 -= 100 + 140*1j

parr9 = parr9.conj()

plt.plot(np.real(parr9), np.imag(parr9))
# %%

# eye 1
cur4 = parse_path("m 64.54559,128.75383 c 5.347746,-3.97843 10.913642,-5.12091 16.971616,0.13363 -5.657205,1.05582 -11.314411,1.63182 -16.971616,-0.13363 z"
)

plist4 = list() # list of points

for i in range(0, N+1):
    plist4.append(cur4.point(i/N))

parr4 = np.array(plist4)

parr4 -= 100 + 140*1j

parr4 = parr4.conj()

plt.plot(np.real(parr4), np.imag(parr4))

# %%

# eye 2
cur5 = parse_path("m 103.0324,128.88746 c 3.20541,-2.11252 5.16328,-5.93219 14.83346,0.80181 -4.94449,1.51376 -9.88897,1.59777 -14.83346,-0.80181 z"
)

plist5 = list() # list of points

for i in range(0, N+1):
    plist5.append(cur5.point(i/N))

parr5 = np.array(plist5)

parr5 -= 100 + 140*1j

parr5 = parr5.conj()

plt.plot(np.real(parr5), np.imag(parr5))

# %%

# wrinkle 1
cur6 = parse_path("m 104.7939,161.40104 c 5.87219,9.86206 5.45336,9.79091 6.89806,12.6622")

plist6 = list() # list of points

for i in range(0, N+1):
    plist6.append(cur6.point(i/N))

parr6 = np.array(plist6)

parr6 -= 100 + 140*1j

parr6 = parr6.conj()

plt.plot(np.real(parr6), np.imag(parr6))

# %%

# wrinkle 2
cur7 = parse_path("m 75.784225,158.8497 c -2.557339,5.76414 -3.800842,11.52827 -4.346726,17.29241")

plist7 = list() # list of points

for i in range(0, N+1):
    plist7.append(cur7.point(i/N))

parr7 = np.array(plist7)

parr7 -= 100 + 140*1j

parr7 = parr7.conj()

plt.plot(np.real(parr7), np.imag(parr7))

# %%

# chin
cur8 = parse_path("m 83.249256,203.45089 c 12.860143,4.00135 23.442994,2.48465 29.765624,-9.35491")

plist8 = list() # list of points

for i in range(0, N+1):
    plist8.append(cur8.point(i/N))

parr8 = np.array(plist8)

parr8 -= 100 + 140*1j

parr8 = parr8.conj()

plt.plot(np.real(parr8), np.imag(parr8))


# %%

# # eye 1
# cur5 = parse_path("m 110.41629,84.152157 c -7.2911,-6.71015 -13.762603,-7.191335 -19.796499,-4.346726 4.361785,6.756624 13.516719,8.179744 19.796499,4.346726 z"
# )
#
# plist5 = list() # list of points
#
# for i in range(0, N):
#     plist5.append(cur5.point(i/N))
#
# parr5 = np.array(plist5)
#
# plt.plot(np.real(parr5), -np.imag(parr5))

# %%

# # test
#
# testy = np.fft.fft(parr)/N
#
# zd = list()
#
# # sum component waves
# for n in range (0, N):
#
#     zt = 0.0
#
#     for k in range(int(-N/2 + 1), int(N/2)):
#         zt += testy[k] * np.exp(1j*n*k*(2*np.pi/N))
#
#     zd.append(zt)
#
# zd = np.array(zd)
#
# %matplotlib qt5
#
# fig = plt.figure()
# ax = plt.axes()
# ax.axis([-60, 60, -80, 80])
# plt.axis('scaled')
#
# for i, elem in enumerate(zd):
#     ax.clear()
#     ax.plot(zd[0:i+1].real, -zd[0:i+1].imag, '-b')
#     plt.pause(0.001)
#


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
#
# # crude animation method
#
# comp = compute_epi(parr, N)
#
# ts = 2*np.pi/comp['DFT'].shape[0]
#
# # draw result
# time = 0
#
# wavex = list()
# wavey = list()
#
# fig = plt.figure()
# ax = plt.axes()
#
# for i in range(N+1):
#     a = draw_epi(comp['frequency'], comp['radii'], time, comp['phase'], wavex, wavey)
#
#     wavex = a[0]
#     wavey = a[1]
#
#     time += ts
#
#     plt.pause(0.0000001)


# %%

# matplotlib animation lib method

comp = compute_epi(parr, N)

ts = 2*np.pi/comp['DFT'].shape[0]

# draw result
time = 0

wavex = list()
wavey = list()

fig = plt.figure()
ax = plt.axes()


def drawnext(i):

    # if not 'wavex' in locals():
    #     wavex = list()
    #     wavey = list()


    global wavex, wavey, comp, x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9

    if i == 0:
        comp = compute_epi(parr, N)
        wavex = list()
        wavey = list()
    elif i == N+1:
        x1=wavex.copy()
        y1=wavey.copy()
        comp = compute_epi(parr1, N)
        wavex = list()
        wavey = list()
    elif i == 2*N:
        x2=wavex.copy()
        y2=wavey.copy()
        comp = compute_epi(parr2, N)
        wavex = list()
        wavey = list()
    elif i == 3*N:
        x3=wavex.copy()
        y3=wavey.copy()
        comp = compute_epi(parr3, N)
        wavex = list()
        wavey = list()
    elif i == 4*N:
        x9=wavex.copy()
        y9=wavey.copy()
        comp = compute_epi(parr9, N)
        wavex = list()
        wavey = list()
    elif i == 5*N:
        x4=wavex.copy()
        y4=wavey.copy()
        comp = compute_epi(parr4, N)
        wavex = list()
        wavey = list()
    elif i == 6*N:
        x5=wavex.copy()
        y5=wavey.copy()
        comp = compute_epi(parr5, N)
        wavex = list()
        wavey = list()
    elif i == 7*N:
        x6=wavex.copy()
        y6=wavey.copy()
        comp = compute_epi(parr6, N)
        wavex = list()
        wavey = list()
    elif i == 8*N:
        x7=wavex.copy()
        y7=wavey.copy()
        comp = compute_epi(parr7, N)
        wavex = list()
        wavey = list()
    elif i == 9*N:
        x8=wavex.copy()
        y8=wavey.copy()
        comp = compute_epi(parr8, N)
        wavex = list()
        wavey = list()

    a = draw_epi(comp['frequency'], comp['radii'], i*2*np.pi/N, comp['phase'], wavex, wavey)

    if 'x1' in globals():
        plt.plot(x1, y1, '-b')
    if 'x2' in globals():
        plt.plot(x2, y2, '-b')
    if 'x3' in globals():
        plt.plot(x3, y3, '-b')
    if 'x4' in globals():
        plt.plot(x4, y4, '-b')
    if 'x5' in globals():
        plt.plot(x5, y5, '-b')
    if 'x6' in globals():
        plt.plot(x6, y6, '-b')
    if 'x7' in globals():
        plt.plot(x7, y7, '-b')
    if 'x8' in globals():
        plt.plot(x8, y8, '-b')
    if 'x9' in globals():
        plt.plot(x9, y9, '-b')

    wavex = a[0]
    wavey = a[1]

def init_func():
    ax.axis([-60, 60, -80, 80])
    plt.axis('scaled')
    plt.xlabel('Dalgarnitude')
    plt.ylabel('Paulness')
    plt.title('Fourier Series Drawing of Dr. Paul Andrew Dalgarno')


ani = animation.FuncAnimation(
    fig, drawnext, init_func=init_func, interval=100/2.4, frames=range(9*(N+1)), repeat=False)

plt.show()

ani.save('/home/sean/Videos/Nic_Cage_Epi_1.mp4')
