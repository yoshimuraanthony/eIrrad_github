from scipy.optimize import curve_fit
from numpy import sqrt, array, where, linspace, pi, exp
import matplotlib.pyplot as plt
from periodic import p_dict

import calcDispCross as cc


m = 5.109989461e5  # mass of electron (eV)
c = 299792458  # speed of light (m/s)
alpha = 1/137.035999084  # fine structure constant (unitless)
hbar = 6.582e-16  # Planck constant (eV s)
invCmtoEV = 100*c*hbar
invÅtoEV = 1e10*c*hbar
invÅtomeV = 1e10*c*hbar*1000
invEVSqtoÅSq = hbar**2*c**2*1e20
scale1 = 1e28 * c**2  # s^2 to 100 fm^2 (barn)
const = pi*hbar**2*alpha**2 * scale1  # prefactor to cross sections
mp = 9.3827231e8  # mass of proton (eV)
MB = 10.811 * mp
MS = 32.06 * mp

# nuclear masses of sputtered atoms
M_dict = {'hBN': MB, 'MoS2': MS}

# vibrational frequencies of A_1' mode in inverse cm (DOI: 10.1038/srep04215)
omega_dict = {
        'hBN': 1378.4, # TO mode in hBN DOI: 10.1103/PhysRevLett.98.095503
        'MoS2': 406.1,
        }

# exp(inverse) fit
fit_dict = {
        'hBN': [5.462949624, -0.8497685369, 0.08390400208],
        'MoS2': [41.18955911, -0.9537691654, 0.5587624902]
        }

# displacement thresholds in eV (formation energies using optB88)
Ed_dict = {
        'hBN': [15.673545530, 11.319425210, 6.602449050, 1.820544710],
        'MoS2': [6.77709924, 5.07687782, 3.3765814, 1.70605652],
        }

# species bring sputtered
spec_dict = {'hBN': 'B', 'MoS2': 'S'}


def hBNPoints(Eb, D0, v0):
    A, B, C = fit_dict['hBN']
    Ed_list = Ed_dict['hBN']  # eV
    omega = omega_dict['hBN'] * invCmtoEV  # eV
    spec = spec_dict['hBN']
    Z = p_dict['B'][0]  # atomic number
    M = p_dict['B'][1] * mp  # mass (eV)

    D = D0*(1 - exp(-v0/sqrt(Eb)))
    P = D*(1 - exp(-A/(Eb - B) - C))
    S_list = []
    for Ed in Ed_list:
        Ec = sqrt(m**2 + M*Ed/2) - m
        S = cc.getAvgCross(Eb, Ec, Ed)
        S_list.append(S)
    
    Sc = 0
    for n in range(1, 4):
        Sc += (P*D)*n * S_list[n]

    Sg = 1
    for n in range(1, 4):
        Sg -= (P*D)*n
    Sg *= S_list[0]

    return Sg + Sc


def points(Eb, a, b):
    Eb = where(abs(Eb - 10) < 1, 10, Eb)
    Eb = where(abs(Eb - 20) < 1, 20, Eb)
    Eb = where(abs(Eb - 30) < 1, 30, Eb)
    Eb = where(abs(Eb - 40) < 1, 40, Eb)
    Eb = where(abs(Eb - 10) < 1, 10, Eb)

    return a*Eb + b


def plot(
        guess = (1, 0),
        Eb_ar = array([10, 20, 30, 40, 50]),
        S_ar = array([12, 25, 31, 42, 55]),
    ):
    fig, ax = plt.subplots()

    # fit S to points
    a, b = curve_fit(points, Eb_ar, S_ar, p0 = guess)[0]
    print("fitted parameters:\n\ta = {:.10g}, b = {:.10g}"
            .format(a, b))

    # coefficient of determination
    mean = sum(S_ar) / len(S_ar)
    ss_tot = sum( [(S - mean)**2 for S in S_ar] )
    ss_res = sum( [(S_ar[n] - points(Eb_ar[n], a, b) )**2
                 for n in range(len(S_ar))] )
    R = 1 - ss_res / ss_tot
    print("coefficient of determination:\n    R = {:.10g}".format(R))
    
    # fit P to 1 - exp^expInverse
    ax.plot(Eb_ar, S_ar, 'o')

    # curve based on fitted parameters
    x_ar = linspace(0, Eb_ar[-1], 500)
    fit_ar = a*x_ar + b

    # plot fitted curve

    ax.plot(x_ar, fit_ar, lw=2)

    plt.savefig('testfit.pdf')
    plt.close(fig)

#-------------------------------------------------------------------------------

def testWhere(x):
    y = where(x > 10, 1, -1)
    return y
#     Eb = where(abs(Eb - 2e4) < 1, 2e4, Eb)
#     Eb = where(abs(Eb - 3e4) < 1, 3e4, Eb)
#     Eb = where(abs(Eb - 4e4) < 1, 4e4, Eb)
#     Eb = where(abs(Eb - 6e4) < 1, 6e4, Eb)
#     Eb = where(abs(Eb - 8e4) < 1, 8e4, Eb)

