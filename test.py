from numpy import arange, array, linspace, where, zeros
from numpy import pi, ceil
import cProfile
import matplotlib.pyplot as plt
from gaussxw import gaussxw, gaussxwab
from numba import jit
from numpy import newaxis as na
from scipy.special import comb, binom, expi
from numpy import ones
from periodic import p_dict

# plotting class

# eIrrad modules
from dxs.classes import *
from eIrrad.plotScatter import ScatterPlot
from eIrrad.plotDispCross import SigmaPlot

# constants, unit conversions, and dictionaries
from constants import *

#--------------------------------------------------------------
# scatter

def scatterlog(outfile='scatterlog.png'):
    plot = ScatterPlot()
    plot.plot(pinfile='newprob.txt', logy=True, showmax=False)
    plot.decorate()
    plot.save(outfile=outfile)
    plot.show()

def scattervert(outfile='scattervert.png'):
    plot = ScatterPlot()
    plot.plot(pinfile='newprob.txt', logy=True, showmax=False, showvert=True)
    plot.decorate(ylim=[1e-16, 10])
    plot.save(outfile=outfile)
    plot.show()

# scatter
#--------------------------------------------------------------
# nimaxCon

def nimaxConBarm():
    nimaxCon(
            mat='Barm',
            precision=100,
            Eb=30,
            tau=240,
            outfile='nimaxConBarm.pdf',
            )

def nimaxConNarm():
    nimaxCon(
            mat='Narm',
            precision=100,
            Eb=30,
            tau=240,
            outfile='nimaxConNarm.pdf',
            )

def nimaxConMoS2():
    nimaxCon(
            mat='MoS2',
            precision=100,
            Eb=20,
            tau=81,
            outfile='nimaxConMoS2.pdf',
            )

def nimaxCon(
        mat = 'MoS2',
        precision = 20,
        Eb = 20,  # keV
        tau = 81,  # fs
        outfile = 'nimaxCon.pdf',
        save = True,
        nfmin = 4,

        figsize = (5.5,5.5),
        xlabel = 'maximim $n_i$',
        ylabel = 'auto',
        title=None,
        xoffset = 0,
        yoffset = None,
        ):
    """Plots nimax convergence."""
    sigma_list = []
    Ed_ar = Ed_dict[mat]
    nimax_ar = arange(4, Ed_ar.size+1)
    for nimax in nimax_ar:
        ed_ar = Ed_ar[:nimax+1]
        ed = ExciDispCross(mat, ed_ar, [Eb, Eb+1], 1, tau=tau)
        ed_dict = ed.getDict()
        sigma = ed_dict['sigma_a2'].sum()
        sigma_list.append(sigma)

    # plot
    fig, ax = plt.subplots(figsize = figsize)
    
    ax.plot(nimax_ar, sigma_list, marker = 'o', label = 'data')
    ax.set_xlabel(xlabel, fontsize = 20)
    if ylabel=='auto':
        ylabel = r'$\sigma(\epsilon_b$={} keV, $\tau$={} fs) (barn)'.format(
                Eb, tau)
    ax.set_ylabel(ylabel, fontsize = 20)
    ax.set_title(title, fontsize = 20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.grid()

    # show convergence
    standard = sigma_list[-1]
    thres = standard / precision
    convergence = None

    for i, y1 in enumerate(sigma_list[:-1]):
        converged = True

        for j, y2 in enumerate(sigma_list[i + 1:]):
            dif = abs(y2 - y1)
            if dif > thres:
                converged = False

        if converged:
            convergence = i
            con = int(nimax_ar[i])
            print('convergence at {}'.format(con))
            break

    # direction of annotation arrow
    mid = len(sigma_list) // 2
    concavity = sigma_list[0] + sigma_list[-1] - 2*sigma_list[mid]
    if concavity > 0 and yoffset==None:
        yoffset = 40
    elif concavity < 0 and yoffset==None:
        yoffset = -40

    ax.annotate(
            r"$\Delta\sigma/\sigma \leq$ {:.2g}%".format(100/precision),
            fontsize=20, ha='center', xy=(con, sigma_list[i]),
            xycoords='data', xytext=(xoffset, yoffset),
            textcoords='offset points',
            arrowprops=dict(facecolor='k', shrink=0.05))
    
    plt.tight_layout()
    if save:
        plt.savefig(outfile, dpi=300)
    plt.show()

    plt.show()


def plotHBN(
        outfile = 'edgePeaks.pdf',
        save = True,
        ebounds = [.001, 1],
        cbounds = None,
        tau = 400,  # fs
        res = 400,
        T = 1000+273.15  # K for speed
        ):
    """Plots hBN figure."""
    Plot = SigmaPlot()

    # get excited simulation
    BarmCross = ExciDispCross(mat='Barm', T=T, ebounds=ebounds, tau=tau,
            res=res)
    NarmCross = ExciDispCross(mat='Narm', T=T, ebounds=ebounds, tau=tau,
            res=res)
    Barm_dict = BarmCross.getDict()
    Narm_dict = NarmCross.getDict()
    sigmaB_ar = Barm_dict['sigma_a2'].sum(1)
    sigmaN_ar = Narm_dict['sigma_a2'].sum(1)
    print('Omega * 1: sigmaB = {:.5g}, sigmaN = {:.5g}'.format(sigmaB_ar[0],
        sigmaN_ar[0]))
    Eb_ar = Barm_dict['Eb_ar']

    # get excited simulation with 1/10 Omega
    B2armCross = ExciDispCross(mat='Barm2', T=T, ebounds=ebounds, tau=tau,
            res=res)
    N2armCross = ExciDispCross(mat='Narm2', T=T, ebounds=ebounds, tau=tau,
            res=res)
    B2arm_dict = B2armCross.getDict()
    N2arm_dict = N2armCross.getDict()
    sigmaB2_ar = B2arm_dict['sigma_a2'].sum(1)
    sigmaN2_ar = N2arm_dict['sigma_a2'].sum(1)
    print('Omega * 1: sigmaB2 = {:.5g}, sigmaN2 = {:.5g}'.format(sigmaB2_ar[0],
        sigmaN2_ar[0]))

    # plot simulation
#     Plot.ax.plot(Eb_ar, sigmaN_ar+sigmaB_ar, lw=2, zorder=3, label='total',
#             color='k')
    Plot.ax.plot(Eb_ar, sigmaN_ar+sigmaB_ar, lw=2, zorder=3, label='total',
            color='k')
    Plot.ax.plot(Eb_ar, sigmaB_ar, label='B', color='tab:green', lw=2)
    Plot.ax.plot(Eb_ar, sigmaN_ar, label='N', color='tab:blue', lw=2)

    # plot 1/10 Omega
#     Plot.ax.plot(Eb_ar, sigmaN2_ar+sigmaB2_ar, lw=2, zorder=3, label='total2',
#             color='k')
#     Plot.ax.plot(Eb_ar, sigmaB2_ar, label='B2', color='tab:green', lw=2,
#             ls='--')
#     Plot.ax.plot(Eb_ar, sigmaN2_ar, label='N2', color='tab:blue', lw=2,
#             ls='--')

    # plot experiment
#     try:
#         expt_t2 = hBNTemp_dict[T]
#         Plot.ax.plot(expt_t2[0], expt_t2[1], 's', color='k', markersize=7,
#                 label='expt')
#     except KeyError:
#         pass

    # text
    Plot.ax.text(.02, .98,
            'hBN (1273 K)', 
            fontsize=18,
            ha='left', va='top',
            transform=Plot.ax.transAxes)

    Plot.decorate(ebounds=ebounds, cbounds=cbounds, legend=False)
    Plot.ax.legend(fontsize=14, loc=4)
#             bbox_to_anchor=(.06, .92), loc='upper left')
    if save:
        Plot.save(outfile=outfile, view=False)
    Plot.show()


def lowEbTemp(ebounds=[.001, .5], save=False, outfile='lowEb.pdf'):
    """Plots T-dependent cross section for very low Eb.

    to see what effect Omega has.
    """
    t001 = TempDispCross(Ed_ar = arange(1,20)*.002, ebounds=ebounds, T=1)     
    t300 = TempDispCross(Ed_ar = arange(1,20)*.002, ebounds=ebounds)
    t001_dict = t001.getDict()
    t300_dict = t300.getDict()
    plot = SigmaPlot()
    plot.plot(t300_dict, label='300 K', color='tab:blue')
    plot.plot(t001_dict, label='1 K', color='tab:green')
    plot.decorate()
    if save:
        plt.savefig(outfile)
    plot.show()

def testF():
    Z = ZB
    M = MB
    Eb = linspace(2e4, 2e5, 200)
    beta = ec.getBeta(Eb)
    Emax = ec.getEmax(Eb, M)
    p = ec.getP(beta)
    pre = pi * (Z*alpha / p / beta)**2  # eV^{-2}
    nf = zeros(1)
    n = zeros(1)
    Ed_ar = Ed_dict['hBN']

    fig, ax = plt.subplots()
    for nf, Ed in enumerate(Ed_ar):
        F = pre*ec.getF(Emax, beta, Emax, nf, n, M, 1/invEVtoÅ, 1e4) \
            - pre*ec.getF(Ed, beta, Emax, nf, n, M, 1/invEVtoÅ, 1e4)
        F = where(F>0, F, 0) * invEVSqtoBarn
        ax.plot(Eb, F, label = nf)
    ax.legend()
    plt.show()

def testExciSigma(mat='hBN', tau=1e4, ebounds=[1e4, 2e5], cbounds=[0,100]):
    # material parameters
    Omega = Omega_dict[mat]  # eV^{-2}
    omega = omega_dict[mat]  # eV
    spec = spec_dict[mat]
    Z = p_dict[spec][0]
    M = p_dict[spec][1] * mp  # eV
    A, B, C = fit_dict[mat]
    D = 4.5/invEVtoÅ
#     Eb = linspace(2e4, 2e5, 200)[:,na,na,na,na,na]

    Eb = linspace(ebounds[0], ebounds[1], 200)[:,na,na]
    beta = ec.getBeta(Eb)
    Emax = ec.getEmax(Eb, M)
    p = ec.getP(beta)
#     Ed_ar = Ed_dict['hBN'][na,:,na,na,na,na]
    Ed_ar = Ed_dict[mat]

    pre = pi * (Z*alpha / p[:,0,0] / beta[:,0,0])**2  # eV^{-2}

    fig, ax = plt.subplots()
    Ni = Ed_ar.size
    ni = arange(Ni)[na,:,na]
    n = arange(Ni)[na,na,:]
    for nf, Ed in enumerate(Ed_ar):
        if Ed < 0:
            Ed = ec.getEmin(beta, Emax, p, Omega, Z)
        s = pre*ec.getExciSigmaA3(Eb, beta, Emax, p, Ed, nf, ni, n,
                M, D, tau, A, B, C
                ).sum((1,2)) * invEVSqtoBarn
        ax.plot(Eb[:,0,0], s, label = nf)
    ax.set_xlim(ebounds)
    ax.set_ylim(cbounds)
    ax.legend()

    # from SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
    if mat=='MoS2':
        expEb_ar = array([20, 30, 40, 60, 80]) * 1000
        expCross_ar = array([6.7, 11.6, 9., 4.2, 5.3])
        expErr_ar = array([0.5, 0.5, 1.0, 0.4, 0.4])
        ax.plot(expEb_ar, expCross_ar, 's', label='experiment', color='black',
                zorder=10)

    plt.show()

def testTempSigma(mat='hBN', tau=1e4, T=300*kb, ebounds=[1e4, 2e5], cbounds=[0,100]):
    # material parameters
    Omega = Omega_dict[mat]  # eV^{-2}
    omega = omega_dict[mat]  # eV
    spec = spec_dict[mat]
    Z = p_dict[spec][0]
    M = p_dict[spec][1] * mp  # eV
    A, B, C = fit_dict[mat]
    D = 4.5/invEVtoÅ
    Eb = linspace(2e4, 2e5, 200)[:,na,na,na,na,na]
    beta = ec.getBeta(Eb)
    Emax = ec.getEmax(Eb, M)
    p = ec.getP(beta)
    Ed = Ed_dict[mat][na,:,na,na,na,na]

    # possible numbers of phonon excitations
    nmax = int(ceil(20*ec.getBEN(omega, T)))  # lim>10 converges summations
    np = arange(nmax)

    # time and weights for GQ over an oscillation period
    t, twt = gaussxwab(10, 0, pi)

    pre = pi * (Z*alpha / p[:,0,0,0,0,0] / beta[:,0,0,0,0,0])**2  # eV^{-2}

    fig, ax = plt.subplots()
    Ni = Ed.size
    ni = arange(Ni)[na,na,na,na,:,na]
    nf = arange(Ni)[na,:,na,na,na,na]
    n = arange(Ni)[na,na,na,na,na,:]
    s_a2 = pre*ec.getTempSigmaAr(Eb, Ed, np, t, twt, nf, ni, n,
                omega, Omega,
                M, Z, T, D, tau, A, B, C
                ) * invEVSqtoBarn

    for k, s_ar in enumerate(s_a2):
        ax.plot(Eb[:,0,0,0,0,0], s_ar, label=k)

    ax.set_xlim(ebounds)
    ax.set_ylim(cbounds)
    ax.legend()

    # from SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
    if mat=='MoS2':
        expEb_ar = array([20, 30, 40, 60, 80]) * 1000
        expCross_ar = array([6.7, 11.6, 9., 4.2, 5.3])
        expErr_ar = array([0.5, 0.5, 1.0, 0.4, 0.4])
        ax.plot(expEb_ar, expCross_ar, 's', label='experiment', color='black',
                zorder=10)

    plt.show()


def testExciSum(mat='hBN',tau=1e4):
    # material parameters
    Omega = Omega_dict[mat]  # eV^{-2}
    omega = omega_dict[mat]  # eV
    spec = spec_dict[mat]
    Z = p_dict[spec][0]
    M = p_dict[spec][1] * mp  # eV
    A, B, C = fit_dict[mat]
    D = 1/invEVtoÅ

#     Eb = linspace(2e4, 2e5, 200)[:,na,na,na,na,na]
    Eb = linspace(2e4, 2e5, 200)[:,na,na,na]
    beta = ec.getBeta(Eb)
    Emax = ec.getEmax(Eb, M)
    p = ec.getP(beta)
    Ed = Ed_dict[mat][na,:,na,na]
    Ed = where(Ed > 0, Ed, ec.getEmin(Ed, Emax, p, Omega, Z))

    pre = pi * (Z*alpha / p[:,0,0,0] / beta[:,0,0,0])**2  # eV^{-2}
#     Ed_ar = Ed_dict['hBN'][na,:,na,na,na,na]

    fig, ax = plt.subplots()
    Ni = Ed.shape[1]
    nf = arange(Ni)[na,:,na,na]
    ni = arange(Ni)[na,na,:,na]
    n = arange(Ni)[na,na,na,:]
    s = pre*ec.getExciSigmaA3(Eb, beta, Emax, p, Ed, nf, ni, n,
            M, D, tau, A, B, C
            ).sum((1,2,3)) * invEVSqtoBarn
    ax.plot(Eb[:,0,0,0], s)
    plt.show()


def testPi():
    mat = 'MoS2'
    A, B, C = fit_dict[mat]
    ni = arange(5)[na,:]
    Eb = linspace(2e4, 2e5, 20)[:,na]
    return ec.getPi(Eb, ni, A, B, C)


def testPython(N=100):
    x_ar=arange(N)
    y_ar=arange(N)
    z_ar=arange(N)
    s=0
    for x in x_ar:
        a = x**2
        for y in y_ar:
            b = y**.5
            for z in z_ar:
                s += a+b+z
    return s


def testBroadcast(N=100):
    x_ar=arange(N)
    y_ar=arange(N)
    z_ar=arange(N)
    return (x_ar[:,na,na]**2 + y_ar[na,:,na]**.5 + z_ar[na,na,:]).sum()

@jit(nopython=True)
def testJBroadcast(N=100):
    x_ar=arange(N)
    y_ar=arange(N)
    z_ar=arange(N)
    return (x_ar[:,na,na]**2 + y_ar[na,:,na]**.5 + z_ar[na,na,:]).sum()

@jit(nopython=True)
def testNumba(N=100):
    x_ar=arange(N)
    y_ar=arange(N)
    z_ar=arange(N)
    s=0
    for x in x_ar:
        a = x**2
        for y in y_ar:
            b = y**.5
            for z in z_ar:
                s += a*b/(1+z)
    return s


@jit(nopython=True)
def testNOrder(N=100):
    x_ar=arange(N)
    y_ar=arange(N)
    z_ar=arange(N)
    s = 0
    for x in x_ar:
        s1 = 0
        a = x**2
        for y in y_ar:
            s2 = 0
            b = y**.5
            for z in z_ar:
                s2 += 1/(1+z)
            s1 += b*s2
        s += a*s1
            
    return s
                
def testWhere(N=1000):
    x_ar = arange(N)
    return where(x_ar < 2, x_ar, 0).sum()

@jit(nopython=True)
def testJWhere(N=1000):
    x_ar = arange(N)
    return where(x_ar < 2, x_ar, 0).sum()


# @jit(nopython=True)
def testBinom(N=100):
    x_ar = arange(N)
    return binom(x_ar, 1).sum()
    
def testComb(N=100):
    x_ar = arange(N)
    y_ar = arange(N)
    return comb(x_ar[na, :], y_ar[:, na]).sum()

@jit(nopython=True)
def myComb(n, k):
    c = 1.
    if n < k:
        return 0.
#     elif k==0:
#         return 1.
#     elif n==k:
#         return 1.
    elif k!=0 and k!=n:
        for l in range(k):
            c*=(n-l)/(l+1)
    return c

def testMyCombAcc(N=9):
    for n in range(N):
        for k in range(N):
            c = comb(n, k)
            mc = myComb(n, k)
            print('({}, {}): {}, {}'.format(n, k, c, mc))
            if c != mc:
                print('not equal!')

@jit(nopython=True)
def testMyCombSpeed(N=100):
    s = 0
    for n in range(N):
        for k in range(N):
            s += myComb(n, k)
    return s
        

@jit(nopython=True)
def testExpi(N=100):
    return expi(arange(1, N))

def combSumArr(N=5):
    ni = arange(N+1)[:, na, na]
    nf = arange(N+1)[na, :, na]
    n = arange(N+1)[na, na, :]
    return (comb(ni, nf) * comb(ni-nf, n)).sum()

def combSumLoop(N=5):
    s = 0
    for ni in range(N+1):
        for nf in range(ni+1):
            cif = comb(ni, nf)
            dn = ni-nf
            for n in range(dn+1):
                cmu = comb(dn, n)
                s += cif*cmu
    return s


def plotDSigma():
    Eb = 1e5
    beta = ec.getBeta(Eb)
    Emax = ec.getEmax(Eb, MS)
    p = ec.getP(beta)
    E_ar = linspace(.1, Emax, 200)

    dSigma_ar = ec.getDSigma(E_ar, beta, Emax, p, ZS) * invEVSqtoBarn

    nf_ar = arange(1)
    pf_ar = ec.getPf(Eb, nf)

    fig, ax = plt.subplots()
    ax.plot(E_ar, dSigma_ar, lw=2)
    ax.set_xlabel('energy transfer (eV)')
    ax.set_ylabel('diff xsection (Barn/eV)')
    ax.set_ylim(0,1000)
    ax.grid()
    plt.tight_layout()

    plt.show()
    

def profileGetAvgSigma():
    cProfile.run(ec.plotAvgSigma(), 'avgSigmaProfile.dat')

# toBarn = ec.invEVSqtoBarn

# M = ec.MS
# Z = ec.ZS
# A, B, C = ec.fit_dict['MoS2']
# omega = ec.omega_dict['MoS2']
# Omega = ec.Omega_dict['MoS2']
# 
# Eb_ar = arange(1, 50)*2e3
# Ed_ar = ec.Ed_dict['MoS2']
# np_ar = arange(12)
# ni_ar = arange(5)
# E_ar, Ewt_ar = gaussxw(NGE)
# t_ar, twt_ar = gaussxwab(NGt, 0, pi)
# 
# Eb_a6 = Eb_ar[:,na,na,na,na,na]
# Ed_a6 = Ed_ar[na,:,na,na,na,na]
# np_a6 = np_ar[na,na,:,na,na,na]
# t_a6 = t_ar[na,na,na,:,na,na]
# nf_a6 = nf_ar[na,:,na,na,na,na]
# ni_a6 = ni_ar[na,na,na,na,na,:]
# E_a6 = E_ar[na,na,na,na,:,na]
# Ewt_a6 = Ewt_ar[na,na,na,na,:,na]
# 
# beta_a6 = ec.getBeta(Eb_a6)
# Emax_a6 = ec.getEmax(Eb_a6, M)
# p_a6 = ec.getP(beta_a6)
# 
# Emin_a6 = ec.getEmin(beta_a6, Emax_a6, p_a6, Omega, Z)

# def runTest():
#     ec.getPfA5(Eb_a6, nf_a6, ni_a6, Eab_a6, EdEff_a6)

def mycomb(a, b):
    pass

# def testComb(N=1e5):
#     a = ones(N)*10
#     comb(a, 5)

# import exciDispCross as ec
