#!/usr/bin/env python

# Anthony Yoshimura

from numpy import array, zeros, append, floor, ceil, arange, linspace
from numpy import dot, tensordot, cross, sign, abs, log, exp
from numpy import sqrt, sin, cos, arccos, radians, degrees, log10, float64
from numpy import e, pi, array_equal, where
from numpy.linalg import norm, inv
from scipy.optimize import curve_fit
from numba import jit

from copy import deepcopy
from inspect import stack
import os
from subprocess import call
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt

from periodic import p_dict
from quantumEspresso.read import readExport
from constants import *
from eIrrad.write import writeKpts

# include date on outfile file names
from datetime import date
from shutil import copyfile

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

"""
version 201118:
    write copy of outfiles with date in names
"""

#------------------------------------------------------------------------------
# maxProbCon

def plotMaxProb(
        infile='maxProb.txt',
        outfile='maxProb.pdf',
        density=True,
        xvar='k'
        ):
    """plots most likely transitions."""
    kden_list = []
    maxp_list = []
    Eb_list = []
    with open(infile) as f:
        for line in f:
            if 'kden' in line:
                for line in f:
                    line_list = line.split()
                    if len(line_list) == 0:
                        break
                    else:
                        kden, Eb = [float(val) for val in line_list[1:]]
                        kden_list.append(kden)
                        Eb_list.append(Eb)

            if 'pden' in line:
                for line in f:
                    line_list = line.split()
                    if len(line_list) == 0:
                        break
                    else:
                        maxp = float(line_list[1])
                        maxp_list.append(maxp)
    kden_ar = array(kden_list)
    maxp_ar = array(maxp_list)
    Eb_ar = array(Eb_list)
    if density:
        maxp_ar *= kden_ar**2
     
    mpl.style.use('figure')
    fig, ax = plt.subplots()
    if xvar[0]=='k' or xvar[0]=='K':
        ax.plot(kden_ar, maxp_ar, marker='o')
        ax.set_xlabel(r'k-point density ($\AA^3$)')
    elif xvar[0]=='e' or xvar[0]=='E':
        ax.plot(Eb_ar/1000, maxp_ar, marker='o')
        ax.set_xlabel('electron energy (keV)')
    ax.set_ylabel(r'probability density ($\AA^3$)')
    ax.set_ylim(0, maxp_ar.max()*1.1)
    plt.tight_layout()
    plt.show()


def writeMaxProb(
        d_list = None,
        pinfile = 'newprob.txt',
        kinfile = 'kpts.txt',
        top = '.',
        outfile = 'maxProb.txt',
        ):
    """writes most likely transitions."""
    if d_list==None:
        d_list = [d for d in os.listdir(top) if d[:2].isdigit()]
        d_list.sort()

    maxp_list = []
    e0_list = []
    k2_l2 = []
    k3_l2 = []
    k2ind_list = []
    k3ind_list = []
    n2_list = []
    n3_list = []
    kden_list = []
    Eb_list = []

    for d in d_list:
        inpath = f'{d}/{pinfile}'
        try:
            with open(inpath) as i:
                print(f'reading {inpath}')
                p_list = []
                e_list = []
                t_l2 = []
                for line in i:
                    if 'beam energy' in line:
                        Eb = float(line.split('=')[1].split()[0])
                    elif 'bands' in line and 'val' not in line:
                        nb = float(line.split()[-1])
                    elif 'val bands' in line:
                        nv = float(line.split()[-1])
                    elif 'k-points' in line:
                        nk = float(line.split('=')[1].split()[0])
                    elif 'area' in line:
                        area = float(line.split('=')[1].split()[0])  # eV^{-2}
                    elif 'cell-height' in line:
                        height = float(line.split('=')[1].split()[0])  # A
                    elif 'energy (eV)' in line:
                        for line in i:
                            e, p, k2, k3, n2, n3 = \
                                    [float(val) for val in line.split()]
                            e_list.append(e)
                            p_list.append(p)
                            t_l2.append([k2, k3, n2, n3])

        except FileNotFoundError:
            continue

        nc = nb - nv
        p_ar = array(p_list)
        e_ar = array(e_list)
        t_a2 = array(t_l2, dtype=int)

        maxp = p_ar.max()
        maxp_idx = where(p_ar==maxp)
        e0 = e_ar[maxp_idx][0]
        k2, k3, n2, n3 = t_a2[maxp_idx][0]
        
        # write kinfile if it doesn't already exist
        writeKpts(top=d)

        k_l2 = []
        with open(f'{d}/{kinfile}') as i:
            for _ in range(2):
                i.readline()

            for line in i:
                k_l2.append([float(val) for val in line.split()])  # eV
        k_a2 = array(k_l2)
        maxk2_ar = k_a2[k2]
        maxk3_ar = k_a2[k3]

        volume = area*height*invEVSqtoÅSq  # A^3
        kden = nk*volume/(2*pi)**3  # A^3
#         maxp *= kden**2  # A^6
        
        e0_list.append(e0)
        k2_l2.append(maxk2_ar)
        k3_l2.append(maxk3_ar)
        k2ind_list.append(k2)
        k3ind_list.append(k3)
        n2_list.append(n2)
        n3_list.append(n3)
        maxp_list.append(maxp)
        kden_list.append(kden)
        Eb_list.append(Eb)
    
    with open(outfile, 'w') as o:
        o.write('dir    kden (A^3)     Eb (eV)\n')
        for d, kden, Eb in zip(d_list, kden_list, Eb_list):
            o.write(f'{d}{kden:>12.6E}{Eb:>12.3E}\n')

        o.write("\nenergy (eV)   pden (A^6)     k   k'  n   n'\n")
        for e0, maxp, k2ind, k3ind, n2, n3 in zip(
                e0_list, maxp_list, k2ind_list, k3ind_list, n2_list, n3_list):
            o.write(
                f'{e0:>11.6f}{maxp:>15.6E}'
                f'   {k2ind:<4}{k3ind:<4}'
                f'{n2:<4}{n3}\n')

        o.write(f'\n   {"k2":<26}{"k3":<36}\n')
        for k2_ar, k3_ar in zip(k2_l2, k3_l2):
#                 o.write('{k2_ar:<36}{k3_ar:<36}\n')
                o.write(
                '{:>13.8f}{:>13.8f}{:>13.8f}{:>13.8f}\n'
                .format(*k2_ar[:-1], *k3_ar[:-1]))

    return maxp_list, e0_list, k2_l2, k3_l2, n2_list, n3_list



# maxProbCon
#------------------------------------------------------------------------------
# specific XVsY convergences

def plotEnCon(precision=20):
    """Plots cutoff energy convergence."""
    plotXVsY(xkey='encut', ykey='total sum',
            xlabel='cutoff energy (eV)',
            ylabel="$S$",
            xoffset=120, twolines=False,
            save=True, outfile='enCon.pdf',
            convergence=True, precision=precision)
        
def plotMaxCon(precision=20, xoffset=0, yoffset=0):
    """Plots |qmax| convergence.

    for hBN: xoffset=150, yoffset=40
    """
    plotXVsY(xkey='max photon energy', ykey='total sum',
            xlabel='max momentum transfer (eV)',
            ylabel="$S$",
#             xbounds=None, xoffset=100,
            xoffset=xoffset, yoffset=yoffset, twolines=False,
            save=True, outfile='maxCon.pdf',
            convergence=True, precision=precision)

def plotZCon(precision=20, xoffset=140, yoffset=-40):
    """Plots cell-height convergence.

    for hBN: xoffset=120, yoffset=40
    """
    plotXVsY(xkey='cell-height', ykey='total sum',
            xlabel='cell height (Å)',
            ylabel="$S$",
            xoffset=xoffset, yoffset=yoffset, twolines=False,
            save=True, outfile='zCon.pdf',
            convergence=True, precision=precision)

# specific convergences
#------------------------------------------------------------------------------
# plotXVsY

unit_dict = {
        'nk': '',
        'cell-height': "Å",
        'max photon energy': 'eV',
        'encut': 'eV',
        }

def plotXVsY(
        infile = 'head.txt',
        xkey = 'nk',
        ykey = 'total prob',

        xlabel = 'number of k-points',
        ylabel = 'excitation probability',
        title = None,
        fit = False,
        convergence = False,
        precision = 100,
        figsize = (6,5),
#         xbounds = None,
#         ybounds = None,

        xoffset = 0,
        yoffset = None,
        twolines = True,

        save = False,
        outfile = 'convergence.png',
        ):
    """
    read set of infiles, plots one parameter vs. another
    {x,y}key: string to look up in infile (str)
    precision: convergence criteria i.e., one part in precision (float)
    """
    d_list = next(os.walk('.'))[1]
    d_list.sort()
    print('searching in the following directories:\n{}'.format(d_list))

    y_list = []
    x_list = []
    for d in d_list:
        try:
            with open('{}/{}'.format(d, infile)) as f:
                for line in f:
                    if xkey in line:
                        for word in line.split():
                            if word[0].isdigit():
                                x = float(word)
                        x_list.append(x)
    
                    if ykey in line:
                        for word in line.split():
                            if word[0].isdigit():
                                y = float(word)
                        y_list.append(y)
                        
        except FileNotFoundError:
            continue
    x_ar = array(x_list)
    xmean = x_ar.mean()
    if xmean > 10000:
        print('converting from eV to keV')
        x_ar /= 1000

    # convert q0max to |qmax|
    if xkey == 'max photon energy':
        x_ar = ((m + x_ar)**2 - m**2)**.5  # eV

    # plot
    fig, ax = plt.subplots(figsize = figsize)
    
#     ax.set_xlim(xbounds)
#     ax.set_ylim(ybounds)
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    ax.set_title(title, fontsize = 20)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax.grid()

    if fit:
        ax.plot(x_ar, y_list, 'o', label = 'data')
        # get inverse power fit
        guess = (1.0, 1.0, 1.0)
        a, b, c = curve_fit(inverse, x_ar, y_list, p0 = guess)[0]
        print("fitted parameters:\n\ta = {:.4g}, b = {:.4g}, c = {:.4g}"
                .format(a, b, c))
    
        # coefficient of determination
        mean = sum(y_list) / len(y_list)
        ss_tot = sum( [(y - mean)**2 for y in y_list] )
        ss_res = sum( [(y_list[n] - inverse(x_ar[n], a,b,c) )**2
                     for n in range(len(y_list))] )
        R = 1 - ss_res / ss_tot
        print("coefficient of determination:\n    R = {:10g}".format(R))
        
        # curve based on fitted parameters
        x_ar = linspace(x_ar[0], x_ar[-1], 200)
        fit_ar = inverse(x_ar, a, b, c)
        ax.plot(x_ar, fit_ar, lw = 2, label = 'R = {:.4g}'.format(R))
        ax.legend(fontsize = 16)

    else:
        ax.plot(x_ar, y_list, marker = 'o', label = 'data')

    # show convergence
    if convergence:
        unit = unit_dict[xkey]
        standard = y_list[-1]
        thres = standard / precision
        convergence = None
    
        for i, y1 in enumerate(y_list[:-1]):
            converged = True
    
            for j, y2 in enumerate(y_list[i + 1:]):
                dif = abs(y2 - y1)
                if dif > thres:
                    converged = False
    
            if converged:
                convergence = i
                con = round(x_ar[i])
                d = d_list[i]
                print('convergence at {} from directory {}'.format(con, d))
                break
    
        # direction of annotation arrow
        mid = len(y_list) // 2
        concavity = y_list[0] + y_list[-1] - 2*y_list[mid]
        if concavity > 0 and yoffset==None:
            yoffset = 40
        elif concavity < 0 and yoffset==None:
            yoffset = -40

        if twolines:
            join = '\n'
        else:
            join = ' '
        
        ax.annotate(
                r"$\Delta S/S \leq$ {:.2g}%"
                "{}at {} {}".format(100/precision, join, con, unit),
                fontsize = 20, ha = 'center', xy = (con, y_list[i]),
                xycoords = 'data', xytext = (xoffset, yoffset),
                textcoords = 'offset points',
                arrowprops = dict(facecolor = 'k', shrink = 0.05))
    
    plt.tight_layout()
    if save:
        plt.savefig(outfile, dpi=300)
    plt.show()

    # make copy with date in name
    today = date.today()
    year = today.year - 2000
    month = today.month
    day = today.day

    if '.' in outfile:
        name, ext = outfile.split('.')
        outfileCopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month, day, ext)
    else:
        outfileCopy = '{}{:02d}{:02d}{:02d}'.format(outfile, year, month, day)

    copyfile(outfile, outfileCopy)

# plotXVsY
#------------------------------------------------------------------------------

def plotOvKCon(
        infile = 'ovKCon.txt',
        figsize = (6,5),
        save = True,
        outfile = 'ovKCon.png',
        title = None,
        ):
    """
    plots average overlap between adjacent inter and intraband states
        in a uniform k-point mesh as a function of smallest k-point spacing
        * run in directory containing
    """
    # get inter and intra band overlaps for each k-mesh
    q_list = []
    interOv_list = []
    intraOv_list = []
    
    with open(infile) as f:
        for line in f:

            if 'nq' in line:
                nq = int(line.split()[-1])

            if 'q (eV)' in line:
                for _ in range(nq):
                    q, interOv, intraOv = [float(val) for val in
                            f.readline().split()]
                    q_list.append(q)
                    interOv_list.append(interOv)
                    intraOv_list.append(intraOv)

    # add q = 0 points
    q_ar = array(q_list + [0.0])
    interOv_ar = array(interOv_list + [0.0])
    intraOv_ar = array(intraOv_list + [1.0])

    # plot overlap vs. q
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(array(q_ar)/1000, interOv_ar, marker='o', lw=2, label='interband')
    ax.plot(array(q_ar)/1000, intraOv_ar, marker='o', lw=2, label='intraband')
    ax.set_xlabel('q-shift (keV)', fontsize = 16)
    ax.set_ylabel('average ovelap', fontsize = 16)
    ax.set_title(title, fontsize = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.grid()
    ax.legend(fontsize = 12)

    plt.tight_layout()
    if save:
        plt.savefig(outfile, dpi = 300)
    plt.show()


def writeOvKCon(
        exroot = 'tmp/hBN.export',
        out = 'scf.out',
        write = True,
        writefile = 'ovKCon.txt',
        ):
    """
    plots average overlap between adjacent inter and intraband states
        in a uniform k-point mesh as a function of smallest k-point spacing
        * run in directory containing
    """
    # get directories for each k-point mesh run
    d_list = [d for d in os.listdir() if d[0].isdigit()]
    d_list.sort()
    nq = len(d_list)

    # get average overlap over adjacent k-points
    interOv_list = []
    intraOv_list = []
    q_list = []
    for q, d in enumerate(d_list):
        st = time()
        qexroot = '{}/{}'.format(d, exroot)
        qout = '{}/{}'.format(d, out)
        try:
            p_a5, C_a5, E_a2, k_a2, b_a2, ne, volume, area, encut = \
                    _readOutput(qexroot)
            nk, nb, nKx, nKy, nKz = C_a5.shape
            height = volume / area

        except FileNotFoundError:
            continue

        interOv, intraOv, qmag = getOvK(C_a5, k_a2)
        interOv_list.append(interOv)
        intraOv_list.append(intraOv)
        q_list.append(qmag)
        print('calculating interOv from {} took {:.5g} seconds'.format(qexroot,
            time() - st))

    if write:
        with open(writefile, 'w') as f:
            f.write('nq = {}\n'.format(len(q_list)))
            f.write('nb = {}\n'.format(nb))
            f.write('encut = {} eV\n'.format(encut))
            f.write('cell-height = {} Å\n\n'.format(height))
            f.write('     q (eV)      interband        intraband\n')

            for q, interOv, intraOv in zip(q_list, interOv_list, intraOv_list):
                f.write('{:>12.6f}{:>16.10f}{:>16.10f}\n'.format(q, interOv,
                    intraOv))

    return interOv_list, q_list
    

@jit(nopython=True)
def getOvK(
        C_a5,
        k_a2,
        ):
    """
    returns average overlap between adjacent inter and intraband states
        in a uniform k-point mesh as a function of smallest k-point spacing
        * run indirectory containing scf.out and outdir
        * all k-points must be listed in same order
    """
    Cc_a5 = C_a5.conjugate()
    nk, nb, nKx, nKy, nKz = C_a5.shape

    # smallest q
    qmag = norm(k_a2[1] - k_a2[0])
    
    # interband and intraband overlaps
    interOv = 0
    intraOv = 0
    ninter = 0
    nintra = 0
    for k1, k1_ar in enumerate(k_a2):
        for k2, k2_ar in enumerate(k_a2):
            if norm(k2_ar - k1_ar) < qmag * 1.01:
                for i in range(nb):
                    cov = (C_a5[k1,i]*Cc_a5[k2,i]).sum()
                    intraOv += (cov * cov.conjugate()).real**.5
                    nintra += 1
                    for j in range(nb):
                        if i != j:
                            cov = (C_a5[k1,i]*Cc_a5[k2,j]).sum()
                            interOv += (cov * cov.conjugate()).real**.5
                            ninter += 1

    return interOv / ninter, intraOv / nintra, qmag


def plotOvLine(
        infile = 'ovLineCon.txt',

        log = False,
        figsize = (6,5),
        xbounds = None,
        ybounds = None,
        save = False,
        outfile = 'ovLineCon.png',
        title = None,
        ):
    """
    plots magnitude of average overlap between states as a function of q-shift
        between their k-points
    """
    q_list = []
    interOv_list = []
    intraOv_list = []
    
    with open(infile) as f:
        for line in f:

            if 'nq' in line:
                nq = int(line.split()[-1])

            if 'q (eV)' in line:
                for _ in range(nq):
                    q, interOv, intraOv = [float(val) for val in
                            f.readline().split()]
                    q_list.append(q)
                    interOv_list.append(interOv)
                    intraOv_list.append(intraOv)

    q_ar = array([0.0] + q_list)
    interOv_ar = array([0.0] + interOv_list)
    intraOv_ar = array([1.0] + intraOv_list)

    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(q_ar/1000, interOv_ar, marker = 'o', lw = 2, label = 'interband')
    ax.plot(q_ar/1000, intraOv_ar, marker = 'o', lw = 2, label = 'intraband')
    ax.set_xlabel('q-shift (keV)', fontsize = 16)
    ax.set_ylabel('average ovelap', fontsize = 16)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.set_title(title, fontsize = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.grid()
    if log:
        ax.set_xscale('log')

    ax.legend(fontsize = 12)

    plt.tight_layout()
    if save:
        if log:
            outfile = 'log' + outfile
        plt.savefig(outfile, dpi = 300)
    plt.show()


def writeOvLine(
        exroot = 'tmp/hBN.export',
        out = 'scf.out',
        outfile = 'ovLineList.txt',
        ):
    """
    returns and writes magnitude of average overlap between interband states
        as a function of q-shift between their k-points
        i.e., average overlap between (n, k) and (m, k+q) for all n and k
        and n != m
    """
    startTime = time()

    d_list = [d for d in os.listdir() if d[0].isdigit()]
    d_list.sort()

    # get Gamma-centered coefficients
    gexroot = '0.0/{}'.format(exroot)
    gout = '0.0/{}'.format(out)
    print('reading Gamma-centered coefficients from {}'.format(gexroot))
    p0_a5, C0_a5, E0_a2, k0_a2, b_a2, ne, volume, area, encut = _readOutput(
            gexroot)
    height = volume / area
    nk, nb, nKx, nKy, nKz = C0_a5.shape

    # get average overlap
    q_list = []
    interOv_list = []
    intraOv_list = []
    for q, d in enumerate(d_list[1:]):
        qexroot = '{}/{}'.format(d, exroot)
        qout = '{}/{}'.format(d, out)
        print('reading q-shifted coefficients from {}'.format(qexroot))
        try:
            pq_a5, Cq_a5, Eq_a2, kq_a2 = _readOutput(qexroot)[:4]
            
            interOv = 0
            intraOv = 0
            for k in range(nk):
                for i in range(nb):
                    intraOv += norm((C0_a5[k, i]*Cq_a5[k, i]\
                                .conjugate()).sum())
                    for j in range(nb):
                        if i != j:
                            interOv += norm((C0_a5[k,i]*Cq_a5[k,j]\
                                    .conjugate()).sum())
            interOv_list.append(interOv/nk/nb/(nb - 1))
            intraOv_list.append(intraOv/nk/nb)
            q_list.append(norm(kq_a2[0] - k0_a2[0]))

        except FileNotFoundError:
            print('    could not find {}'.format(qexroot))
            continue

    with open(outfile, 'w') as f:
        f.write('nq = {}\n'.format(len(q_list)))
        f.write('nb = {}\n'.format(nb))
        f.write('nk = {}\n'.format(nk))
        f.write('encut = {} eV\n'.format(encut))
        f.write('cell-height = {} Å\n\n'.format(height))
        f.write('    q (eV)      interband       intraband\n')

        for q, interOv, intraOv in zip(q_list, interOv_list, intraOv_list):
            f.write('{:>12.6f}{:>16.10f}{:>16.10f}\n'.format(q, interOv,
                intraOv))

        f.write('\ntotal time = {:.5g} seconds\n'.format(time() - startTime))

    return array(q_list), array(interOv_list), array(intraOv_list)


def plotSingleOv(
        exroot = 'tmp/hBN.export',
        out = 'scf.out',
        n0 = 0,
        nq = 1,
        k = 0,

        log = False,
        figsize = (6,5),
        xbounds = None,
        ybounds = None,
        save = False,
        outfile = 'singOv.png',
        title = None,
        ):
    """
    returns and writes magnitude of overlap between states
        (n0, k) and (nq, k+q) 
    """
    startTime = time()

    d_list = [d for d in os.listdir() if d[0].isdigit()]
    d_list.sort()

    # get Gamma-centered coefficients
    gexroot = '0.0/{}'.format(exroot)
    gout = '0.0/{}'.format(out)
    print('reading Gamma-centered coefficients from {}'.format(gexroot))
    p0_a5, C0_a5, E0_a2, k0_a2, b_a2, ne, volume, area, encut = _readOutput(
            gexroot)
    height = volume / area
    nk, nb, nKx, nKy, nKz = C0_a5.shape

    # get average overlap
    q_list = []
    ov_list = []
    for q, d in enumerate(d_list[1:]):
        qexroot = '{}/{}'.format(d, exroot)
        qout = '{}/{}'.format(d, out)
        print('reading q-shifted coefficients from {}'.format(qexroot))
        try:
            pq_a5, Cq_a5, Eq_a2, kq_a2 = _readOutput(qexroot, qout)[:4]
            
            overlap = norm((C0_a5[k, n0]*Cq_a5[k, nq].conjugate()).sum())
            ov_list.append(overlap)
            q_list.append(norm(kq_a2[0] - k0_a2[0]))

        except FileNotFoundError:
            print('    could not find {}'.format(qexroot))
            continue

    q_ar = array(q_list, dtype = float64)
    ov_ar = array(ov_list, dtype = float64)

    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(q_ar/1000, ov_ar, marker = 'o', lw = 2)
    ax.set_xlabel('q-shift (keV)', fontsize = 16)
    ax.set_ylabel('overlap', fontsize = 16)
    ax.text(0.02, 0.98, 'k = {}, n0 = {}, nq = {}'.format(k, n0, nq),
            transform = ax.transAxes, ha = 'left', va = 'top', fontsize = 16)
    ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)
    ax.set_title(title, fontsize = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.grid()
    if log:
        ax.set_xscale('log')

    plt.tight_layout()
    if save:
        if log:
            outfile = 'log' + outfile
        plt.savefig(outfile, dpi = 300)
    plt.show()

    return q_ar, ov_ar


def getSummand(
        Eb = 6e4,  # eV
        ):
    """
    returns the summands for a q-shift amplitudes at each k-point
        * summand_a4[q, k, vb, cb] --> summand
        * run with center and shifted directories in current directory
    """
    # read WAVECAR and OUTCAR information
    d_list = os.listdir()
    d_list.sort()
    p_a6, C_a6, E_a3 = rpa._getPa6Ca6Ea3(d_list = d_list)
    E_a2, k_a2, wt_ar, nelect, cellVolume, area, encut = rf._readOUTCAR(
            '000/OUTCAR')

    Cc_a6 = C_a6.conjugate()
    nqpts, nk, nbands, nKx, nKy, nKz = C_a6.shape
    nocc = int(round(nelect/2))
    nunocc = nbands - nocc

    # beam momentum
    E1 = Eb + m
    gamma = E1/m
    gammap = gamma + 1  # for p3z calculation
    p1z = (E1**2 - m**2)**.5

    # to obtain best p3 index
    pz_ar = p_a6[0, 0, 0, :, 3]

    # get dSigma for q-shift from each k-point
    summand_a4 = zeros((nqpts, nk, nocc, nunocc))
    for q in range(1, nqpts):
        for k in range(nk):
    
            E2, p2x, p2y, p2z = p_a6[0, k, 0, 0, 0]
            p3x, p3y = p_a6[q, k, 0, 0, 0, 1:3]
            print('q = {:5g} eV'.format(p3x - p2x))
    
            # p3z(p1, p2, p3perp) thesis appendix H
            p3z = (p1z + p2z - ((p1z - gamma*p2z)**2
                + gammap*(gamma*(
                    p2x**2 + p2y**2 - p3x**2 - p3y**2)
                    -
                    (p2x - p3x)**2 - (p2y - p3y)**2
                ))**.5)/gammap
    
            E3 = (p3x**2 + p3y**2 + p3z**2 + m**2)**.5
            E4 = E1 + E2 - E3
            p4x = p2x - p3x
            p4y = p2y - p3y
            p4z = p1z + p2z - p3z
    
            u = (E1 - E4)**2 - p4x**2 - p4y**2 - (p1z - p4z)**2
    
            # common factor in M terms
            pre = ((E1+m)*(E2+m)*(E3+m)*(E4+m))**-.5
    
            # u-channel from ornl/notes/noSpinSums.nb
            M = 1/u \
            * (((E3 + m)*p2x + (E2 + m)*p3x) \
            * (E1 + m)*p4x \
            + ((E2 + m)*(E3 + m) + p2x*p3x \
            + (p2y + 1j*p2z)*(p3y - 1j*p3z)) \
            * ((E1 + m)*(E4 + m) \
            + 1j*p1z*(p4y - 1j*p4z)) \
            + (m*p2y + E3*(p2y + 1j*p2z) + 1j*m*p2z \
            + E2*p3y + m*p3y - 1j*(E2 + m)*p3z) \
            * (E4*1j*p1z + 1j*m*p1z \
            + E1*p4y + m*p4y - 1j*(E1 + m)*p4z) \
            + (E3*(-1j*p2y + p2z) + E2*(1j*p3y + p3z) \
            + m*(-1j*p2y + p2z + 1j*p3y + p3z)) \
            * (E4*p1z + E1*(1j*p4y + p4z) \
            + m*(p1z + 1j*p4y + p4z)))
    
            # factor resulting from wavepacket overlap
            overlap = (E4*E3/E2/E1)**.5 / abs(E3*p4z - E4*p3z)
    
            # d\sigma divided by factorable constants
            dSig = overlap * pre * M
            K3z = abs(pz_ar - p3z).argmin() 
    
            # sum over plane wave coefficients to determine probility
            for v in range(nocc):
                for c_ in range(nunocc):
                    c = c_ + nocc
    
                    Cv = C_a6[0, k, v, 0, 0, 0]
                    Cc = Cc_a6[q, k, c, 0, 0, K3z]
                    amp = Cv * Cc * dSig * (p3x - p2x)**2
    
                    summand_a4[q, k, v, c_] = (amp * amp.conjugate()).real
    
    # (4\pi\alpha)^2, x2 t-u symmetry, x4^2 noSpinSums.nb
    # /4 spin avg, /16 overlap prefactor
    # spin degeneracy: square "noscatter" twice in getTotProb 
#    return 8 * (pi * alpha)**2 * summand_a4 / nk**2 / area**2
    return 8 * (pi * alpha)**2 * summand_a4

#---------------------------- FITTING FUNCTIONS -------------------------------

def linear(x, a, b):
    return a*x + b

def squareRoot(x, a, b, c):
    return a*(x + b)**.5 + c

def inverse(x, a, b, c):
    return a/(x + b) + c

def invSq(x, a, b, c):
    return a/(x + b)**2 + c

def invSqrt(x, a, b, c, d):
    return a/(x + c)**.5 + b

def invPwr(x, a, b, c, d):
    return a/abs(x + b)**c + d

def invDecay(x, a, b, c, d):
    return a/(x + d)*exp(-b*x) + c

def decay(x, a, b, c, d):
    return a*e**(b*(x + c)) + d

def slog(x, a, b, c):
    return a*log(x + b) + c

#---------------------------- PHYSICAL CONSTANTS ------------------------------

m = 5.109989461e5  # mass of electron (eV)
M = 9.3827231e8  # mass of proton (eV)
c  = 299792458  # speed of light (m/s)
a0 = 5.29177e-11  # Bohr radius (m)
ev = 1.60217662e-19  # electronVolt (J)
e0 = 8.85418782e-12  # permittivity of free space (SI)
kb = 8.6173303e-5  # Boltzmann constant (eV/K)
alpha = 1/137.035999084  # fine structure constant (unitless)
hbar = 6.582119569e-16  # planck constant (eV s)

#----------------------------- UNIT CONVERSIONS -------------------------------

invÅtoEV = 1e10*c*hbar
invÅtomeV = 1e10*c*hbar*1000
invEVSqtoÅSq = hbar**2*c**2*1e20

#---------------------------------- SCRATCH -----------------------------------

#     maxp_list = []
#     e0_list = []
#     k2_l2 = []
#     k3_l2 = []
#     n2_list = []
#     n3_list = []
# 
#         e0_list.append(e0)
#         k2_l2.append(maxk2_ar)
#         k3_l2.append(maxk3_ar)
#         n2_list.append(n2)
#         n3_list.append(n3)
