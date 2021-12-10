from numpy import array, linspace, concatenate, zeros
from numpy import log, exp
from scipy.optimize import curve_fit
from scipy.special import factorial
from subprocess import call

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

# include date on outfile file names
from datetime import date
from shutil import copyfile

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

# constants, unit conversions, and dictionaries
from constants import *

# VASP modules
from vasp.plotLPZ import plotCompLPZ

#------------------------------------------------------------------------------
# plotPVsEb

color_dict = {'hBN': 'tab:green', 'MoS2': 'tab:blue'}
linestyle_dict = {'hBN': '-', 'MoS2': '-'}
guess_dict = {
#         'hBN': (5.78, -1.07, 0.06),
        'hBN': (5.78, -1.07, 0.06),
        'MoS2': (47., -2.90, 0.40),
        }
writeGuess_dict = {'hBN': (147.0, 0.400, 0.002, 1.5),
        'MoS2': (2426., 2.256, 0.0003, 1.8)}

def plotSVsEb(outfile='sVsEb.pdf', ebounds=[0,100], pbounds=[0,1]):
    """Creates sVsEb figure (formmerly pVxEb.png)."""
    Plot = ProbPlot()
#     Plot.addP('hBN', color='tab:green', label='hBN', linestyle='-',
#             ebounds=ebounds)
    Plot.addP('hBN', color=None, label='hBN', linestyle='-', ebounds=ebounds)
#     Plot.addP('MoS2', color='tab:blue', label='MoS$_2$', linestyle='-',
#             ebounds=ebounds)
#     Plot.decorate(ebounds=ebounds)
    Plot.decorate(ebounds=ebounds, pbounds=pbounds,
            ylabel=r'S$_\infty(\epsilon_b)$')

    # Add arrow to show connection to S_\infty convergence plot
    S = Plot.s_ar[7]  # Eb=40
    Plot.ax.plot([0, 40], [S, S], ls='--', lw=2, color='tab:green')
    Plot.save(outfile=outfile)

    Plot.show()
    Plot.close()


class ProbPlot(object):
    """
    class to add p vs eb curves to single figure
    """
    def __init__(self, fig=None, ax=None, figsize=(6,5)):
        """
        intializes figure and axes
        """
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def addP(self,
            mat = 'hBN',
            color = 'tab:green',
            label = 'hBN',
            linestyle = '-',
            ebounds = [0, 100],

            infile = 'pVsEb.txt',
            headinfile = 'head.txt',
#             root1 = '/home/yoshia/research/radEffects/materials',
            root1 = '/g/g16/yoshia/scripts/python/eIrrad/plots/sum',
            root2 = 'Eb',
            ):
        """
        plots probability as a function of beam energy from to
        mat: material for which p is plotted (list of str)
            * materials are keys in calcdispcross dictionaries
        root1: path from / to material directory (str)
        root2: path from material directory to eb (str)
        """
        print('{}:'.format(mat))
        inpath = '{}/{}/{}/{}'.format(root1, mat, root2, infile)
        print('inpath = {}'.format(inpath))
        guess = guess_dict[mat]
        writeguess = writeGuess_dict[mat]

        try:
            Eb_ar, p_ar, s_ar, r_ar = readPVsEb(infile=inpath)  # ev
        except FileNotFoundError:
            print('could not find {}. calculating from raw data' \
                    .format(infile))
            writeroot = '{}/{}/{}'.format(root1, mat, root2)
            print('writeroot = {}'.format(writeroot))
            Eb_ar, p_ar, s_Er, r_ar = writePVsEb(infile=headinfile,
                    outfile=inpath, root=writeroot, guess=writeguess,
                    )  # ev
    
        Eb_ar /= 1000  # kev
    
        # fit s to inverse
        a, b, c = curve_fit(inverse, Eb_ar, s_ar, p0 = guess)[0]
        print("fitted parameters:\n\ta = {:.10g}, b = {:.10g}, c = {:.10g}"
                .format(a, b, c))
    
        # coefficient of determination
        mean = s_ar.sum() / s_ar.size
        ss_tot = ((s_ar - mean)**2).sum()
        ss_res = ((s_ar - inverse(Eb_ar, a, b, c))**2).sum()
        R = 1 - ss_res / ss_tot
        print("coefficient of determination:\n    R = {:.10g}".format(R))
        
        # curve based on fitted parameters
        x_ar = linspace(ebounds[0], ebounds[1], 500)
    
        # fit p to 1 - exp^expinverse
        self.ax.plot(Eb_ar, s_ar, 'o', color = color, label='data')
#         self.ax.plot(Eb_ar, p_ar, 'o', color = color)
#         fit_ar = expinverse(x_ar, a, b, c)
        fit_ar = inverse(x_ar, a, b, c)
#         fit_ar = expinvdecay(x_ar, a, b, c, d)

        # plot fitted curve
        label = 'R = {:.4g}'.format(R)
        self.ax.plot(x_ar, fit_ar, lw=2, label=label, color=color,
                linestyle=linestyle)

        self.s_ar = s_ar

    def decorate(self,
            ebounds = [0, 100],
            pbounds = [0, 1.05],
            xlabel = 'beam energy (kev)',
            ylabel = 'excitation probability',
            forslides = False,
            xticks = [0, 50, 100],
            yticks = [0, 1],
            legend = True,
            grid = True,
            ):
        """
        adds plot attributes
        """
        self.ax.set_xlim(ebounds)
        self.ax.set_ylim(pbounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14)
        plt.tight_layout()

        # axes for slides
        if forslides:
            for spine in self.ax.spines:
                self.ax.spines[spine].set_color('gray')
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
    
            # white ticks labels
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.set_xlabel(xlabel, fontsize=20)
            self.ax.set_ylabel(ylabel, fontsize=20)
            self.ax.tick_params(axis='x', colors='gray', labelsize=14)
            self.ax.tick_params(axis='y', colors='gray', labelsize=14)
            self.ax.tick_params(axis='x', labelsize=16)
            self.ax.tick_params(axis='y', labelsize=16)
            self.ax.set_xticks(xticks)
            self.ax.set_yticks(yticks)

    def save(self,
            outfile = 'pVsEb.pdf',
            dest = '.',
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        plots probability as a function of beam energy from to
        dest: directory to which plot is saved (str)
        """
        if dest == '.':
            outpath = outfile
        else:
            outpath = '{}/{}'.format(dest, outfile)
        plt.savefig(outpath, dpi=dpi, transparent=transparent)
    
        # make copy with date in name
        if writedate:
            today = date.today()
            year = today.year - 2000
            month = today.month
            day = today.day
        
            name, ext = outpath.split('.')
            outpathcopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month, day, ext)
        
            copyfile(outpath, outpathcopy)

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return ProbPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()


def readPVsEb(infile='pVsEb.txt'):
    """
    returns Eb_ar and P_ar read in from infile
    """
    Eb_list = [] 
    P_list = []
    S_list = []
    R_list = []
    with open(infile) as f:
        f.readline()
        for line in f:
            Eb, P, S, R = [float(val) for val in line.split()]
            Eb_list.append(Eb)  # eV
            P_list.append(P)
            S_list.append(S)
            R_list.append(R)

    return array(Eb_list), array(P_list), array(S_list), array(R_list)  # eV


def writePVsEb(
        infile = 'head.txt',
        outfile = 'pVsEb.txt',
        root = '.',
        getData = True,
        guess = (2426., 2.256, 0.0003, 1.8),  # MoS2
        ):
    """
    writes prob as a function of Eb using k-point convergence fit
        * run in directory containing k-convergences for each Eb
    infile: summary file containing total prob for each k-point and Eb (str)
    outfile: file to which prob and Eb is written (str)
    """
    P_list = []
    S_list = []
    R_list = []
    Eb_list = []
    top_list = next(os.walk(root))[1]
    top_list.sort()

    for top in top_list:
        roottop = '{}/{}'.format(root, top)
        try:
            P, S, R = getFittedP(infile, roottop, guess)
            Eb = float(top)
            P_list.append(P)
            S_list.append(S)
            R_list.append(R)
            Eb_list.append(Eb)
        except ValueError:
            continue

    with open(outfile, 'w') as f:
        f.write('{:<12}{:<16}{:<16}{}\n'.format('Eb (eV)', 'P', 'S', 'R'))
        for Eb, P, S, R in zip(Eb_list, P_list, S_list, R_list):
            f.write('{:<12.10g}{:<16.10f}{:<16.10f}{:.10f}\n'
                    .format(Eb, P, S, R))

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

    # return arrays
    if getData:
        return array(Eb_list), array(P_list), array(S_list), array(R_list)


def getFittedP(
        infile = 'head.txt',
        root = '.',
        guess = (147.0, 0.400, 0.002, 2.5),
        ):
    """
    returns asymptotic probability for infinitely dense k-point mesh
        * fits 'total sum' vs 'nk' to inverse decay function
    infile: output summary file written by calcProb.writeProb (str)
    root: where to look for infile (str)
    guess: guess parameters for inverse decay fit (list of 4 floats)
    """
    print('{}:'.format(root))
    d_list = next(os.walk(root))[1]
    d_list.sort()

    S_list = []
    nk_list = []
    for d in d_list:
        try:
            path = '{}/{}/{}'.format(root, d, infile)
            with open(path) as f:
                for line in f:
                    if 'nk' in line:
                        nk_list.append(int(line.split()[-1]))
    
                    if 'total sum' in line:
                        for word in line.split():
                            if word[0].isdigit():
                                S = float(word)
                        S_list.append(S)
                        break
        except FileNotFoundError:
            continue

    nk_ar = array(nk_list)
#     P_ar = array(P_list)
    S_ar = array(S_list)

    # get inverse power fit
    a, b, c, S = curve_fit(invDecay, nk_list, S_list, p0 = guess)[0]
    print("fitted parameters:\n\ta = {:.4g}, b = {:.4g}, c = {:.4g}, "
            "S = {:.4g}".format(a, b, c, S))

    # coefficient of determination
    mean = S_ar.sum() / S_ar.size
    ss_tot = ((S_ar - mean)**2).sum()
    ss_res = ((S_ar - invDecay(nk_ar, a, b, c, S))**2).sum()
    R = 1 - ss_res / ss_tot
    print("coefficient of determination:\n\tR = {:.10g}".format(R))

    return 1 - exp(-S), S, R

# plotPVsEb
#------------------------------------------------------------------------------
# plotPnVsEb (DEPRECATED. USE plotExcProb.py)

fit_dict = {
        'hBN': [5.462949624, -0.8497685369, 0.08390400208],
        'MoS2': [49.0586673, -3.867010425, 0.3546924901],  # R = 0.9999947621
        'arm': [5.462949624, -0.8497685369, 0.08390400208],
        }

def plotMoS2Zoom(
        outfile = 'MoS2PVsEbZoom.pdf',
        ebounds = [0, 20],
        pbounds = [0, 0.4],
        view = True
        ):
    plotMoS2(outfile=outfile, ebounds=ebounds, pbounds=pbounds, view=view)
    

def plotMoS2(
        outfile='MoS2PVsEb.pdf',
        glinestyle = '--',
        elinestyle = '-',
        gcolor = 'black',
        N = 4,
        colormap = 'plasma',
        offset = -5,  # indexing from 1
        multiplier = 10,
        ebounds = [0, 100],
        pbounds = [0, 1.0],
        view = False,

        forslides = False,
        grid = False,
        transparent = True,
        legend = True,
        ):
    """plots probabilities of n = 0 thru 4 excitations vs Eb."""
    Plot = NPlot()
    Plot.addProb(mat='MoS2', n=0, label=0, linestyle='--', ebounds=ebounds,
            color=gcolor)

    cmap = plt.get_cmap(colormap, 10*N)
    for n in range(1, N+1):
        Plot.addProb(mat='MoS2', n=n, label='$n_i$={}'.format(n),
                linestyle='-', ebounds=ebounds,
                color=cmap(multiplier*n + offset))

    Plot.decorate(ebounds=ebounds, pbounds=pbounds, forslides=forslides,
            grid=grid, legend=legend)
    Plot.save(outfile, view=view, transparent=transparent)
    Plot.show()


def plotHBNZoom(
        outfile = 'hBNPVsEbZoom.pdf',
        ebounds = [0, 20],
        pbounds = [0, 0.4],
        view = True
        ):
    plotHBN(outfile=outfile, ebounds=ebounds, pbounds=pbounds, view=view)
    

def plotHBN(
        outfile='hBNPVsEb.pdf',
        glinestyle = '--',
        elinestyle = '-',
        gcolor = 'black',
        N = 4,
        colormap = 'plasma',
        offset = -5,
        multiplier = 10,
        ebounds = [0, 100],
        pbounds = [0, 1.05],
        view = False,
        
        forslides = False,
        grid = False,
        transparent = True,
        legend = True,
        ):
    """Plots probabilities of n = 0 thru 4 excitations vs Eb."""
    Plot = NPlot()
    Plot.addProb(mat='hBN', n=0, label=0, linestyle='--', ebounds=ebounds,
            color=gcolor)

    cmap = plt.get_cmap(colormap, 10*N)
    for n in range(1, N+1):
        Plot.addProb(mat='hBN', n=n, label='$n_i$={}'.format(n), linestyle='-',
                ebounds=ebounds, color=cmap(multiplier*n + offset))

    Plot.decorate(ebounds=ebounds, pbounds=pbounds, legend=legend, grid=grid,
            forslides=forslides)
    Plot.save(outfile, view=view, transparent=transparent)
    Plot.show()

class NPlot(object):
    """
    class to add P(n) vs Eb curves to single figure
    """
    def __init__(self, fig=None, ax=None, figsize=(6,5)):
        """
        intializes figure and axes
        """
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def addProb(self,
            mat = 'hBN',
            n = 1,
            label = 1,
            color = 'tab:green',
            linestyle = '-',
            ebounds = [0, 100],  # keV
            ):
        """
        plots probability as a function of beam energy from to
        mat: material for which prob is plotted (str)
            * materials are keys in fit_dict
        n: number of excited electrons (nonneg int)
        """
        Eb_ar = linspace(ebounds[0], ebounds[1], 500)  # keV
        A, B, C = fit_dict[mat]
        S_ar = A/(Eb_ar - B) + C  # A, B, C were fitted to keV
        P0_ar = S_ar**n*exp(-S_ar)/factorial(n)  # prob of n excitations

        self.ax.plot(Eb_ar, P0_ar, lw=2, label=n, color=color,
                    linestyle=linestyle)

    def decorate(self,
            ebounds = [0, 100],
            pbounds = [0, 1.05],
            xlabel = 'beam energy (kev)',
            ylabel = 'excitation probability',
            forslides = False,
            xticks = [0, 50, 100],
            yticks = [0, 1],
            legend = True,
            grid = True,
            ):
        """
        adds plot attributes
        """
        self.ax.set_xlim(ebounds)
        self.ax.set_ylim(pbounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14)
        plt.tight_layout()

        # axes for slides
        if forslides:
            for spine in self.ax.spines:
                self.ax.spines[spine].set_color('gray')
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
    
            # white ticks labels
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.set_xlabel(xlabel, fontsize=20)
            self.ax.set_ylabel(ylabel, fontsize=20)
            self.ax.tick_params(axis='x', colors='gray', labelsize=14)
            self.ax.tick_params(axis='y', colors='gray', labelsize=14)
            self.ax.tick_params(axis='x', labelsize=16)
            self.ax.tick_params(axis='y', labelsize=16)
            self.ax.set_xticks(xticks)
            self.ax.set_yticks(yticks)

    def save(self,
            outfile = 'hBNPVsEb.pdf',
            dest = '.',
            view = True,
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        plots probability as a function of beam energy from to
        dest: directory to which plot is saved (str)
        """
        if dest == '.':
            outpath = outfile
        else:
            outpath = '{}/{}'.format(dest, outfile)
        plt.savefig(outpath, dpi=dpi, transparent=transparent)
    
        # make copy with date in name
        if writedate:
            today = date.today()
            year = today.year - 2000
            month = today.month
            day = today.day
        
            name, ext = outpath.split('.')
            outpathcopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month, day, ext)
        
            copyfile(outpath, outpathcopy)

        if view:
            call('wsl-open {}'.format(outpath), shell=True)

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return NPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()

# plotPnVsEb
#------------------------------------------------------------------------------
# plotEigs

def plotHBNProjsEx(n=1, outfile='hBNProjVsPosE',
        xbounds=[0,5], ybounds=[-5,2], size=15, view=True, grid=False):
    top = '/home/yoshia/research/ornl/eIrrad/sputter/excited/hBN/{}'.format(n)
    Plot = EigPlot()
    Plot.addEigs(top=top, atom=15, size=size)
    Plot.decorate(xbounds=xbounds, ybounds=ybounds, grid=grid)
    Plot.save(outfile='{}{}.pdf'.format(outfile,n), view=view)
    Plot.close()


def plotMoS2Projs(outfile='MoS2ProjVsPos.pdf', xbounds=[0,5], ybounds=[-3,0],
        size=15, view=True, grid=False):
    top = '/home/yoshia/research/ornl/eIrrad/sputter/ground/MoS2/frozenSpin1'
    Plot = EigPlot()
    Plot.addEigs(top=top, atom=68, size=size)
    Plot.decorate(xbounds=xbounds, ybounds=ybounds, grid=grid)
    Plot.save(outfile=outfile, view=view)
    Plot.close()


def plotMoS2Eigs(outfile='MoS2EigVsPos.pdf', xbounds=[0,5], ybounds=[-3,1],
        view=True):
    top = '/home/yoshia/research/ornl/eIrrad/sputter/ground/MoS2/frozenSpin1'
    Plot = EigPlot()
    Plot.addEigs(top=top, projection=False)
    Plot.decorate(ybounds=ybounds)
    Plot.save(outfile=outfile, view=view)
    Plot.close()


def plotHBNProjs(outfile='hBNProjVsPos.pdf', xbounds=[0,5], ybounds=[-5,2],
        size=15, view=True, grid=False):
    top = '/home/yoshia/research/ornl/eIrrad/sputter/ground/hBN/frozenSpin1'
    Plot = EigPlot()
    Plot.addEigs(top=top, atom=15, size=size)
    Plot.decorate(xbounds=xbounds, ybounds=ybounds, grid=grid)
    Plot.save(outfile=outfile, view=view)
    Plot.close()


def plotHBNEigs(outfile='hBNEigVsPos.pdf', xbounds=[0,5], ybounds=[-5,2],
        view=True):
    top = '/home/yoshia/research/ornl/eIrrad/sputter/ground/hBN/frozenSpin1'
    Plot = EigPlot()
    Plot.addEigs(top=top, atom=15, projection=False)
    Plot.decorate(ybounds=ybounds)
    Plot.save(outfile=outfile, view=view)
    Plot.close()


class EigPlot(object):
    """
    class to add eigenval vs displacement distance curves to single figure
    """
    def __init__(self, fig=None, ax=None, figsize=(6,5)):
        """
        intializes figure and axes
        """
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def addEigs(self,
            top = '/home/yoshia/research/ornl/eIrrad/sputter/ground/hBN/' \
                    'frozenSpin1',
            colormap = 'viridis_r',
            projection = True,
            atom = 15,  # sputtered B from hBN.  S = 68
            size = 10,
            linestyle = '-',
            excited = False,
            calc = 'excited'
            ):
        """
        plots eigenvalues as a function of displacement from pristine site
        mat: material for which p is plotted (list of str)
            * materials are keys in calcdispcross dictionaries
        """
        eig_a2, occ_a2, proj_a2, disp_ar = readProjs(top=top, atom=atom)

        # convert bands into segments that can be colored coded by occupation
        nb, nd = eig_a2.shape
        bands_a3 = zeros((nb, nd, 2))
        bands_a3[:, :, 1] = eig_a2
        bands_a3[:, :, 0] = disp_ar
        points_a4 = bands_a3.reshape(nb, nd, 1, 2)
        segs_a4 = concatenate([points_a4[:, :-1], points_a4[:, 1:]], axis=2)
        segs_a3 = segs_a4.reshape(nb*(nd - 1), 2, 2)

        # get average occupation of each segment as rank 1 array
        mocc_a2 = 0.5*(occ_a2[:, :-1] + occ_a2[:, 1:])
        mocc_ar = mocc_a2.reshape(nb*(nd - 1))

        # plot LineCollection with colormap and colorbar
        lc = LineCollection(segs_a3, cmap=colormap)
        lc.set_array(mocc_ar)
        if projection:
            lc.set_linewidth(.5)
        else:
            lc.set_linewidth(2)
        line = self.ax.add_collection(lc)
        cb = self.fig.colorbar(line)
        cb.set_label('occupation', fontsize=18)
        cb.ax.invert_yaxis()

        # set linethickness prop to projection onto atom
        if projection:
            lc2 = LineCollection(segs_a3, cmap=colormap)
            lc2.set_array(mocc_ar)
            proj_ar = proj_a2[:, :-1].reshape(nb*(nd - 1))
            lc2.set_linewidth(proj_ar*size)
            self.ax.add_collection(lc2)


    def decorate(self,
            xbounds = [0, 4],
            ybounds = [-4, 2],
            xlabel = 'displacement (Å)',
            ylabel = 'energy (eV)',
            forslides = False,
            xticks = [],
            yticks = [],
            grid = False,
            ):
        """
        adds plot attributes
        """
        self.ax.set_xlim(xbounds)
        self.ax.set_ylim(ybounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        plt.tight_layout()

        # axes for slides
        if forslides:
            for spine in self.ax.spines:
                self.ax.spines[spine].set_color('gray')
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
    
            # white ticks labels
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.set_xlabel(xlabel, fontsize=20)
            self.ax.set_ylabel(ylabel, fontsize=20)
            self.ax.tick_params(axis='x', colors='gray', labelsize=14)
            self.ax.tick_params(axis='y', colors='gray', labelsize=14)
            self.ax.tick_params(axis='x', labelsize=16)
            self.ax.tick_params(axis='y', labelsize=16)
            self.ax.set_xticks(xticks)
            self.ax.set_yticks(yticks)

    def save(self,
            outfile = 'eigVsPos.pdf',
            dest = '.',
            view = True,
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        plots probability as a function of beam energy from to
        dest: directory to which plot is saved (str)
        """
        if dest == '.':
            outpath = outfile
        else:
            outpath = '{}/{}'.format(dest, outfile)
        plt.savefig(outpath, dpi = dpi, transparent=transparent)
    
        # make copy with date in name
        if writedate:
            today = date.today()
            year = today.year - 2000
            month = today.month
            day = today.day
        
            name, ext = outpath.split('.')
            outpathcopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month, day, ext)
        
            copyfile(outpath, outpathcopy)

        if view:
            call('wsl-open {}'.format(outpath), shell=True)

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return EigPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()


def readEigs(
        infile = 'OUTCAR',
        top = '/home/yoshia/research/radEffects/materials/hBN/sputter/' \
            'frozenSpin1',
        ):
    """
    returns eigenvalues read from OUTCARs for systems of various displacements
        * file structure of working directory: {displacement in Å}/OUTCAR
    """
    d_list = next(os.walk(top))[1]
    d_list.sort()
    disp_ar = array([float(val) for val in d_list])

    eig_l2 = []
    occ_l2 = []
    for d in d_list:
        with open('{}/{}/{}'.format(top, d, infile)) as f:
            for line in f:
                if 'NBANDS' in line:
                    nb = int(line.split()[-1])

                elif 'band energies' in line:
                    eig_list = []
                    occ_list = []
                    for _ in range(nb):
                        n, eig, occ = [float(val) for val in
                                f.readline().split()]
                        eig_list.append(eig)
                        occ_list.append(occ)
                    break

            eig_l2.append(eig_list)
            occ_l2.append(occ_list)

    return array(eig_l2).transpose(), array(occ_l2).transpose(), disp_ar


def readProjs(
        infile = 'PROCAR',
        top = '/home/yoshia/research/ornl/eIrrad/sputter/excited/hBN/1',
        atom = 15,
        ):
    """
    returns eigenvalues read from OUTCARs for systems of various displacements
        * file structure of working directory: {displacement in Å}/OUTCAR
    atom: sputtered atom index starting from 1 (pos int)
    """
    d_list = next(os.walk(top))[1]
    d_list.sort()
    disp_ar = array([float(val) for val in d_list])

    E_l2 = []
    occ_l2 = []
    proj_l2 = []
    for d in d_list:
        E_list = []
        occ_list = []
        proj_list = []
        path = '{}/{}/{}'.format(top, d, infile)
        print('reading from {}'.format(path))
        with open(path) as f:
            for line in f:
                if line[:4] == 'band':
                    _, E, occ = [float(val) for val in line.split()[1::3]]
                    E_list.append(E)
                    occ_list.append(occ)

                elif line[:3] == '{:>3}'.format(atom):
                    proj = float(line.split()[-1])
                    proj_list.append(proj)

        E_l2.append(E_list)
        occ_l2.append(occ_list)
        proj_l2.append(proj_list)

    return array(E_l2).T, array(occ_l2).T, array(proj_l2).T, disp_ar


# plotEigs
#------------------------------------------------------------------------------
# plotKCon

def plotAllKCons():
    """Plots k-convergences for all beam energies.

    Run indirectory containing beam energy directories.
    """
    d_list = [d for d in os.listdir() if '000' in d]
    d_list.sort()

    for d in d_list:
        dkeV = round(float(d)/1000)
        ylabel = rf"$S(N_k, \epsilon_b = {dkeV}$ keV)"
#         ylabel = r"$S(N_k, \epsilon_b = {}$ keV)".format(dkeV)
#         print(ylabel)
        plotKCon(top=d, ylabel=ylabel)

        
def plotKCon(
        infile = 'head.txt',  # use head201106.txt for older hBN
        top = '.',
        
        figsize = (6,5),
        save = True,
        ylabel = '$S$',
        outfile = 'kCon.pdf',
        title = None,

        ybounds = [0.,1.],
        guess = (147.0, 0.400, 0.002, 1.5),
        ):
    """Plots probability vs number of k-points.

    Run in directory containing k-point convergence runs
    """
    d_list = [d for d in os.listdir(top) if 'x01' in d and '._' not in d]
    d_list.sort()

    P_list = []
    S_list = []
    nk_list = []
    for d in d_list:
        try:
            with open(f'{top}/{d}/{infile}') as f:
                print(f'd = {d}')
                for line in f:
                    if 'nk' in line:
                        nk_list.append(int(line.split()[-1]))
    
                    if 'total p' in line:
                        for word in line.split():
                            if word[0].isdigit():
                                prob = float(word)
                        P_list.append(prob)

                    if 'total s' in line:
                        for word in line.split():
                            if word[0].isdigit():
                                S = float(word)
                        S_list.append(S)
                        break
        except FileNotFoundError:
            continue

    nk_ar = array(nk_list)
    P_ar = array(P_list)
    S_ar = array(S_list)

    # plot
    fig, ax = plt.subplots(figsize = figsize)

    ax.set_xlabel('number of k-points', fontsize = 18)
    ax.set_ylabel(ylabel, fontsize = 18)
    if ybounds == 'auto':
        ymax = 1.05*max(S_ar)
        ax.set_ylim(0, ymax)
    else:
        ax.set_ylim(ybounds)
    ax.set_xlim(0, nk_ar.max())
        
    ax.set_title(title, fontsize = 16)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax.grid()

    # fit S to inverse decay
    ax.plot(nk_list, S_list, 'o', label = 'data')
    a, b, c, d = curve_fit(invDecay, nk_list, S_ar, p0 = guess)[0]
    print("fitted parameters:\n\ta = {:.4g}, b = {:.4g}, c = {:.4g}, "
            "d = {:.4g}".format(a, b, c, d))

    # coefficient of determination
    mean = S_ar.sum() / S_ar.size
    ss_tot = ((S_ar - mean)**2).sum()
    ss_res = ((S_ar - invDecay(nk_ar, a, b, c, d))**2).sum()
    R = 1 - ss_res / ss_tot
    print("coefficient of determination:\n    R = {:10g}".format(R))
    
    # curve based on fitted parameters
    x_ar = linspace(nk_list[0], nk_list[-1], 200)
    fit_ar = invDecay(x_ar, a, b, c, d)

    # show S_\infty
    ax.axhline(y=d, ls='dashed', lw=2, color='tab:green')
    ax.text(x_ar.mean(), d-0.04, r"S$_\infty$", ha='center', va='top',
            fontsize=20, color='tab:green')

    ax.plot(x_ar, fit_ar, lw=2, label='R = {:.4g}'.format(R))
    ax.legend(fontsize=14)

    plt.tight_layout()
    if save:
        plt.savefig(f'{top}/{outfile}', dpi = 300)
    plt.show()

# plotKCon
#----------------------------- FITTING FUNCTIONS ------------------------------

def inverse(x, a, b, c):
    return a/(x - b) + c

def expInverse(x, a, b, c):
    return 1 - exp(-a/(x - b) - c)

def invDecay(x, a, b, c, d):
    return a/(x - b)*exp(-c*x) + d

def expInvDecay(x, a, b, c, d):
    return 1 - exp(-a/(x - b)*exp(-c*x) - d)

def asymLog(x, a, b, c):
    return a - log(b + 1/(x-c)**4)

#----------------------------- Z-LOCAL POTENTIAL ------------------------------

def plotS(
        outfile='SLPZ.pdf',
        root = '/home/yoshia/research/ornl/atoms/lda/S',
        view = False,
        ):
    plotCompLPZ(outfile=outfile, root=root, view=view)


def plotB(
        outfile = 'BLPZ.pdf',
        root = '/home/yoshia/research/ornl/atoms/lda/B',
        view = False,
        ):
    plotCompLPZ(outfile=outfile, root=root, view=view)


#--------------------------- CALLING FROM TERMINAL ----------------------------

if __name__ == '__main__':
    plotPVsEb()

#---------------------------------- SCRATCH -----------------------------------

# guess_dict = {'hBN': (5.46, -0.85, 0.08, 0.), 'MoS2': (47., -2.90, 0.40, 0.)}
# guess_dict = {'hBN': (5.46, -0.85), 'MoS2': (47., -2.90)}

#             f.readline()
#             nk, nb, ni = [int(val) for val in f.readline().split()[3::4]]
#             for n in range(3):
#                 f.readline()
# 
#             for b in range(nb):

#     Plot.ax.annotate("", color='tab:green', xy=(), ls='dashed')
#     Plot.ax.annotate('',
#                 xy=(40, S), xycoords='data', xytext=(0, S), textcoords='data',
#                 arrowprops=dict(color='tab:green', lw=2, ls='--',
#                 arrowstyle='-', mutation_scale=30)
#                 )
