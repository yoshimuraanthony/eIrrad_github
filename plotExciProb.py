from numpy import arange, linspace, exp
from numpy import newaxis as na
from scipy.special import factorial
import matplotlib.pyplot as plt

# include date on outfile file names
from datetime import date
from shutil import copyfile

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

# constants, unit conversions, and dictionaries (see constants.fit_dict)
from constants import *

#--------------------------------- PLOTTING ----------------------------------

class NPlot(object):
    """Class to add P(n) vs Eb curves to single figure"""
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
            nmax = 3,
            color = 'tab:green',
            linestyle = '-',
            ebounds = [0, 100],  # keV
            ):
        """
        plots probability as a function of beam energy from to
        mat: material for which prob is plotted (str)
            * materials are keys in fit_dict
        n: number of excited electrons (int >= 0)
#         nmax: largest number of excitations with positive Ed (pos int)
        """
        Eb_ar = linspace(ebounds[0], ebounds[1], 500)  # keV
        A, B, C = fit_dict[mat]
        S_ar = A/(Eb_ar - B) + C  # A, B, C were fitted to keV

        # lump all negative Ed's into one event
        if n > nmax:
            n_a2 = arange(nmax+1)[:, na]
            P0_ar = 1 - (S_ar**n_a2*exp(-S_ar)/factorial(n_a2)).sum(0)
            label = '$n_i>{}$'.format(nmax)
        else:
#         P0_ar = S_ar**n*exp(-S_ar)/factorial(n)  # prob of n excitations
            P0_ar = S_ar**n*exp(-S_ar)/factorial(n)  # prob of n excitations
            label = '$n_i={}$'.format(n)

        self.ax.plot(Eb_ar, P0_ar, lw=2, label=label, color=color,
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
            view = False,
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

#------------------------------- SPECIAL CASES -------------------------------

def plotMoS2(
        glinestyle = '--',
        elinestyle = '-',
        gcolor = 'black',
        nmax = 3,
        colormap = 'plasma',
        offset = -3,  # indexing from 1
        multiplier = 10,
        ebounds = [0, 100],
        pbounds = [0, 1],

        save = True,
        outfile='MoS2PVsEb.pdf',
        view = False,

        forslides = False,
        grid = False,
        transparent = False,
        legend = True,
        ):
    """Plots probabilities of n = 0 thru 4 excitations vs Eb."""
    Plot = NPlot()
    Plot.addProb(mat='MoS2', n=0, linestyle='--', ebounds=ebounds,
            color=gcolor)

    cmap = plt.get_cmap(colormap, 10*nmax)
    for n in range(1, nmax+2):
        Plot.addProb(mat='MoS2', n=n, nmax=nmax,
                linestyle='-', ebounds=ebounds,
                color=cmap(multiplier*(n-1) + offset))

    Plot.decorate(ebounds=ebounds, pbounds=pbounds, forslides=forslides,
            grid=grid, legend=legend)

    Plot.ax.text(
#             .04, .98,
            .5, .98,
            'MoS$_2$',
#             ha='left', va='top',
            ha='center', va='top',
            fontsize=24,
            transform=Plot.ax.transAxes)

    if save:
        Plot.save(outfile, view=view, transparent=transparent)
    Plot.show()


def plotHBN(
        outfile='hBNPVsEb.pdf',
        save = True,
        view = False,
        
        glinestyle = '--',
        elinestyle = '-',
        gcolor = 'black',
        nmax = 3,
        colormap = 'plasma',
        offset = -3,
        multiplier = 10,
        ebounds = [0, 100],
        pbounds = [0, 1],

        forslides = False,
        grid = False,
        transparent = False,
        legend = True,
        ):
    """
    plots probabilities of n = 0 thru 4 excitations vs Eb
    """
    Plot = NPlot()
    Plot.addProb(mat='hBN', n=0, linestyle=glinestyle, ebounds=ebounds,
            color=gcolor)

    cmap = plt.get_cmap(colormap, 10*nmax)
    for n in range(1, nmax+2):
        Plot.addProb(mat='hBN', n=n, nmax=nmax,
                linestyle=elinestyle, ebounds=ebounds,
                color=cmap(multiplier*(n-1) + offset))

    Plot.decorate(ebounds=ebounds, pbounds=pbounds, legend=legend, grid=grid,
            forslides=forslides)

    Plot.ax.text(
#             .02, .98,
            .5, .98,
            'hBN',
#             ha='left', va='top',
            ha='center', va='top',
            fontsize=24,
            transform=Plot.ax.transAxes)

    if save:
        Plot.save(outfile, view=view, transparent=transparent)
    Plot.show()


def plotMoS2Zoom(
        outfile = 'MoS2PVsEbZoom.pdf',
        ebounds = [0, 20],
        pbounds = [0, 0.4],
        view = True
        ):
    """Plots probabilities of excitations zoomed into low probabilities."""
    plotMoS2(outfile=outfile, ebounds=ebounds, pbounds=pbounds, view=view)
    

def plotHBNZoom(
        outfile = 'hBNPVsEbZoom.pdf',
        ebounds = [0, 20],
        pbounds = [0, 0.4],
        view = True
        ):
    plotHBN(outfile=outfile, ebounds=ebounds, pbounds=pbounds, view=view)
