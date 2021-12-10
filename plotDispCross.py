from numpy import pi, log, cos, sin, sqrt, ceil, exp, inf, abs, append
from numpy import array, linspace, zeros, ones, arange, where, insert
from numpy import newaxis as na
from scipy.optimize import curve_fit
from scipy.special import comb, factorial
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# include date on outfile file names
from datetime import date
from shutil import copyfile

# catch warnings (e.g. RuntimeWarning) as an exception
# import warnings
# warnings.filterwarnings('error')

# Gaussian quadratre
from gaussxw import gaussxw, gaussxwab

# periodic table for atomic masses
from periodic import p_dict

# displacement cross section classes
from dxs.classes import *

# constants, unit conversions, and dictionaries
from constants import *

#--------------------------------- PLOTTING ----------------------------------

class SigmaPlot(object):
    """Class to add cross section curves to single figure"""

    def __init__(self, fig=None, ax=None, figsize=(6, 5)):
        """Initializes figure and axes."""
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def plot(self,
            sigma_dict,
            index = 0,
            linestyle = '-',
            color = 'tab:blue',
            label = 'ground state',
            ):
        """Plots cross sections as functions of TEM energy.

        sigma_dict: dictionary containing plottable data (dict)
        """
        Eb_ar = sigma_dict['Eb_ar']
        sigma_a2 = sigma_dict['sigma_a2'].T
        if len(sigma_a2.shape) == 2:
            sigma_ar = sigma_a2[index]
        else:
            sigma_ar = sigma_a2

        self.ax.plot(Eb_ar, sigma_ar, linewidth=2, zorder=3, color=color,
                    linestyle=linestyle, label=label)

        self.Ebmin = Eb_ar.min()
        self.Ebmax = Eb_ar.max()

    def plotSum(self,
            sigma_dict,
            linestyle = '-',
            color = 'black',
            label = 'total',
            ):
        """Plots cross sections as functions of TEM energy.

        sigma_dict: dictionary containing plottable data (dict)
        """
        Eb_ar = sigma_dict['Eb_ar']
        sigma_ar = sigma_dict['sigma_a2'].sum(1)
        self.ax.plot(Eb_ar, sigma_ar, linewidth=2, zorder=3, color=color,
                    linestyle=linestyle, label=label)

    def plotExpt(self, mat):
        """Plots experimental data.

        from SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
        """
        if mat=='MoS2':
            expEb_ar = array([20, 30, 40, 60, 80])
            expCross_ar = array([6.7, 11.6, 9., 4.2, 5.3])
            expErr_ar = array([0.5, 0.5, 1.0, 0.4, 0.4])
        elif mat=='hBN':
            expEb_ar = array([30, 60])
            expCross_ar = array([20, 20])

        self.ax.plot(expEb_ar, expCross_ar, 's', label='expt', color='black',
                zorder=10)

    def decorate(self,
            ebounds = 'auto',
            cbounds = None,
            xlabel = 'beam energy (keV)',
            ylabel = 'cross section (barn)',
            forSlides = False,
            xticks = [0, 50, 100],
            yticks = [0, 50],

            legend = True,
            loc = 'upper right',
            ncol = 2,
            bbox_to_anchor = (1, 1),
            handleheight=None,
            handlelength=None,

            grid = False,
            ):
        """
        adds plot attributes
        ebounds: TEM energy range in keV (list of two floats)
        cbounds: cross section range in barn (list of two floats)
        """
        # boundaries
        if ebounds=='auto':
            ebounds=(self.Ebmin, self.Ebmax)
        self.ax.set_xlim(ebounds)
        self.ax.set_ylim(cbounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14, bbox_to_anchor=bbox_to_anchor, loc=loc,
                    handleheight=handleheight, handlelength=handlelength,
                    ncol=ncol,
                    )
        plt.tight_layout()

        # axes for slides
        if forSlides:
            for spine in self.ax.spines:
                self.ax.spines[spine].set_color('gray')
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
    
            # white ticks labels
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.set_xlabel(xlabel, fontsize=20)
            self.ax.set_ylabel(ylabel, fontsize=20)
            self.ax.tick_params(axis = 'x', colors = 'gray', labelsize = 14)
            self.ax.tick_params(axis = 'y', colors = 'gray', labelsize = 14)
            self.ax.tick_params(axis = 'x', labelsize = 16)
            self.ax.tick_params(axis = 'y', labelsize = 16)
            self.ax.set_xticks(xticks)
            self.ax.set_yticks(yticks)
            
    def save(self,
            outfile = 'cross.pdf',
            dest = '.',
            view = False,
            dpi = 300,
            transparent = False,
            writedate = True,
            ):
        """
        format and save cross section plot
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
            outpathCopy = '{}{:02d}{:02d}{:02d}.{}'.format(name, year, month,
                    day, ext)
        
            copyfile(outpath, outpathCopy)

        if view:
            call('wsl-open {}'.format(outpath), shell=True)
    
    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return SigmaPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()

#------------------------------- SPECIAL CASES -------------------------------

def plotMoS2(
        outfile = 'MoS2.pdf',
        save = True,
        ebounds = [10, 95],
        cbounds = [0, 30],
        colormap = 'plasma',
        res = 400,
        tau = 81,  # fs
        offset = 5,  # indexing from 1
        multiplier = 10,
        NEd = 4,
        ):
    """Plots MoS2 figure.

    NEd: number of excitations plotted individually
        * All nf >= NEd are summed over
    """
    Plot = SigmaPlot()

    # get excited simulation
    MoS2Cross = ExciDispCross(mat='MoS2', tau=tau, res=res, ebounds=ebounds)
    MoS2_dict = MoS2Cross.getDict()
    sigma_a2 = MoS2_dict['sigma_a2'].T
    Eb_ar = MoS2_dict['Eb_ar']

    # get ground state simulation
    MoS20Cross = TempDispCross(mat='MoS2', res=res, ebounds=ebounds)
    MoS20_dict = MoS20Cross.getDict()
    sigma0_ar = MoS20_dict['sigma_a2'][:, 0]

    # plot sum of n>3 excitations
    cmap = plt.get_cmap(colormap, NEd*multiplier) 
    sigma_ar = sigma_a2[NEd-1:].sum(0)
    Plot.ax.plot(Eb_ar, sigma_ar, linewidth=2,
            color = cmap(multiplier*3 + offset), label='$n_f>{}$'.format(NEd-2))

    # inidividual excitation contributions
    for nf, sigma_ar in reversed(list(enumerate(sigma_a2[:NEd-1]))):
        color = cmap(multiplier*nf)
        Plot.ax.plot(Eb_ar, sigma_ar,
                linewidth=2, linestyle='-',
                color=color, label='$n_f={}$'.format(nf))

    # plot ground state cross section
    Plot.ax.plot(Eb_ar, sigma0_ar, linewidth=2, zorder=3,
            label='ground', color='k', linestyle='--')
    
    # plot expt data from SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
    exptEb_ar, exptSigma_ar, exptErr_ar = expt_dict['MoS2']
    Plot.ax.plot(exptEb_ar, exptSigma_ar, 's',
            color='k', markersize=7, label='expt')

    # text
    Plot.ax.text(
            .02, .98,
            'MoS$_2$ (300 K)',
            fontsize=18,
            ha='left', va='top',
            transform=Plot.ax.transAxes)

    Plot.decorate(ebounds=(round(ebounds[0]), ebounds[1]), cbounds=cbounds,
            bbox_to_anchor=(0, .92), loc='upper left')

    if save:
        Plot.save(outfile=outfile, view=False)
    Plot.show()


def plotHBN(
        outfile = 'edgeBigLegend.pdf',
        save = True,
        ebounds = [20,100],
        cbounds = [0,60],
        tau = 240,  # fs
        res = 400,
        T = 1000 + 273.15, # K

        handleheight=1.9,
        handlelength=3.5,
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
    Eb_ar = Barm_dict['Eb_ar']

    # get ground state simulation
    Barm0Cross = TempDispCross(mat='Barm', T=T, ebounds=ebounds, res=res)
    Narm0Cross = TempDispCross(mat='Narm', T=T, ebounds=ebounds, res=res)
    Barm0_dict = Barm0Cross.getDict()
    Narm0_dict = Narm0Cross.getDict()
    sigmaB0_ar = Barm0_dict['sigma_a2'][:, 0]
    sigmaN0_ar = Narm0_dict['sigma_a2'][:, 0]

    # plot simulation
    Plot.ax.plot(Eb_ar, sigmaN_ar+sigmaB_ar, lw=2, zorder=3, color='k',
            label='T',
#             label='total',
            )
    Plot.ax.plot(Eb_ar, sigmaB_ar, color='tab:green', lw=2,
            label='B',
            )
    Plot.ax.plot(Eb_ar, sigmaN_ar, color='tab:blue', lw=2,
            label='N',
            )

    # plot ground state simulation
    Plot.ax.plot(Eb_ar, sigmaN0_ar+sigmaB0_ar, lw=2, zorder=3, color='k', ls='--',
            label='t',
#             label='total (ground)',
            )
    Plot.ax.plot(Eb_ar, sigmaB0_ar, color='tab:green', ls='--', lw=2,
            label='b',
#             label='B (ground)',
            )
    Plot.ax.plot(Eb_ar, sigmaN0_ar, color='tab:blue', ls='--', lw=2,
            label='n',
#             label='N (ground)',
            )

    # plot experiment
    try:
        expt_t2 = hBNTemp_dict[T]
        Plot.ax.plot(expt_t2[0], expt_t2[1], 's', color='k', markersize=7,
#                 label='e',
#                 label='expt',
                )
    except KeyError:
        pass

    # text
    Plot.ax.text(
            .02, .98,
#             .5, .98,
            'hBN (1273 K)', 
            fontsize=18,
            ha='left', va='top',
#             ha='center', va='top',
            transform=Plot.ax.transAxes)

    Plot.decorate(ebounds=(round(ebounds[0]), ebounds[1]), cbounds=cbounds,
#             bbox_to_anchor=(.5, .92), loc='upper center')
            bbox_to_anchor=(.06, .92), loc='upper left', ncol=2,
            handleheight=handleheight, handlelength=handlelength,
            )

    if save:
        Plot.save(outfile=outfile, view=False)
    Plot.show()


def statPlot(cbounds=[0,100]):
    """Plots static displacement cross sections."""
    MoS2Cross = sc.DispCross(mat='MoS2')
    MoS2_dict = MoS2Cross.getDict()
    hBNCross = sc.DispCross(mat='hBN')
    hBN_dict = hBNCross.getDict()
    Plot = SigmaPlot()
    Plot.plot(MoS2_dict, label='MoS2', color='tab:blue')
    Plot.plot(hBN_dict, label='hBN', color='tab:green')
    Plot.decorate(cbounds=cbounds)
    Plot.show()


def tempPlot(cbounds=[0,100]):
    """Plots temp-dependent displacement cross sections."""
#     MoS2_dict = tc.getSigmaDict(mat='MoS2')
#     hBN_dict = tc.getSigmaDict(mat='hBN')
    MoS2Cross = tc.TempDispCross(mat='MoS2')
    MoS2_dict = MoS2Cross.getDict()
    hBNCross = tc.TempDispCross(mat='hBN')
    hBN_dict = hBNCross.getDict()
    Plot = SigmaPlot()
    Plot.plot(MoS2_dict, label='MoS2', color='tab:blue')
    Plot.plot(hBN_dict, label='hBN', color='tab:green')
    Plot.decorate(cbounds=cbounds)
    Plot.show()


def exciPlot(cbounds=[0,100], tau=200):
    """Plots excitation facilitated displacement cross sections."""
    MoS2Cross = ec.ExciDispCross(mat='MoS2', tau=tau)
    MoS2_dict = MoS2Cross.getDict()
    oldMoS2_dict = ec.getSigmaDict(mat='MoS2', tau=tau)
    oldhBN_dict = ec.getSigmaDict(mat='hBN')
    hBNCross = ec.ExciDispCross(mat='hBN')
    hBN_dict = hBNCross.getDict()
    Plot = SigmaPlot()
    Plot.plot(MoS2_dict, label='MoS2', color='tab:blue')
    Plot.plot(hBN_dict, label='hBN', color='tab:green')
    Plot.plot(oldMoS2_dict, color='tab:orange', linestyle='--')
    Plot.plot(oldhBN_dict, color='tab:red', linestyle='--')
    Plot.decorate(cbounds=cbounds)
    Plot.show()

#---------------------------------- SLIDERS ----------------------------------

class ExciSliders(object):
    """
    class to add streak curves (time vs. breakout angle) to single figure
    """
    def __init__(self, mat='MoS2', ebounds=[20,160], track=0,
            fig=None, ax=None, figsize=(7,7)):
        """
        intializes figure and axes
        mat: irradiated material (str)
        ebounds: TEM energy range in keV (list of two floats)
        track: excitation number to which ylim is scaled (int >= 0)
        """
        self.mat = mat
        self.ebounds = ebounds
        self.track = track
        self.T = 1  # K
        self.tau = 50  # fs
        DispCross = ExciDispCross(mat=self.mat, tau=self.tau, T=self.T)
        self.sigma_dict = DispCross.getDict()
        self.Eb_ar = self.sigma_dict['Eb_ar']

        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

        # make room for slider
        plt.subplots_adjust(bottom=0.3)

        # sliders
        self.tax = plt.axes([0.15, 0.15, 0.7, 0.03])
        self.tslider = Slider(self.tax, '$\tau$ (fs)', 1, 1000,
                valinit=self.tau, valstep=1)
        self.tslider.on_changed(self.update)

        # sliders
        self.Tax = plt.axes([0.15, 0.10, 0.7, 0.03])
        self.Tslider = Slider(self.Tax, 'T (K)', 1, 1000,
                valinit=self.T, valstep=1)
        self.Tslider.on_changed(self.update)

    def plot(self):
        """
        plots cross sections as functions of TEM energy
        sigma_dict: dictionary containing plottable data (dict)
        """
        sigma_a2 = self.sigma_dict['sigma_a2'].T
        self.dat_list = []
        for nf, sigma_ar in enumerate(sigma_a2):
            dat, = self.ax.plot(self.Eb_ar, sigma_ar,
                    linewidth=2, zorder=3, linestyle='-', label=nf)
            self.dat_list.append(dat)
        self.ax.set_ylim([0, sigma_a2[:-1].max()*1.1])

    def plotSum(self):
        """
        plots cross sections as functions of TEM energy
        sigma_dict: dictionary containing plottable data (dict)
        """
        sigma_ar = self.sigma_dict['sigma_a2'].T.sum(1)
        self.dat, = self.ax.plot(self.Eb_ar, sigma_ar, linewidth=2, zorder=3,
                color='tab:blue', linestyle='-', label='prediction')
        self.ax.set_ylim([0, sigma_ar.max()*1.1])

    def plotExpt(self, mat):
        """
        from SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
        """
        if mat=='MoS2':
            expEb_ar = array([20, 30, 40, 60, 80])
            expCross_ar = array([6.7, 11.6, 9., 4.2, 5.3])
            expErr_ar = array([0.5, 0.5, 1.0, 0.4, 0.4])
        elif mat=='hBN':
            expEb_ar = array([30, 60])
            expCross_ar = array([20, 20])

        self.ax.plot(expEb_ar, expCross_ar, 's', label='expt', color='black',
                zorder=10)

    def decorate(self,
            xlabel = 'beam energy (keV)',
            ylabel = 'cross section (barn)',
            forSlides = False,
            xticks = [0, 50, 100],
            yticks = [0, 50],
            legend = True,
            grid = True,
            ):
        """
        adds plot attributes
        cbounds: cross section range in barn (list of two floats)
        """
        # boundaries
        self.ax.set_xlim(self.ebounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14, bbox_to_anchor=bbox_to_anchor, loc=loc)

    def update(self, val):
        self.tau = self.tslider.val
        self.T = self.Tslider.val
        DispCross = ExciDispCross(mat=self.mat, tau=self.tau, T=self.T)
        self.sigma_dict = DispCross.getDict()
        sigma_a2 = self.sigma_dict['sigma_a2'].T
    
        self.ax.set_ylim([0, sigma_a2[:-1].max()*1.1])
        for dat, sigma_ar in zip(self.dat_list, sigma_a2):
            dat.set_ydata(sigma_ar)
    
        self.fig.canvas.draw_idle()


def main(mat='MoS2'):
    SS = ExciSliders(mat=mat)
    SS.plot()
    SS.plotExpt(mat=mat)
    SS.decorate()
    plt.show()

#---------------------------------- SCRATCH ----------------------------------

#     BarmEd_ar = Ed_dict['Barm']
#     NarmEd_ar = Ed_dict['Narm']
#     Plot.plot(Barm_dict, label='armchair', color='tab:blue')
#     Plot.plot(hBN_dict, label='hBN', color='tab:green')
#     for n, Ed in enumerate(BarmEd_ar):
#         Plot.plot(Barm_dict, index=n, color=None, label=n)
#         Plot.plot(Barm_dict, index=n, label='armchair', color='tab:blue')
#         Plot.plot(hBN_dict, index=n, label='hBN', color='tab:green')

#         ediff = ebounds[1] - ebounds[0]
#         self.ax.set_xlim(ebounds[0]-.1*ediff, ebounds[1]+.1*ediff)

def plotColorMap(cbounds=[0,100], colormap='plasma'):

    Plot = SigmaPlot()
    nT = len(hBNTemp_dict)
    cmap = plt.get_cmap(colormap, 700)

    for T, data_t2 in hBNTemp_dict.items():
        color = cmap(int(T - 500 - 273.15))
        label = '{} K'.format(int(T))

        Barm_dict = ec.getSigmaDict(mat='Barm', T=T)
        Narm_dict = ec.getSigmaDict(mat='Narm', T=T)
    
        sigmaB_ar = Barm_dict['sigma_a2'].sum(1)
        sigmaN_ar = Narm_dict['sigma_a2'].sum(1)
        Eb_ar = Barm_dict['Eb_ar']
        Plot.ax.plot(Eb_ar, sigmaN_ar+sigmaB_ar, linewidth=2, zorder=3,
                label=label, color=color)
        Plot.ax.plot(data_t2[0], data_t2[1], 's', color=color)
#         Plot.plotSum(Barm_dict, label='B', color='tab:green')
#         Plot.plotSum(Narm_dict, label='N', color='tab:blue')
#         Plot.plotExpt(mat='hBN')

    Plot.decorate(cbounds=cbounds)
    Plot.show()

# Cretu's unmodified
#         ebounds = [20,105],
#         cbounds = [0,60],
