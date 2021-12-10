from numpy import array, where, inf
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# include date on outfile file names
from datetime import date
from shutil import copyfile

# constants, unit conversions, and dictionaries (see constants.fit_dict)
from eIrrad.write import writeKpts
from constants import *


def plot(
        density=False,
        title='e-induced excitations in PE',
        outfile='scatter.png',
#         maxlabel = r"8, (1/2, 1/2, 0) $\rightarrow$ 16, (1/2, 1/2, 0)",
        maxlabel = r"8M$\rightarrow$16M"
        ):
    plot = ScatterPlot()
    plot.plot('newprob.txt', density=density, maxlabel=maxlabel)
    if density:
        ylabel = r'probability density ($\AA^6$)',
    else:
        ylabel='excitation probability'
    plot.decorate(ylabel=ylabel, title=title)
    plot.save(outfile=outfile)
    plot.show()


class ScatterPlot(object):
    """Class to plot exctitation probability densities."""
    def __init__(self,
            fig = None,
            ax = None,
            figsize = (6,6),
            ):
        """Intializes figure and axes."""
        mpl.style.use('figure')

        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

        # domain and range boundaries
        self.xmax = 0
        self.ymax = 0
        self.ymin = inf

    def readData(self,
            pinfile = 'prob.txt',
            kinfile = 'kpts.txt',
            ):
        """Read in excitation probabilities."""
        x_list = []  # energy  (eV)
        y_list = []  # probability 
        t_list = []  # transitions
        k_l2 = []  # k-points (eV)
        
        with open(f'{pinfile}') as f:
            for line in f:
                if 'bands' in line and 'val' not in line:
                    nb = float(line.split()[-1])
                elif 'val bands' in line:
                    nv = float(line.split()[-1])
                    nc = nb - nv
                if 'k-points' in line:
                    nk = float(line.split('=')[1].split()[0])
                elif 'area' in line:
                    area = float(line.split('=')[1].split()[0])  # eV^{-2}
                elif 'cell-height' in line:
                    height = float(line.split('=')[1].split()[0])  # A
                elif 'energy (eV)' in line:
                    for line in f:
                        x, y, k2, k3, n2, n3 = [float(val) for val in line.split()]
                        x_list.append(x)
                        y_list.append(y)
                        t_list.append([k2, k3, n2, n3])

        # write kinfile if it doesn't already exist
        writeKpts()
        with open(f'{kinfile}') as f:
            for _ in range(2):
                f.readline()

            for line in f:
                k_l2.append([float(val) for val in line.split()])

        # k-point density
        area *= invEVSqtoÅSq
        kden = nk*area*height/(2*pi)**3  # A^3

        return array(x_list), array(y_list), \
                array(t_list, dtype=int), \
                array(k_l2), nk, nv, nc, kden

    def plot(self,
            pinfile = 'prob.txt',
            kinfile = 'kpts.txt',
            logy = False,
            density = True,
            showmax = True,
            maxlabel = None,
            color = None,
            showvert = False,
            vcolor = 'tab:orange',
            ):
        """Plot probabilities."""
        x_ar, y_ar, t_a2, k_a2, nk, nv, nc, kden = self.readData(pinfile, kinfile)
        if density:
            y_ar *= kden**2  # A^3

        xmax = x_ar.max()
        xlimmult = 1.05
        ymax = y_ar.max()
        ymin = y_ar.min()

        if logy:
            self.ax.semilogy(x_ar, y_ar, 'o', markersize=5, label='nonvertical')
            ylimmult = 10
        else:
            self.ax.plot(x_ar, y_ar, 'o', markersize=5, label='nonvertical')
            ylimmult = 1.1

        if xmax>self.xmax:
            self.xmax = xmax*xlimmult
        if ymax*ylimmult>self.ymax:
            self.ymax = ymax*ylimmult
        if ymin/ylimmult<self.ymin:
            self.ymin = ymin*ylimmult

        if showmax:
            ymax_idx = where(y_ar==ymax)
            x0 = x_ar[ymax_idx]
            k2, k3, n2, n3 = t_a2[ymax_idx][0]
            k2_ar = k_a2[k2-1]
            k3_ar = k_a2[k3-1]
            print(f'vb = {n2}, vk = {k2_ar}\ncb = {n3}, ck = {k3_ar}')

            if maxlabel==None:
                maxlabel = rf"({n2},{k2}) $\rightarrow$ ({n3},{k3})"

            self.ax.annotate(
                    maxlabel,
                    fontsize=16, ha='left', va='center', xy=(x0, ymax),
                    xycoords='data', xytext=(50, 0),
                    textcoords='offset points',
                    arrowprops=dict(facecolor='k', shrink=0.05))

        if showvert:
            vidx_ar = where(t_a2[:,0]==t_a2[:,1])
            vx_ar = x_ar[vidx_ar]
            vy_ar = y_ar[vidx_ar]

            if logy:
                self.ax.semilogy(vx_ar, vy_ar, 'o', markersize=5, color=vcolor,
                        label='vertical')
                ylimmult = 10
            else:
                self.ax.plot(vx_ar, vy_ar, 'o', markersize=5, color=vcolor,
                        label='vertical')
                ylimmult = 1.1

    def decorate(self,
            xlim = 'auto',
            ylim = 'auto',
            xlabel = 'energy tranfer (eV)',
            ylabel = r'probability density ($\AA^6$)',
            title = 'e-induced excitations',
            grid = False,
            ):
        """
        adds plot attributes
        ebounds: TEM energy range in keV (list of two floats)
        cbounds: cross section range in barn (list of two floats)
        """
        if xlim=='auto':
            xlim = (0, self.xmax)
        self.ax.set_xlim(xlim)
        if ylim=='auto':
            ylim = (self.ymin, self.ymax)
        self.ax.set_ylim(ylim)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)

        if grid:
            self.ax.grid()

        if self.showvert:
            self.ax.legend()

        plt.tight_layout()

    def save(self,
            outfile = 'scatter.png',  # pdf too big: 3.3 Mb for PE 10x3x1 kpts
            dest = '.',
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

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return ScatterPlot(fig=self.fig, ax=self.ax)

    def show(self):
        plt.show()


class ScatterSlider(object):
    """Class to plot exctitation probabilities.

    adjustable Eb with sliders
    """
    def __init__(self,
            infile = 'newprob.txt',
            top = '.',
            fig = None,
            ax = None,
            figsize = (7,7),
            Ebinit = 1e4,  # eV
            Ebmax = 11252.9,  # eV  (241Am gamma compton scattering)
            showvert = True,
            ):
        """Intializes figure and axes.

        Ebinit: initial slider value for incident photon energy (eV)
        """
        # Under construction
        # get Ebmax, Ebinit, and valstep from directories
        Eb_list = [float(d) for d in next(os.walk(top))[1] if d.isdigit()]
        Eb_list.sort()
        Eb_ar = array(Eb_list)
        dif_list = Eb_ar[1:] - Eb_ar[:-1]
        
        mpl.style.use('slider')

        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

        # make room for slider axes at bottom of figure
        plt.subplots_adjust(bottom=0.2, left=0.2)

        # slider: ballistic electron energy
        self.Ebax = plt.axes([0.2, 0.05, 0.65, 0.03])
        self.Eb = Ebinit  # initial slider position in eV
        Ebmax_rnd = round((Ebmax - Ebmax%200)/1000)
        self.slider = Slider(self.Ebax, r'$E_b$', 2, Ebmax_rnd, valinit=Ebinit/1000,
                valstep=2)  # keV
        self.slider.valtext.set_text('{:.4g} keV'.format(self.Eb/1000))
        self.slider.label.set_size(14)
        self.slider.valtext.set_size(14)
        self.slider.on_changed(self.update)

        # button: toggle linear or log y-axis
        self.logax = plt.axes([0.05, 0.11, 0.07, 0.05])
        self.linax = plt.axes([0.05, 0.05, 0.07, 0.05])
        self.logbut = Button(self.logax, 'log', hovercolor='0.975')
        self.linbut = Button(self.linax, 'lin', hovercolor='0.975')
        self.logbut.on_clicked(self.uselog)
        self.linbut.on_clicked(self.uselin)
        self.logbut.label.set_size(14)
        self.linbut.label.set_size(14)

        # class variables
        self.infile = infile
        self.Ebdir = '{:0>5}'.format(int(self.Eb))
        self.showvert=False
        if showvert:
            self.showvert=True

        # initial plot
        self.readData()
        self.plot()
        self.show()

    #------------------------------- plot --------------------------------------

    def readData(self):
        """Read in excitation probabilities for each energy transfer."""
        x_list = []
        y_list = []
        k2_list = []
        k3_list = []

        with open('{}/{}'.format(self.Ebdir, self.infile)) as f:
            for line in f:
                if 'energy (eV)' in line:
                    for line in f:
                        x, y, k2, k3 = [float(val) for val in line.split()[:4]]
                        x_list.append(x)
                        y_list.append(y)
                        k2_list.append(k2)
                        k3_list.append(k3)

        self.x_ar = array(x_list)
        self.y_ar = array(y_list)
        k2_ar = array(k2_list)
        k3_ar = array(k3_list)
        self.vidx_ar = where(k2_ar==k3_ar)
        self.vx_ar = self.x_ar[self.vidx_ar]
        self.vy_ar = self.y_ar[self.vidx_ar]

    def readYData(self):
        """Read in excitation probabilities for each energy transfer."""
        with open('{}/{}'.format(self.Ebdir, self.infile)) as f:
            for line in f:
                if 'energy (eV)' in line:
                    for n, line in enumerate(f):
                        self.y_ar[n] = float(line.split()[1])

    def plot(self, label=r'$d\sigma/d\theta$'):
        self.plot, = self.ax.plot(self.x_ar, self.y_ar, 'o', markersize=5,
                label='nonvertical')
        if self.showvert:
            self.vplot, = self.ax.plot(self.vx_ar, self.vy_ar, 'o',
                    markersize=5, color='tab:orange', label='vertical')
            self.ax.legend()
        self.ax.set_xlim(0, self.x_ar.max()*1.05)
        self.ax.set_ylim(0, self.y_ar.max()*1.1)
        self.ax.set_xlabel('energy transferred (eV)')
        self.ax.set_ylabel('probability')
        self.ax.set_title('electron-induced excitation probs hBN')

    #------------------------------ slider -------------------------------------

    def update(self, val):
        self.Eb = self.slider.val * 1000  # eV
        self.Ebdir = '{:0>5}'.format(int(self.Eb))
        self.slider.valtext.set_text('{:.4g} keV'.format(self.Eb/1000))
        self.readYData()
        self.plot.set_ydata(self.y_ar)
        if self.showvert:
            self.vplot.set_ydata(self.y_ar[self.vidx_ar])

    #------------------------------ button -------------------------------------

    def uselog(self, val):
        self.ax.set_yscale('log')
        self.ax.set_ylim(self.y_ar.min()/10, self.y_ar.max()*10)

    def uselin(self, val):
        self.ax.set_yscale('linear')
        self.ax.set_ylim(0, self.y_ar.max()*1.1)

    #------------------------------ saving -------------------------------------

    def show(self):
        plt.show()

    def clear(self):
        plt.cla()

    def close(self):
        plt.close(self.fig)

    def copy(self):
        return Plot(fig=self.fig, ax=self.ax)

#---------------------------------- TERMINAL------------------------------------

if __name__ == '__main__':
    plot()

#---------------------------------- SCRATCH ------------------------------------

#         self.plot = self.ax.scatter(self.x_ar, self.y_ar)
#         self.plot, = self.ax.plot(self.x_ar, self.y_ar, 'o', markersize=5)
#         self.ax.set_ylim(self.y_ar.min(), self.y_ar.max()*1.1)

#         self.plot.set_xdata(self.x_ar)
#         self.ax.set_ylim(self.y_ar.min(), self.y_ar.max()*1.1)
#         self.plot.set_offsets(self.x_ar, self.y_ar)

#             cb = ymax_idx % nc
#             NKxNKxNV = floor(ymax_idx/nc)
