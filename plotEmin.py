from numpy import array
import matplotlib.pyplot as plt

# include date on outfile file names
from datetime import date
from shutil import copyfile

# displacement cross section classes
from statDispCross import DispCross

class EminPlot(object):
    """Class to plot Emin vs Eb"""
    def __init__(self, fig=None, ax=None, figsize=(6, 5)):
        """Initializes figure and axes."""
        if fig==None or ax==None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig, ax

    def plot(self,
            mat = 'MoS2',
            ebounds = [0, 1],

            linestyle = '-',
            color = 'tab:blue',
            label = 'MoS$_2$'
            ):
        """Plots cross sections as functions of TEM energy.

        sigma_dict: dictionary containing plottable data (dict)
        """
        dc = DispCross(mat=mat, ebounds=ebounds)
        Eb_ar = dc.Eb_ar / 1000  # keV
        Emin_ar = dc.Emin_ar * 1000   # meV
        self.ax.plot(Eb_ar, Emin_ar, linewidth=2, zorder=3, color=color,
                    linestyle=linestyle, label=label)

        self.Ebmin = Eb_ar.min()
        self.Ebmax = Eb_ar.max()

    def decorate(self,
            xbounds = 'auto',
            ybounds = None,
            xlabel = 'beam energy (keV)',
            ylabel = 'minimum transfer (meV)',
            legend = True,
            grid = False,
            ):
        """
        adds plot attributes
        xbounds: TEM energy range in keV (list of two floats)
        ybounds: cross section range in barn (list of two floats)
        """
        # boundaries
        if xbounds=='auto':
            xbounds=(self.Ebmin, self.Ebmax)
        self.ax.set_xlim(xbounds)
        self.ax.set_ylim(ybounds)
        self.ax.set_xlabel(xlabel, fontsize=18)
        self.ax.set_ylabel(ylabel, fontsize=18)
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        if grid:
            self.ax.grid()
        if legend:
            self.ax.legend(fontsize=14)
        plt.tight_layout()

    def save(self,
            outfile = 'emin.pdf',
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

def plot(ebounds=[0, .5], outfile='emin.pdf', save=True):
    """Plots Emin for MoS2 and Barm."""
    plot = EminPlot()
    plot.plot('MoS2', ebounds=ebounds, label='MoS$_2$', color='tab:green')
    plot.plot('Barm', ebounds=ebounds, label='hBN', color='tab:blue')
    plot.decorate()
    plot.save()
    plot.show()
