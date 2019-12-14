# coding: utf8

"""Extension for setting material data or to read in and interpolate it from
appropriate data bases.

Authors : Carlo Barth

"""

# Let users know if they're missing any of our hard dependencies
# (this is section is copied from the pandas __init__.py)
hard_dependencies = ('parse', 'yaml')
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append(dependency)

if missing_dependencies:
    raise ImportError("Missing required dependencies {0}".format(
        missing_dependencies))

from pypmj.internals import _config, ConfigurationError
import numpy as np
import os
from scipy.interpolate import pchip
from scipy.interpolate import UnivariateSpline
from . import refractiveIndexInfo as rii
import logging
logger = logging.getLogger(__name__)

# Load values from configuration
RI_DBASE = _config.get('Data', 'refractiveIndexDatabase')

# Check if the configured refractiveIndexDatabase is valid
_err_msg = 'The configured refractiveIndexDatabase {} does'.format(RI_DBASE) +\
           ' not seem to be an appropriate data base for this module.'
if not os.path.isdir(RI_DBASE):
    raise ConfigurationError(_err_msg)
content = os.listdir(RI_DBASE)
for sub in ['filmetrics', 'glass', 'main', 'organic', 'other', 'library.yml']:
    if sub not in content:
        raise ConfigurationError(_err_msg + ' Missing: {}'.format(sub))


# =============================================================================
class MaterialData(object):
    """
    Class MaterialData
    --------------------------

    Makes different refractive index databases accessible with Python,
    currently RefractiveIndex.info and Filmetrics.

    Usage:
    ------
    Create an instance for the desired material, e.g. silicon, using
        >> refInfo = MaterialData( material = 'silicon' )
    You can easily calculate the refractive index data at desired wavelength
    using
        >> refInfo.getNKdata( yourWavelengths )
    You can get the information provided with each dataset by
        >> refInfo.getAllInfo()
    and plot the known data using
        >> refInfo.plotData() # you can also provide desired wavelengths here

    Further, you can get a list of knownMaterials by
        >> refInfo = MaterialData()
        >> print refInfo.getKnownMaterials()


    Extending the data:
    -------------------
    To add new materials, simply add a value to the "materials" dictionary
    (see below). For the RefractiveIndex.info-database you have to provide
    shelf and book (look inside the database folder) and can then add all
    file names to the 'sets'-list. For filemetrics, you can simply add the
    filename as value for 'fimetricsFile'. You can further provide a
    preferred database.


    Parameters:
    -----------------
    material : str, number or None, default None
        If str, it is the name of the material in the materials-dictionary. If
        number, fixed refractive index of this value.
    unitOfLength : float, default 1.
        multiplier for the wavelengths you will provide (e.g.: 1. = meter,
        1.e-9 = nanometer).
    database {'RefractiveIndex.info', 'filmetrics', 'preferred'},
             default 'preferred'
        The database to use. 'preferred' makes use of the 'preferredDbase'
        key in the materials-dictionary.
    dataset : str
        The dataset name to be loaded
    interpolater {'spline', 'pchip'}, default 'pchip'
        SciPy-interpolater to use.
    splineDegree : int, default: 3
        Only for interpolater 'spline'. Specifies the k-parameter in
        scipy.UnivariateSpline.
    NformulaSampleVals: int, default: 1000
        Number of (logspaced) values for sampling of formula-based data.
    """

    materials = {
        'air': {
            'fixedN': 1.
        },
        'Sapphire': {
            'shelf': 'main',
            'book': 'Al2O3',
            'sets': ['Laytec.yml','Hagemann.yml','Kischkat.yml','Malitson-e.yml',
                     'Malitson-o.yml','Malitson.yml'],
            'fimetricsFile': 'Al2O3.txt',
            'preferredDbase': 'filmetrics'
            },
        'CuBi2O4': {
            'shelf': 'main',
            'book': 'CuBi2O4',
            'sets': ['CuBi2O4.yml'],
            'preferredDbase': 'RefractiveIndex.info'                                   
        },
        'gallium_arsenide': {
            'shelf': 'main',
            'book': 'GaAs',
            'sets': ['Aspnes.yml', 'Skauli.yml'],
            'fimetricsFile': 'GaAs.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'gallium_nitride': {
            'shelf': 'main',
            'book': 'GaN',
            'sets': ['Barker-o.yml'],
            'fimetricsFile': 'GaN.txt',
            'preferredDbase': 'filmetrics'
        },
        'gallium_phosphide': {
            'shelf': 'main',
            'book': 'GaP',
            'sets': ['Aspnes.yml', 'Bond.yml'],
            'fimetricsFile': 'GaP.txt',
            'preferredDbase': 'filmetrics'
        },
        'glass_CorningEagleXG': {
            'shelf': 'glass',
            'book': 'corning',
            'sets': ['EagleXG.yml'],
            'preferredDbase': 'RefractiveIndex.info'
        },
        'gold': {
            'shelf': 'main',
            'book': 'Au',
            'sets': ['Rakic.yml'],
            'fimetricsFile': 'Au.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'InP': {
            'shelf': 'main',
            'book': 'InP',
            'sets': ['Aspnes.yml','Pettit.yml','Hampus.yml'],
            'fimetricsFile': 'InP.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'ITO': {
            'shelf': 'other/mixed crystals',
            'book': 'In2O3-SnO2',
            'sets': ['Konig.yml'],
            'fimetricsFile': 'ITO.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },                
        'PMMA': {
            'shelf': 'organic',
            'book': '(C5O2H8)n - poly(methyl methacrylate)',
            'sets': ['Szczurowski.yml','nk_PMMA_TL_Doguschan.yml'],
            'preferredDbase': 'RefractiveIndex.info'
        },
        'silicon': {
            'shelf': 'main',
            'book': 'Si',
            'sets': ['Vuye-20C.yml', 'Li-293K.yml','Sutter.yml'],
            'fimetricsFile': 'Si.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_nitride': {
            'shelf': 'main',
            'book': 'Si3N4',
            'sets': ['Kischkat.yml', 'Philipp.yml'],
            'fimetricsFile': 'Si3N4.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_oxide': {
            'shelf': 'main',
            'book': 'SiO',
            'sets': ['Hass.yml'],
            'fimetricsFile': 'SiO.txt',
            'preferredDbase': 'filmetrics'
        },
        'silicon_dioxide': {
            'shelf': 'main',
            'book': 'SiO2',
            'sets': ['Malitson.yml','Gao.yml'],
            'fimetricsFile': 'SiO2.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'silver': {
            'shelf': 'main',
            'book': 'Ag',
            'sets': ['Rakic.yml','Johnson.yml','Palik.yml'],
            'fimetricsFile': 'Ag.txt',
            'preferredDbase': 'RefractiveIndex.info'
        },
        'sol_gel': {
            'fixedN': 1.42
        },
        'water': {
            'fixedN': 1.33
        }
    }

    databases = {'RefractiveIndex.info': {'wavelengthUnit': 1.e6},
                 'filmetrics': {'wavelengthUnit': 1.e9}}
    dtype = [('wvl', np.float64), ('n', np.float64), ('k', np.float64)]

    def __init__(self, material=None, unitOfLength=1.,
                 database='preferred', dataset =None,interpolater='pchip',
                 splineDegree=3, NformulaSampleVals=1000):

        if material is None:
            return

        # If a fixed refractive index is given as material
        if isinstance(material, (int, float, complex)):
            self.fixedN = True
            self.totalWvlRange = (-np.inf, np.inf)
            self.n = material
            self.name = 'CustomMaterial'
            return
        else:
            self.fixedN = False
            self.name = material

        # If the material dictionary has the key fixedN
        if material in self.materials:
            if 'fixedN' in self.materials[material]:
                self.fixedN = True
                self.n = self.materials[material]['fixedN']
                return

        self.NformulaSampleVals = NformulaSampleVals
        self.db = database
        if material is not None:
            if database == 'preferred':
                self.db = self.materials[material]['preferredDbase']

        self.knownInterpolaters = ['pchip', 'spline']
        self.interpolater = interpolater
        self.splineDegree = splineDegree
        assert interpolater in self.knownInterpolaters,\
            'The interpolater {0} is unknown. Known interpolaters are: {1}.'.\
            format(interpolater, self.knownInterpolaters)
        assert self.db in self.getKnownDatabases(),\
            'The database {0} is unknown. Known databases are: {1}.'.format(
            self.db, self.getKnownDatabases())

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = None
        self.unitOfLength = unitOfLength
        self.interpolaterSet = False
        self.material = material
        self.getData4Material()

    def getKnownDatabases(self):
        return list(self.databases.keys())

    def getKnownMaterials(self):
        return list(self.materials.keys())

    def convertWvl(self, wavelengths):
        """converts to database specific wavelength unit."""
        if self.fixedN:
            return wavelengths
        return wavelengths * self.databases[self.db]['wavelengthUnit'] * \
            self.unitOfLength

    def getYML(self, shelf, book, fname):
        """composes a valid YML-filename for RefractiveIndex.info database."""
        return os.path.join(RI_DBASE,
                            shelf, book, fname)

    def getYMLinfo(self, shelf, book, fname):
        """returns the references and comments provided in the YML files."""
        return rii.getInfo(self.getYML(shelf, book, fname))

    def getFilmetricsData(self, filename, returnNameOnly=False):
        """produces a numpy-structured array from the data provided in the
        Filemetrics-file."""
        self.fullName = os.path.join(RI_DBASE, 'filmetrics', filename)
        data = np.loadtxt(self.fullName)
        structuredData = np.zeros((len(data[:, 0])), self.dtype)
        cols = [c[0] for c in self.dtype]
        for dim in range(data.shape[1]):
            structuredData[cols[dim]] = data[:, dim]
        return structuredData

    def getData4Material(self):
        """produces a numpy-structured array from all valid datafiles for the
        given material."""

        if self.material not in list(self.materials.keys()):
            raise Exception('Material is specified. Known materials are:' +
                            '\n\t{0}\n'.format(self.getKnownMaterials()) +
                            'Please specify shelf, book and ymlFile instead.')

        mat = self.materials[self.material]
        if self.db == 'filmetrics':
            if 'fimetricsFile' not in list(mat.keys()):
                raise Exception('No filmetrics data available for {0}'.format(
                                self.material))
                return
            self.setInterpolater(self.getFilmetricsData(mat['fimetricsFile']))
            return

        # This code is only executed if database == "RefractiveIndex.info"
        shelf = mat['shelf']
        book = mat['book']
        sets = mat['sets']
        if self.dataset is not None:
            index = None
            for iS,s in enumerate(sets):
                if s == self.dataset:
                    index = iS
                    break
            if index is not None:
                sets = [sets[index]]   
        # Determine the sorting indices of the sets, to use them in ascending
        # wavelength order
        setMaxVals = []
        for s in sets:
            yamlf = self.getYML(shelf, book, s)
            setMaxVals.append(np.max(rii.getRange(yamlf)))
        sortIdx = sorted(list(range(len(setMaxVals))),
                         key=lambda k: setMaxVals[k])

        # Loop over sorted sets
        for i in sortIdx:
            s = sets[i]
            yamlf = self.getYML(shelf, book, s)
            thisRange = rii.getRange(yamlf)
            returnedData = rii.getData(yamlf, thisRange[0],
                                       returnExistingDataOnly=True)

            if hasattr(returnedData, '__call__'):  # data in form of formula
                wvl = np.logspace(np.log10(thisRange[0]),
                                  np.log10(thisRange[1]),
                                  self.NformulaSampleVals)
                if i == sortIdx[0]:
                    data = np.zeros((len(wvl)), self.dtype)
                    data['wvl'] = wvl
                    data['n'] = returnedData(wvl)
                else:
                    newData = np.zeros((len(wvl)), self.dtype)
                    newData['wvl'] = wvl
                    newData['n'] = returnedData(wvl)

                    # Avoid overlap
                    idx = np.where(newData['wvl'] > np.max(data['wvl']))
                    data = np.append(data, newData[idx])

            elif len(returnedData) == 3:  # data in form of tabulated values

                wvl, dataN, dataK = returnedData
                if i == sortIdx[0]:
                    data = np.zeros((len(wvl)), self.dtype)
                    data['wvl'] = wvl
                    data['n'] = dataN
                    data['k'] = dataK
                else:
                    newData = np.zeros((len(wvl)), self.dtype)
                    newData['wvl'] = wvl
                    newData['n'] = dataN
                    newData['k'] = dataK

                    # Avoid overlap
                    idx = np.where(newData['wvl'] > np.max(data['wvl']))
                    data = np.append(data, newData[idx])

        # All data is loaded, now the interpolaters can be set
        self.setInterpolater(data)

    def setInterpolater(self, data):
        """Initialzes the desired interpolater."""
        order = np.argsort(data, order='wvl')
        self.knownData = data[order]
        self.totalWvlRange = (np.min(data['wvl']), np.max(data['wvl']))
        self.limits = [self.knownData['n'][0] + 1.j * self.knownData['k'][0],
                       self.knownData['n'][-1] + 1.j * self.knownData['k'][-1]]

        # spline
        if self.interpolater == 'spline':
            self.nInterp = UnivariateSpline(data['wvl'], data['n'], s=0,
                                            k=self.splineDegree)
            self.kInterp = UnivariateSpline(data['wvl'], data['k'], s=0,
                                            k=self.splineDegree)
        # pchip
        elif self.interpolater == 'pchip':
            self.nInterp = pchip(data['wvl'], data['n'])
            self.kInterp = pchip(data['wvl'], data['k'])
        self.interpolaterSet = True

    def getAllInfo(self, log=True, return_output=False):
        """prints out all available accompanying data (wavelength range,
        references, comments)"""

        if self.fixedN:
            logger.info('*** Using fixed refractive index of {0} ***'.format(
                self.n))
            return

        if self.material not in list(self.materials.keys()):
            raise Exception('Material is specified. Known materials are:' +
                            '\n\t{0}\n'.format(self.getKnownMaterials()) +
                            'Please specify shelf, book and ymlFile instead.')
        mat = self.materials[self.material]
        op_lines = []
        if self.db == 'filmetrics':
            op_lines.append(
                u'\n***\nInfo for file: {0}\nwith wavelength range: {1}nm:'.
                format(self.fullName, self.totalWvlRange))
            op_lines.append(u'No reference data available for filmetrics' +
                            u' files yet.')
            op_lines.append(u'Please see' +
                            ur'http://www.filmetrics.de/'+
                            u'refractive-index-database')
            op_lines.append(u'***')
            return

        shelf = mat['shelf']
        book = mat['book']
        sets = mat['sets']
        for s in sets:
            yamlf = self.getYML(shelf, book, s)
            thisRange = rii.getRange(yamlf)
            info = self.getYMLinfo(shelf, book, s)
            op_lines.append(
                u'\n***\nInfo for file: {0}\nwith wavelength range: {1}µm:'.
                format(yamlf, thisRange))
            if 'REFERENCES' in info:
                op_lines.append(u'\nReferences:\n----------\n')
                op_lines.append(info['REFERENCES'])
            if 'COMMENTS' in info:
                op_lines.append(u'\nComments:\n----------\n')
                op_lines.append(info['COMMENTS'])
            op_lines.append(u'***')
        output = u'\n'.join(op_lines)
        if log:
            logger.info(output)
        if return_output:
            return output

    def checkIfSet(self):
        if self.fixedN:
            return
        if not self.interpolaterSet:
            raise Exception('No valid interpolater set.')
            return

    def getNKdata(self, wavelengths, absorption=True, convert=True,
                  extrapolation=True, suppressWarnings=True):
        self.checkIfSet()

        if self.fixedN:
            return self.n

        if convert:
            wavelengths = self.convertWvl(wavelengths)

        extrapolationNeeded = False
        informedAboutExtrapolation = False
        if not (self.totalWvlRange[0] <= np.min(wavelengths) and
                self.totalWvlRange[1] >= np.max(wavelengths)):
            extrapolationNeeded = True
            if extrapolation:
                if not informedAboutExtrapolation:
                    if not suppressWarnings:
                        logger.info('Found wavelengths for which no data ' +
                                    'is known. Using extrapolation.')
                    informedAboutExtrapolation = True
            else:
                raise Exception('Given wavelength is outside known ' +
                                'wavelength data range. Known Range: {0}µm'.
                                format(self.totalWvlRange) +
                                ' Your range is ({0}, {1})µm'.format(
                                    np.min(wavelengths), np.max(wavelengths)))

        if absorption:
            res = self.nInterp(wavelengths) + 1.j * self.kInterp(wavelengths)
            if extrapolation and extrapolationNeeded:
                return self.extrapolate(wavelengths, res)
            else:
                return res
        else:
            res = self.nInterp(wavelengths)
            if extrapolation and extrapolationNeeded:
                return np.real(self.extrapolate(wavelengths, res))
            else:
                return res

    def getPermittivity(self, wavelengths, absorption=True, convert=True,
                        extrapolation=True):
        return np.square(self.getNKdata(wavelengths,
                                        absorption,
                                        convert,
                                        extrapolation))

    def extrapolate(self, wavelengths, data2extrapolate):
        wvls = np.array(self.totalWvlRange)
        if isinstance(wavelengths, (int, float, complex)):
            if wavelengths < wvls[0]:
                return self.limits[0]
            else:
                return self.limits[1]
        else:
            minIdx = np.where(wavelengths < wvls[0])
            maxIdx = np.where(wavelengths > wvls[1])
            extrapolatedResult = data2extrapolate
            extrapolatedResult[minIdx] = self.limits[0]
            extrapolatedResult[maxIdx] = self.limits[1]
            return extrapolatedResult

    def getMinMaxPermittivityInWavelengthRange(self, wvlRange,
                                               convert=True,
                                               Nsamples=1e3):
        if not isinstance(wvlRange, np.ndarray):
            wvlRange = np.array(wvlRange)
        if convert:
            wvlRange = self.convertWvl(wvlRange)
        wvls = self.totalWvlRange
        rangeOfInterest = np.array([max([wvls[0], wvlRange[0]]),
                                    min([wvls[1], wvlRange[1]])])
        wavelengths = np.linspace(rangeOfInterest[0], rangeOfInterest[1],
                                  Nsamples)
        perms = self.getPermittivity(wavelengths,
                                     absorption=False,
                                     convert=False)
        return np.array([np.min(perms), np.max(perms)])

    def plotData(self, wavelengths=None, wvlRange=None, Nvals=1000,
                 convert=True, plotKnownValues=False, show=True):
        self.checkIfSet()
        import matplotlib.pyplot as plt

        if self.fixedN:
            self.getAllInfo()
            logger.debug('-> Nothing to plot.')
            return
        if wavelengths is not None:
            wvls = wavelengths
        elif wvlRange:
            wvls = np.linspace(wvlRange[0], wvlRange[1], Nvals)
        else:
            wvls = np.linspace(self.totalWvlRange[0], self.totalWvlRange[1],
                               Nvals)
        if convert and (wavelengths is not None or wvlRange is not None):
            wvls = self.convertWvl(wvls)
        interp = self.getNKdata(wvls, convert=False)

        windowIdx = np.where(
            np.logical_and(
                np.min(wvls) <= self.knownData['wvl'],
                self.knownData['wvl'] <= np.max(wvls)))
        knownDataWindow = self.knownData[windowIdx]

        plt.subplot(2, 1, 1)
        plt.plot(wvls, interp.real, '-', color='b',
                 label='$n$ interpolated', lw=2)
        if plotKnownValues:
            plt.plot(knownDataWindow['wvl'], knownDataWindow['n'], 'o',
                     color='b', label='$n$')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('n')
        plt.legend(frameon=False, loc='best')

        plt.subplot(2, 1, 2)
        plt.plot(wvls, interp.imag, '-', color='g',
                 label='$k$ interpolated', lw=2)
        plt.autoscale(enable=True, axis='x', tight=True)
        if plotKnownValues:
            plt.plot(knownDataWindow['wvl'], knownDataWindow['k'], 'o',
                     color='g', label='$k$')
        plt.xlabel(u'wavelength in µm')
        plt.ylabel('k')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.legend(frameon=False, loc='best')
        plt.suptitle('Material: {0}, Database: {1}'.format(
            self.material, self.db))
        if show:
            plt.show()


if __name__ == '__main__':

    d = MaterialData('silicon')
    d.plotData()

    # Wavelength parameters
    wavelength_start = 400.e-9
    wavelength_end = 1500.e-9
    wavelength_stepSize = 1.e-9
    wvl = np.arange(wavelength_start, wavelength_end + wavelength_stepSize,
                    wavelength_stepSize)

#     data1 = MaterialData(database = 'RefractiveIndex.info',
#                                material = 'gallium_arsenide')
#     data1.getAllInfo()
#
#     data2 = MaterialData(database = 'filmetrics',
#                                 material = 'gallium_arsenide')
#     data2.getAllInfo()
#
#     plt.figure(1)
#     data1.plotData(wavelengths = wvl, convert=True, show=False)
# #     data1.plotData()
#     plt.figure(2)
#     data2.plotData(wavelengths = wvl, convert=True, show=False)
#     plt.show()
#     quit()

    mats = MaterialData().getKnownMaterials()
    mats = ['silver']
    for m in mats:
        try:
            logger.debug(m)
            refInfo = MaterialData(material=m)
            refInfo.plotData(wavelengths=wvl, convert=True)
        except:
            pass
