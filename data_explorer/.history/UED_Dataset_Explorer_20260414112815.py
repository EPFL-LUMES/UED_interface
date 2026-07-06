import time
t_start = time.time()
import warnings
import pyqtgraph.opengl as gl
from io import StringIO
import random
from ctypes import windll
from scipy import ndimage as nd
from numba import vectorize, float64# , char, guvectorize, float32
from numba import njit, prange, jit
import ftplib
import os, psutil
# import atexit
import glob
# from inspect import CO_VARKEYWORDS
import numbers


import sys
import zipfile

from functools import partial
from re import T
import io
import win32clipboard
import pyqtgraph.exporters
from PIL import Image

from datetime import datetime as dtm
import imageio.v2 as iio
import math
import h5py
import hdf5plugin
import natsort
import numpy as np
import numpy_indexed as npi
import pandas as pd
import pandasgui as pdgui

import PyQt5
import pyqtgraph as pg

import tifffile
import wmi
from lmfit.models import (ConstantModel, GaussianModel, LinearModel, LorentzianModel, QuadraticModel, VoigtModel)
from matplotlib import cm
from PyQt5 import QtWidgets, uic
from PyQt5.Qt import QStandardItem, QStandardItemModel
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtGui
from scipy import special
from scipy.optimize import curve_fit
from scipy.special import wofz
from numba_progress import ProgressBar
import copy
import matplotlib
import pyqtgraph.console
from PySide2.QtWidgets import QMessageBox as qmsg



matplotlib.use('Qt5Agg')

warnings.filterwarnings("ignore")

windll.shcore.SetProcessDpiAwareness(1)

pg.setConfigOptions(antialias=True)

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


new_line = '\n'
jet = cm.get_cmap('jet', 12)
colors_list = []
for i in range(jet.N):
    colors_list.append(tuple([z * 255 for z in jet(i)[:-1]]))
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 12), color=colors_list)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, numbers.Number):
                return format(value, '.5g')
            else:
                return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])


class StandardItem(QStandardItem):

    def __init__(self, txt='', font_size=12, set_bold=False, color=QColor(0, 0, 0)):
        super().__init__()

        fnt = QFont('Open Sans', font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)


def find_drive_by_label(label="TOSHIBA EXT"):

    looking_for = label
    drive_names = []
    drive_letters = []
    letter = ""
    c = wmi.WMI()
    for drive in c.Win32_LogicalDisk():
        drive_names.append(str(drive.VolumeName).strip())
        drive_letters.append(str(drive.Caption).strip())

    if looking_for.strip() not in drive_names:
        print("The drive is not connected currently.")
    else:
        letter = str(drive_letters[drive_names.index(looking_for)])

    return letter


def path_cut(path_pieces):
    pieces2 = path_pieces.copy()
    a = 0
    for piece in path_pieces:
        try:
            a = float(piece)
        except Exception:
            pieces2.pop(0)
        if a > 2000:
            break
    return pieces2


def send_to_clipboard(clip_type, data):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()


def sort_by(reference, targets):
    results = []
    a = np.array(reference).argsort()
    for t in targets:
        t = [t[i] for i in a]
        results.append(t)
    return results


def moments_gauss(data, x):
    offset = np.min(data)
    height = np.max(data) - offset
    HM = offset + height / 2
    over = np.where(data > HM)[0]
    fwhm = x[over[-1]] - x[over[0]]
    center = x[np.argmax(data)]
    sigma = fwhm / 2.
    gamma = sigma
    amplitude = height * sigma / 0.3989423
    return amplitude, center, sigma, gamma, offset


def fit_gauss(data, x):
    model = GaussianModel() + QuadraticModel()
    initial_guess = moments_gauss(data, x)
    params = model.make_params(amplitude=initial_guess[0], center=initial_guess[1], sigma=initial_guess[2], gamma=initial_guess[3], c=initial_guess[4], b=0, a=0)
    y = data
    result = model.fit(y, params, x=x, max_nfev=400)
    return result


def moments_voigt(data, x):
    offset = np.min(data)
    height = np.max(data)-offset
    HM = offset + height/2
    over = np.where(data > HM)[0]
    fwhm = x[over[-1]] - x[over[0]]
    center = x[np.argmax(data)]
    sigma = fwhm/4.
    gamma = sigma
    amplitude = height*sigma * \
        np.sqrt(2*np.pi)/np.real(wofz(1J*gamma/(sigma*np.sqrt(2))))
    return amplitude, center, sigma, gamma, offset


def fit_voigt(data, x):
    model = VoigtModel() + LinearModel()
    # x = np.arange(data.shape[0])
    initial_guess = moments_voigt(data, x)
    params = model.make_params(amplitude=initial_guess[0], center=initial_guess[1], sigma=initial_guess[2], gamma=initial_guess[3], intercept=initial_guess[4], slope=0)
    params['gamma'].vary = True
    params['gamma'].expr = ''
    result = model.fit(data, params, x=x, max_nfev=400)
    return result


def rotation_matrix(theta, x, y, z):  # defines the 3D rotation matrix from a quaternion composed by theta and the components of the rotation axis
    q0 = np.cos(theta / 2)
    q1 = x * np.sin(theta / 2)
    q2 = y * np.sin(theta / 2)
    q3 = z * np.sin(theta / 2)
    rot = np.array([
        [
            q0**2 + q1**2 - q2**2 - q3**2,
            2 * q1 * q2 - 2 * q0 * q3,
            2 * q1 * q3 + 2 * q0 * q2,
        ],
        [
            2 * q1 * q2 + 2 * q0 * q3,
            q0**2 - q1**2 + q2**2 - q3**2,
            2 * q2 * q3 - 2 * q0 * q1,
        ],
        [
            2 * q1 * q3 - 2 * q0 * q2,
            2 * q2 * q3 + 2 * q0 * q1,
            q0**2 - q1**2 - q2**2 + q3**2,
        ],
    ])
    return rot


def rotate_list_vectors(list_vec, rotation_matrix):
    result = []
    for vec in list_vec:
        result.append(np.dot(rotation_matrix, vec))
    return result


def calc_pattern_FCC(
        base1=np.array([1, 1, 1]), base2=np.array([0, 2, -2]), r=8):
    xf = []
    yf = []
    peak_names = []
    factors = []
    for h in range(-r + 1, r):
        for k in range(-r + 1, r):
            for l in range(-r + 1, r):  # noqa: E741
                if ((h % 2 == 0) and (k % 2 == 0) and
                        (l % 2 == 0)) or ((h % 2 == 1) and (k % 2 == 1) and (l % 2 == 1)):
                    if np.dot(np.array([h, k, l]), np.array([1, -1, 0])) == 0:
                        factors.append(np.array([h, k, l]))

    base1_n = base1 / np.linalg.norm(base1)
    base2_n = base2 / np.linalg.norm(base2)

    for arr in factors:
        xf.append(np.dot(arr, base2_n) / np.linalg.norm(base2))
        yf.append(np.dot(arr, base1_n) / np.linalg.norm(base1))
        peak_names.append(str(arr))
    xf = np.array(xf)
    yf = np.array(yf)
    return xf, yf, peak_names


def calc_pattern_HCP(base1=np.array([0, 0, 2]), base2=np.array([2, 0, 0]), r=6):
    xf_h = []
    yf_h = []
    peak_names_h = []
    factors_h = []
    for h in range(-r + 1, r):
        for k in range(-r + 1, r):
            for l in range(-r + 1, r): # noqa: E741
                if not (l % 2 != 0 and (h + 2 * k) % 3 == 0):
                    if np.dot(np.array([h, k, l]), np.array([0, 1, 0])) == 0:
                        factors_h.append(np.array([h, k, l]))
    base1_n = base1 / np.linalg.norm(base1)
    base2_n = base2 / np.linalg.norm(base2)

    for arr in factors_h:
        xf_h.append(np.dot(arr, base2_n) / np.linalg.norm(base2))
        yf_h.append(np.dot(arr, base1_n) / np.linalg.norm(base1))
        peak_names_h.append(str(arr))
    xf_h = np.array(xf_h)
    yf_h = np.array(yf_h)
    return xf_h, yf_h, peak_names_h


dummy_arr = np.random.rand(1, 514, 514)
df = pd.DataFrame()
df['dummy'] = np.arange(dummy_arr.shape[0])
filepath = os.path.realpath(__file__)
dirpath = os.path.dirname(filepath)
os.chdir(dirpath)
sys.path.append(dirpath)
img_arr = dummy_arr.copy()
dummy_arr = 0
img_arr = np.zeros((512, 512))
img_arr_2 = np.zeros((512,512))
if img_arr.ndim == 2:
    img_arr = np.array([img_arr])
pg.setConfigOptions(imageAxisOrder='row-major')
app = QtWidgets.QApplication(sys.argv)
app.setStyle('fusion')
w = uic.loadUi("./main_window_hor.ui", QtWidgets.QMainWindow())
# apply_stylesheet(app, theme='dark_blue.xml')

palette = QPalette()
palette.setColor(QPalette.Window, QColor(53, 53, 53))
palette.setColor(QPalette.WindowText, Qt.white)
palette.setColor(QPalette.Base, QColor(25, 25, 25))
palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
palette.setColor(QPalette.ToolTipBase, Qt.black)
palette.setColor(QPalette.ToolTipText, Qt.white)
palette.setColor(QPalette.Text, Qt.white)
palette.setColor(QPalette.Button, QColor(53, 53, 53))
palette.setColor(QPalette.ButtonText, Qt.white)
palette.setColor(QPalette.BrightText, Qt.red)
palette.setColor(QPalette.Link, QColor(42, 130, 218))
palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
palette.setColor(QPalette.HighlightedText, Qt.black)
app.setPalette(palette)





def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

def dropEvent(self, event):
    global df, img_arr, loading_function
    files = [u.toLocalFile() for u in event.mimeData().urls()]
    print('filename =',files)
    if files[0].endswith('.h5'):
        # print('is serie?', check_if_serie(files))
        if check_if_serie(files):
            load_H5_serie(files)
        else:
            loading_function(zipped = False, dropped_path = ' ', file_list=files)
        scroll_data(0)
        plot_metadata()

    elif files[0].endswith('.zip'):
        loading_function(zipped = True, dropped_path = files)
        scroll_data(0)
        plot_metadata()
    else:
        loading_function(zipped = False, dropped_path = files[0])
        scroll_data(0)
        plot_metadata()
        
def mousePressEvent(self, event):
    qindex = self.indexAt(event.pos())
    print(qindex.row(), qindex.column())

    
setattr(QtWidgets.QMainWindow, 'dragEnterEvent', dragEnterEvent) 
setattr(QtWidgets.QMainWindow, 'dropEvent', dropEvent)
# setattr(QtWidgets.QTableView, 'mousePressEvent', mousePressEvent)

def copy_table_entry(event):
    global fit_dict
    columns = list(fit_dict.keys())
    value = fit_dict[columns[event.row()]]
    #copy value to clipboard
    clipboard = QtWidgets.QApplication.clipboard()
    clipboard.setText(str(value))

    
w.tableView.doubleClicked.connect(copy_table_entry)


w_pat = uic.loadUi("./patterns.ui", QtWidgets.QMainWindow())
w_prog = uic.loadUi("./progress.ui", QtWidgets.QDialog())
w_plot = uic.loadUi("./plotter.ui")
w_timeres = uic.loadUi("./time_resolution_v2.ui", QtWidgets.QDialog())
w_curve_name_dialog = uic.loadUi("./curve_name.ui", QtWidgets.QDialog())
w_radial_average = uic.loadUi("./radial_average.ui", QtWidgets.QMainWindow())
w_slice_data = uic.loadUi("./slice_data.ui", QtWidgets.QDialog())
w_simulator = uic.loadUi("./Simulator.ui", QtWidgets.QMainWindow())
w_calibration = uic.loadUi("./calibration_dialog.ui", QtWidgets.QDialog())
w_3dView = uic.loadUi("./3d_view_dialog.ui", QtWidgets.QDialog())
def show_error(text = 'Error!'):
    qmsg.critical(None,'Error',text,buttons=qmsg.Discard)


def metadata_panel_show():
    if w.tabWidget.isHidden():

        w.tabWidget.show()
        w.gridLayout.setColumnStretch(0,2)
        w.gridLayout.setColumnStretch(1,1)
        w.actionMetadata_Panel.setChecked(True)
    else:
        w.tabWidget.hide()
        w.gridLayout.setColumnStretch(1,0)
        w.gridLayout.setColumnStretch(0,1)
        w.actionMetadata_Panel.setChecked(False)

w.actionMetadata_Panel.triggered.connect(metadata_panel_show)

w.show()


starting_path = r'D:\UED_measurements'
filename_opened = ''

def open_console():
    namespace = globals()
    console = pyqtgraph.console.ConsoleWidget(namespace=namespace, text="Welcome to UED Dataset Explorer Console\nAll modules and variables loaded in the main window are available here, otherwise you can load whatever module you need, provided that it is installed in your current python kernel\n")
    console.show()


w.actionConsole.triggered.connect(open_console)

class SPE_File(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.int16)[0])
        self._ydim = np.int64(self.read_at(656, 1, np.int16)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, np.int8)
        rawtime = self.read_at(172, 6, np.int8)
        strdate = ''
        for ch in rawdate:
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate, "%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim, np.int32)
        return img.reshape((self._ydim, self._xdim))

    def load_accum(self):
        self._acc = np.int64(self.read_at(668, 4, np.uint32)[0])
        return(self._acc)

    def load_gates(self):
        self._gates = np.int64(self.read_at(112, 2, np.uint16)[0])
        return(self._gates)

    def load_ROI(self):
        self._ROI = np.array([np.int64(self.read_at(1512, 2, np.uint16)[0]), np.int64(self.read_at(1514, 2, np.uint16)[0]), np.int64(self.read_at(1516, 2, np.uint16)[0]),
                            np.int64(self.read_at(1518, 2, np.uint16)[0]), np.int64(self.read_at(1520, 2, np.uint16)[0]), np.int64(self.read_at(1522, 2, np.uint16)[0])])
        return(self._ROI)

    def close(self):
        self._fid.close()

def loading_function(zipped = True, dropped_path = None, file_list = None):
    global starting_path, filename_opened, df, img_arr, imgON, imgOFF
    df = 0
    path = starting_path
    chosen = False
    self_df = 0

    if zipped is False:
        if dropped_path is None:
            path = QtWidgets.QFileDialog.getExistingDirectory(w, "Select Folder", path)
        else:
            path = dropped_path

        w.setWindowTitle(f"UED Dataset Explorer - {path}")
        if path != '' and file_list is None:
            chosen = True
            list_tiff = natsort.natsorted(glob.glob(path + '/' + '*.tiff'))
            list_h5 = natsort.natsorted(glob.glob(path + '/' + '*.h5'))
            list_spe = natsort.natsorted(glob.glob(path + '/' + '*.SPE'))
            if len(list_tiff) > 0:
                file_list = list_tiff
            elif len(list_h5) > 0:
                file_list = list_h5
            else:
                file_list = list_spe
            starting_path = path
            filename_opened = os.path.normpath(path).split(os.sep)[-1]
        else:
            filename_opened = os.path.normpath(file_list[0]).split(os.sep)[-1]
            chosen = True
    elif zipped is True:
        if dropped_path is None:
            fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select Archive", path, "Zip Files (*.zip)")
        else:
            fname = dropped_path
        if fname[0] != '':
            chosen = True
            archive = zipfile.ZipFile(fname[0], 'r')
            file_list = archive.namelist()[1:]
            file_list = natsort.natsorted(file_list)
        starting_path = os.path.split(fname[0])[0]
        filename_opened, _ = os.path.splitext(os.path.split(fname[0])[1])
    is_Tiff = False
    img_list = []
    img_list_2 = []
    file_name = []
    i = 0
    img_arr = 0
    img_arr_2 = 0
    if chosen:
        with pg.ProgressDialog("Loading images...", 0, len(file_list)) as dlg:
            dlg.setWindowTitle("Loading images...")
            dlg.setFixedSize(400, 100)
            comp = 0
            for file in file_list[:]:
                if "master" in file:
                    break
                #print(file.split('/')[-1])
                file_name.append(file.split('/')[-1])
                if dlg.wasCanceled():
                    break
                if file.endswith('.tiff'):
                    is_Tiff = True
                    if zipped:
                        file = archive.open(file)
                    try:
                        with tifffile.TiffFile(file) as tif:
                            img_list.append(np.array(tif.asarray(), dtype=np.int32))
                            if i == 0:
                                self_df = pd.DataFrame(tif.imagej_metadata, index=[0])
                            else:
                                df1 = pd.DataFrame(tif.imagej_metadata, index=[i])
                                self_df = pd.concat([self_df, df1], ignore_index=True)
                    except Exception:
                        print(f"Error opening TIFF {file}")
                    i = i + 1
                elif file.endswith('.h5'):
                    if zipped:
                        file = archive.open(file)
                    try:
                        with h5py.File(file, 'r') as h5:
                            dset_name = list(h5.keys())[0]
                            dset = h5[dset_name]
                            acquisition_mode = 'INTS'
                            try:
                                acquisition_mode = dset.attrs['Acquisition_mode']
                            except Exception:
                                acquisition_mode = 'INTS'
                            # attributes = list(dset.attrs.keys())
                            precision = np.int32
                            w.label_7.setText('Pump ON')
                            
                            if acquisition_mode == 'EXTG':
                                precision = np.int32                     
                            img_list.append(np.array(dset[:], dtype=precision))
                            if i == 0:
                                self_df = pd.DataFrame([dict(dset.attrs.items())], index=[0])
                            else:
                                df1 = pd.DataFrame([dict(dset.attrs.items())], index=[i])
                                self_df = pd.concat([self_df, df1])
                            #check if a column with the name 'Acquisition_mode' exists
                            if 'Acquisition_mode' in self_df.columns:
                                if acquisition_mode == 'EXTG' and w.checkBox_pump.isChecked():
                                    ## withdraw compromised data
                                    try:
                                        data_pump = np.array(h5['data_diode'][:], dtype=np.int16)
                                        if (data_pump[1] > data_pump[0] and np.sum(data_pump[::2])==0) or (data_pump[1] < data_pump[0]  and np.sum(data_pump[1::2])==0):
                                            img_list_2.append(np.array(h5[dset_name+'2'][:], dtype=precision))
                                        else:
                                            comp = comp+1
                                            print(comp, 'th compromised data', data_pump)
                                            del img_list[-1]
                                            self_df.drop(index=self_df.index[-1],axis=0,inplace=True)
                                            i = i-1
                                    except:
                                        img_list_2.append(np.array(h5[dset_name+'2'][:], dtype=precision))
                                    
                    except Exception:
                        print(Exception)
                        print(f"Error opening H5 {file}")
                    i = i + 1
                    # print('number of images: ', i)
                
                elif file.endswith('.SPE'):
                    try:
                        raw_image = iio.imread(file)
                        img_list.append(np.array(raw_image, dtype=np.int32))
                        if i == 0:
                            self_df = pd.DataFrame([dict(raw_image.meta)], index=[0])
                        else:
                            df1 = pd.DataFrame([dict(raw_image.meta)], index=[i])
                            # selkf_df = pd.concat([self_df, df1])
                    except Exception:
                        print(f"Error opening SPE {file}")
                    i = i + 1
                    
                dlg += 1
            print('Number of compromised images = ',comp)
            if not img_list_2:
                img_arr = check_image_range(img_list)
            else:
                #convert list to array without increasing memory usage=
                print('building img_arr')
                imgON = np.stack(img_list)
                img_list = []
                imgOFF = np.stack(img_list_2)
                img_list_2 = []
                img_arr = imgON
                print('building done')
            if is_Tiff:
                try:
                    self_df = self_df.drop(
                        columns=['ImageJ', 'images', 'hyperstack', 'mode'])
                    self_df = self_df.rename(
                        columns={
                            'temperature_sample': 'Temperature_A',
                            'temperature_finger': 'Temperature_B',
                            'pressure': 'Pressure',
                            'time': 'Time'
                        })
                    if 'timestamp' in self_df.columns:
                        self_df = self_df.rename(columns={'timestamp': 'Time'})
                except Exception:
                    show_error("No metadata found")
            df = self_df
            self_df = 0
            print('length of arguement: ',len(df['LTS_position']))
            print('lenght of df : ', len(df))
            print(df['LTS_position'])
            # print('lenght of imgON:', len(imgON))
            # print('lenght of imgOFF:', len(imgOFF))
            
            df.rename(columns={"delay": "LTS_position"}, inplace=True)
            df['Delay_ps'] = df['LTS_position']*6.66666 

            if 'Time_for_humans' not in df.columns:
                df['Time_for_humans'] = str(dtm.fromtimestamp(df['Time']))
            print('length df after loading ',len(df))

    w.widget.setLevels(0, 10*np.mean(img_arr[0]))
    w.widget.setHistogramRange(0, 10*np.mean(img_arr[0]))
    # check_shutter()
    print('length df after check ',len(df))
    # try:
    #     print('stocazzo',img_arr.shape)
    #     df1 = counting_loops(df)
    #     df = df1
    #     print('length df after the try ',len(df))
    # except Exception:
    #     pass

def counting_loops(df):
    # print(df['LTS_position'])
    #check if df has a column called 'LTS_position'
    if 'LTS_position' in df.columns:
        #count how many times LTS_position[0] is present
        count = df['LTS_position'].value_counts()[df['LTS_position'][0]]
        #create a set of unique values of LTS_position
        unique = set(df['LTS_position'])
        if len(unique) > 1:
            #create an array of increasing count values repeated a number of times equal to len(unique)
            count_arr = np.repeat(np.arange(count), len(unique))
            count_arr = count_arr[:len(df['LTS_position'])]
            #reshape the array to be the same length as df['LTS_position'] and drop elements if necessary
            
            
            # count_arr = count_arr.reshape(df['LTS_position'].shape)
            #create a column called 'LTS_loop' and assign it the values of count_arr
            df['LTS_loop'] = count_arr+1
    return df



#transform a sting of numbers separated by commas into a list of integers with ':' denoting a range of numbers
def str_to_list(string):
    list = []
    for i in string.split(','):
        if ':' in i:
            list.extend(range(int(i.split(':')[0]), int(i.split(':')[1]) + 1))
        else:
            list.append(int(i))
    return list


def loops_selector():
    global df, img_arr,w
    selected_loops, ok = QtWidgets.QInputDialog.getText(w, "Select loops", f"Separate values with ',' and ranges with ':'\nThere are {df['LTS_loop'].max()} loops\nExample: 1,2,3:5 will select loops 1,2,3,4,5")
    if ok:
        selected_loops = str_to_list(selected_loops)
        df = df[df['LTS_loop'].isin(selected_loops)]
        img_arr = img_arr[df.index]
        df = df.reset_index(drop=True)
        scroll_data(0)
        
w.actionSelect_loops.triggered.connect(loops_selector)


def check_shutter():
    global df, img_arr
    if 'shutter' in df.columns:
        #only keep frames where shutter is 1
        if len(set(df['shutter'])) > 1:
            df = df[df['shutter'] == 1]
            img_arr = img_arr[df.index]
            #reset index of df
            df = df.reset_index(drop=True)

def remove_hot(imgON, imgOFF, n=2):
    print(np.shape(imgON))
    ratio = []
    if len(np.shape(imgON)) > 2:
        for o in range(np.shape(imgON)[0]):
            diff = np.divide(imgON[o], imgOFF[o], out=np.ones_like(imgON[o]), where=imgOFF[o]!=0)
            mean=np.mean(diff)
            s = np.std(diff)
            pic_pos_above = np.where(diff > n)
            pic_pos_below = np.where(diff < 1/n)
            pic_pos_off = np.where((imgOFF[o] == 0) & (imgON[o]>n))
            for i in range(len(pic_pos_off[0])):
                imgON[o][pic_pos_off[0][i], pic_pos_off[1][i]] = 0
            for i in range(len(pic_pos_above[0])):
                imgON[o][pic_pos_above[0][i], pic_pos_above[1][i]] = imgOFF[o][pic_pos_above[0][i], pic_pos_above[1][i]]
            for i in range(len(pic_pos_below[0])):
                imgON[o][pic_pos_below[0][i], pic_pos_below[1][i]] = imgOFF[o][pic_pos_below[0][i], pic_pos_below[1][i]]
            ratio.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
    else:
        print('one image processing')
        diff = np.divide(imgON, imgOFF, out=np.ones_like(imgON), where=imgOFF!=0)
        pic_pos_above = np.where(diff > n)
        pic_pos_below = np.where(diff < 1/n)
        pic_pos_off = np.where((imgOFF == 0) & (imgON > n))
        ratio.append(1e2*(len(pic_pos_above[0])+len(pic_pos_below[0])+len(pic_pos_off[0]))/(512*512))
        for i in range(len(pic_pos_off[0])):
            imgON[pic_pos_off[0][i], pic_pos_off[1][i]] = 0
        for i in range(len(pic_pos_above[0])):
            imgON[pic_pos_above[0][i], pic_pos_above[1][i]] = imgOFF[pic_pos_above[0][i], pic_pos_above[1][i]]
        for i in range(len(pic_pos_below[0])):
            imgON[pic_pos_below[0][i], pic_pos_below[1][i]] = imgOFF[pic_pos_below[0][i], pic_pos_below[1][i]]
    print('average hot pixels = ',np.mean(np.array(ratio)))
    return imgON, imgOFF

def remove_hot_instruction():
    global img_arr, img_arr_2
    img_arr, img_arr_2 = remove_hot(img_arr, img_arr_2, n=2)
    # scroll_data(0)

w.actionRemove_hot_n_2.triggered.connect(remove_hot_instruction)


def load_npy():
    global img_arr, df
    path = find_drive_by_label() + r"\\My Drive\\UED_measurements\\"
    fname = ''
    fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select File", path, "Numpy Files (*.npy)")
    img_arr = np.load(fname[0])
    length = img_arr.shape[0]
    df = pd.DataFrame(np.arange(length), columns=["frame"])
    scroll_data(0)


def load_single_SPE():
    global img_arr, df, w
    path = find_drive_by_label() + r"\\My Drive\\UED_measurements\\"
    fname = ''
    fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select File", path, "SPE Files (*.SPE)")
    img1 = iio.imread(fname[0])
    img_arr = np.array([img1], dtype=np.int32)

    df = pd.DataFrame({key: pd.Series(value) for key, value in dict(img1.meta).items()})

    scroll_data(0)


def check_image_range(img_list):
    d1 = len(img_list)
    d2 = img_list[0].shape[0]
    d3 = img_list[0].shape[1]
    img_arr = np.empty((d1, d2, d3), dtype=np.int32)
    coords = (np.array(np.where(img_list[0] > 1e7)))
    for i in range(len(img_list)):
        if len(coords[0]) > 0:
            img_list[0][coords[0], coords[1]] = 0
        img_arr[i] = np.abs(img_list[0])
        img_list = img_list[1:]
    return img_arr.astype(np.uint32)


def save_array_to_npy():
    global img_arr, starting_path
    fname = QtWidgets.QFileDialog.getSaveFileName(w, "Save to NPY", starting_path, "Numpy Files (*.npy)")
    np.save(fname[0], img_arr)


def save_image_to_npy():
    global img, starting_path
    fname = QtWidgets.QFileDialog.getSaveFileName(w, "Save to NPY", starting_path, "Numpy Files (*.npy)")
    np.save(fname[0], img)


w.actionImage.triggered.connect(save_image_to_npy)
w.actionArray.triggered.connect(save_array_to_npy)
w.actionOpen_SPE.triggered.connect(load_single_SPE)


def load_single_H5_serie():
    global img_arr, df
    path = find_drive_by_label() + r"\\My Drive\\UED_measurements\\"
    fname = ''
    fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select File", path, "H5 Files (*.h5)")
    with h5py.File(fname[0], 'r') as h5:
        dset_name = list(h5.keys())[0]
        dset = h5[dset_name]
        # attributes = list(dset.attrs.keys())
        df1 = pd.DataFrame([dict(dset.attrs.items())], index=[0])
        df = df1.explode(list(df1.columns))
        df.reset_index(drop=True, inplace=True)
        df['LTS_position'] = np.stack(df['LTS_position'])
        df['Delay_ps'] = np.stack(df['LTS_position'])*6.66
        img_arr = np.array(dset[:], dtype=np.int64)
        coords = (np.array(np.where(img_arr > 1e7)))
        img_arr[coords[0], coords[1], coords[2]] = 0
        img_arr = np.rot90(img_arr, axes=(1, 2))
    df = counting_loops(df)
    scroll_data(0)
    plot_metadata()

def get_file_from_ftp(filename):
    #replace '\' with '/' in filename
    
    new_filename = filename.replace('\\', '/')
    new_filename = new_filename[2:]
    if filename[0] != 'Z':
        return filename
    else:
        ftp = ftplib.FTP('192.168.1.9', 'anonymous')
        r = io.BytesIO()
        comm = 'RETR ' + new_filename
        ftp.retrbinary(comm, r.write)
        return r
    
def check_if_serie(file_list):
    file = file_list[0]
    file = get_file_from_ftp(file)
    with h5py.File(file, 'r') as h5:
        dset_name = list(h5.keys())[0]
        dset = h5[dset_name]
        attributes = list(dset.attrs.keys())
        df1 = pd.DataFrame([dict(dset.attrs.items())], index=[0])
        print(len(df1[attributes[0]][0]))
        if np.array(df1[attributes[0]][0]).shape[0] > 1:
            return True
        else:
            return False


def load_H5_serie(file_list=None):
    global img_arr, df
    path = find_drive_by_label() + r"\\My Drive\\UED_measurements\\"
    
    if file_list is None or file_list is False:
        path = QtWidgets.QFileDialog.getExistingDirectory(w, "Select Folder", path)
        if path != '':
            # chosen = True
            list_h5 = natsort.natsorted(glob.glob(path + '/' + '*.h5'))
            file_list = list_h5
    for file in file_list:
        if file == file_list[0]:
            file = get_file_from_ftp(file)
            with h5py.File(file, 'r') as h5:
                dset_name = list(h5.keys())[0]
                dset = h5[dset_name]
                attributes = list(dset.attrs.keys())
                df1 = pd.DataFrame([dict(dset.attrs.items())], index=[0])
                df = df1.explode(attributes)
                df.reset_index(drop=True, inplace=True)
                df['LTS_position'] = np.stack(df['LTS_position'])
                df['Delay_ps'] = np.stack(df['LTS_position'])*6.66
                if 'Time_for_humans' not in df.columns:
                    times = df['Time'].to_list()
                    tfh = [str(dtm.fromtimestamp(float(x))) for x in times]
                    df['Time_for_humans'] = tfh

                print(f"{time.time()-t_start:.3f}s Attributes set")
                img_arr = dset[:]
                print(f"{time.time()-t_start:.3f}s Img_arr set")
                coords = (np.array(np.where(img_arr[0] > 1e7)))
                
                print(f"{time.time()-t_start:.3f}s Coords set")
                img_arr[:, coords[0], coords[1]] = 0
                print(f"{time.time()-t_start:.3f}s Zeros set")
                img_arr = np.rot90(img_arr, axes=(1, 2))
                print(f"{time.time()-t_start:.3f}s Rotation set")
                scroll_data(0)
        else:
            file = get_file_from_ftp(file)
            with h5py.File(file, 'r') as h5:
                dset_name = list(h5.keys())[0]
                dset = h5[dset_name]
                img_arr1 = np.array(dset[:], dtype=np.int64)
                coords = (np.array(np.where(img_arr1 > 1e7)))
                img_arr1[coords[0], coords[1], coords[2]] = 0
                img_arr1 = np.rot90(img_arr1, axes=(1, 2))
                img_arr = (img_arr + img_arr1)/2
                scroll_data(0)
        # print(coords)
        # df1.to_pickle("test.pickle")
        # print(df1)
    plot_metadata()


def check_chunks(file_list):
    sum_chunks = 0
    for file in file_list:
        with h5py.File(file, 'r') as h5:
            dset_name = list(h5.keys())[0]
            dset = h5[dset_name]
            sum_chunks += dset.id.get_num_chunks()
    return sum_chunks

w.actionOpen_NPY.triggered.connect(load_npy)
w.actionRadial_Average.triggered.connect(w_radial_average.show)
w.actionOpen_single_H5_serie.triggered.connect(load_H5_serie)

loading_thread = 0
fit_number = 0


def open_folder():
    global loading_thread, fit_number, img_arr
    fit_number = 0
    loading_function(zipped=False)
    print('loading done properly')
    scroll_data(0)
    print('scroll done properly')
    plot_metadata()
    print('plot done properly')


def open_ZIP():
    global loading_thread, fit_number
    fit_number = 0
    loading_function(zipped=True)
    scroll_data(0)
    plot_metadata()


def loading_pb(i):
    w_prog.progressBar.setValue(int(i))

def rotate_array():
    global img_arr
    angle = w.doubleSpinBox.value()
    nd.rotate(img_arr, angle, axes=(1, 2), reshape=False, output=img_arr)
    scroll_data(0)

# def get_quality():
#     global imgON
    
    
    
w.pushButton_5.clicked.connect(rotate_array)
# w.pushButton_6.clicked.connect(get_quality)



def plot_metadata():
    global df
    try:
        x_pressure = np.arange(np.stack(df['Pressure']).shape[0])
        y_pressure = np.stack(df['Pressure'])
        plot_p.setData(x_pressure, y_pressure, clear=True, pen=pg.mkPen('m', width=2))
        vLine_p.setPos(0)
        y_temperatureA = np.stack(df['Temperature_A'])
        y_temperatureB = np.stack(df['Temperature_B'])
        plot_tA.setData(x_pressure, y_temperatureA, clear=True, pen=pg.mkPen('g', width=2))
        plot_tB.setData(x_pressure, y_temperatureB, clear=True, pen=pg.mkPen('r', width=2))
        vLine_t.setPos(0)
    except Exception:
        print(f'{bcolors.FAIL}Error in plot_metadata{bcolors.ENDC}')


image_offset = [0, 0]
image_scale = [1, 1]

w.actionOpen_Folder.triggered.connect(open_folder)
w.actionOpen_ZIP.triggered.connect(open_ZIP)
img = img_arr[0].copy()
w.horizontalScrollBar.setMaximum(img_arr.shape[0] - 1)
w_radial_average.horizontalScrollBar.setMaximum(img_arr.shape[0] - 1)
baseplot = pg.PlotItem()
w.widget = pg.ImageView(w.frame_2, view=baseplot)
w.gridLayout.addWidget(w.widget, 0, 1)
w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
w.widget.autoRange()
w.widget.autoLevels()
w.widget.autoHistogramRange()
w.widget.setColorMap(cmap)
w.widget.ui.roiBtn.hide()
w.widget.ui.menuBtn.hide()
roi = pg.ROI([0, 0], [200, 50], pen=pg.mkPen('m', width=2), hoverPen=pg.mkPen('m', width=4))
roi.setSize([img_arr.shape[2]*image_scale[0]*0.1, img_arr.shape[1]*image_scale[1]*0.1], update=True)
roi.addRotateHandle([0, 0], [0.5, 0.5])
roi.addScaleHandle([1, 1], [0, 0])
roi.setZValue(1e9)
w.widget.addItem(roi)
w.roi_chkbx.setChecked(True)
plot_v = w.widget_3.plot()
#set plot title
w.widget_3.setTitle("Vertical Binning")
plot_v_fit = w.widget_3.plot()
plot_h = w.widget_2.plot()
w.widget_2.setTitle("Horizontal Binning")
plot_h_fit = w.widget_2.plot()
fit_dict = {}
vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1))
fwhm_roi = pg.EllipseROI(pos=[0, 0], size=[1, 1], pen=pg.mkPen('m', width=2), movable=False)
w.widget.addItem(vLine)
w.widget.addItem(hLine)
w.widget.addItem(fwhm_roi)
vLine.hide()
hLine.hide()
fwhm_roi.hide()

[fwhm_roi.removeHandle(h) for h in fwhm_roi.getHandles()]

w.widget_4.setTitle("Pressure")
#set x and y labels
w.widget_4.setLabel('left', "Pressure (mbar)")
w.widget_4.setLabel('bottom', "Frame")
plot_p = w.widget_4.plot()
vLine_p = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
w.widget_4.addItem(vLine_p)
w.widget_5.setTitle("Temperature")
w.widget_5.setLabel('left', "Temperature (K)")
w.widget_5.setLabel('bottom', "Frame")
plot_tA = w.widget_5.plot()
plot_tB = w.widget_5.plot()
vLine_t = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1))
w.widget_5.addItem(vLine_t)

w.widget_6.setTitle("ROI Evolution")
plot_evolution = w.widget_6.plot()


def roi_show(i):
    if i == 0:
        w.widget.removeItem(roi)
        w.widget.removeItem(vLine)
        w.widget.removeItem(hLine)
    if i == 2:
        w.widget.addItem(roi)
        w.widget.addItem(vLine)
        w.widget.addItem(hLine)


def roi_autoscale():
    global colormap
    selected = roi.getArrayRegion(img, w.widget.getImageItem())
    minimum = np.min(selected)*0
    maximum = np.max(selected)
    w.widget.setLevels(minimum, maximum)
    w.widget.setHistogramRange(minimum, maximum)
    if colormap == 'Hot-Cold':
        min = np.min(selected)
        max = np.max(selected)
        biggest = max if max > abs(min) else abs(min)
        w.widget.setLevels(-biggest, biggest)
        w.widget.setHistogramRange(-biggest, biggest)

w.pushButton.clicked.connect(roi_autoscale)

#set roi scaleSnap when w.checkBox_2 is checked, otherwise set to False
def roi_scale_snap(i):
    if i == 0:
        roi.scaleSnap = False
        roi.translateSnap = False
    if i == 2:
        roi.scaleSnap = True
        roi.translateSnap = True

w.checkBox_2.stateChanged.connect(roi_scale_snap)


def generate_dummy_data():
    global img_arr, df
    

    #generate an array of 500 images 512x512 with value 1
    img_arr = np.ones((1500, 512, 512), dtype=np.uint32)
    #generate an array of 50...100 repeated 150 times
    delays = np.tile(np.arange(50, 100), 30)
    df = pd.DataFrame({'LTS_position': delays})
    scroll_data(0)

w.actionDummy_data.triggered.connect(generate_dummy_data)




def roi_update_plot(ro=0, index=0):
    global img, fit_dict, vLine, hLine, fwhm_roi, image_scale, image_offset, df
    selected = roi.getArrayRegion(img, w.widget.getImageItem())
    data_v = selected.sum(axis=0)
    data_h = selected.sum(axis=1)
    data_count = selected.sum()
    x_data_v = (np.arange(len(data_v))+image_offset[0])*image_scale[0]
    x_data_v = x_data_v + roi.pos()[0]
    x_data_h = (np.arange(len(data_h))+image_offset[1])*image_scale[1]
    x_data_h = x_data_h + roi.pos()[1]

    plot_v.setData(x_data_v, data_v, clear=True, pen=pg.mkPen('m', width=2))
    plot_h.setData(x_data_h, data_h, clear=True, pen=pg.mkPen('m', width=2))
    
    roi_dict = {'x_i': roi.pos()[0], 'x_f' : roi.pos()[0] + len(data_v), 'y_i': roi.pos()[1], 'y_f' : roi.pos()[1] + len(data_h), 'width': len(data_v), 'height': len(data_h), 'angle': roi.angle()}
    #print roi_dict to w.label_2 with each entry in a row
    w.label_2.setText(f"ROI x: [{roi_dict['x_i']:.2f} , {roi_dict['x_f']:.2f}]\nROI y:[{roi_dict['y_i']:.2f} , {roi_dict['y_f']:.2f}] \nROI width: {roi_dict['width']:.2f}\nROI height: {roi_dict['height']:.2f}\nROI angle: {roi_dict['angle']:.2f}\nROI count : {data_count:.4e}")

    
    if w.checkBox_3.isChecked():
        x = df[w.comboBox.currentText()]
        z = x*0.
        # y = np.random.rand(len(df['LTS_position']))
        y = np.array([roi.getArrayRegion(image, w.widget.getImageItem()) for image in img_arr])
        y_text =  w.comboBox_2.currentText()
        if y_text == 'Sum':
            z = np.sum(y, axis=(1, 2))
        if y_text == 'Avg':
            z = np.mean(y, axis=(1, 2))
        if y_text == 'CoMx':
            #calculate center of mass for each image
            z = np.array([nd.measurements.center_of_mass(y[i]) for i in range(len(y))])[:,1]
        if y_text == 'CoMy':
            #calculate center of mass for each image
            z = np.array([nd.measurements.center_of_mass(y[i]) for i in range(len(y))])[:,0]
        if y_text == 'Pos of xMax':
            #calculate position of max for each image
            z = np.array([np.unravel_index(np.argmax(y[i]), y[i].shape) for i in range(len(y))])[:,1]
        if y_text == 'Pos of yMax':
            #calculate position of max for each image
            z = np.array([np.unravel_index(np.argmax(y[i]), y[i].shape) for i in range(len(y))])[:,0]
        
        # print(z.shape)
        plot_evolution.setData(x, z, clear=True, pen=pg.mkPen('m', width=2))
        #set plot_evolution also with symbol='o' and symbolPen='m'
        plot_evolution.setSymbol('o')
        

    if w.fit_chkbx.isChecked():
        vLine.show()
        hLine.show()
        fwhm_roi.show()
        if w.fit_G.isChecked():
            try:
                data_v_fit = fit_gauss(data_v, x_data_v)
                data_h_fit = fit_gauss(data_h, x_data_h)
            except Exception:
                pass
        if w.fit_V.isChecked():
            try:
                data_v_fit = fit_voigt(data_v, x_data_v)
                data_h_fit = fit_voigt(data_h, x_data_h)
            except Exception:
                pass
        plot_v_fit.setData(x_data_v, data_v_fit.best_fit, clear=True, pen=pg.mkPen('y', width=2))
        plot_h_fit.setData(x_data_h, data_h_fit.best_fit, clear=True, pen=pg.mkPen('y', width=2))
        pos_x = data_v_fit.values['center'] * np.cos(roi.angle() % 360 / 180 * np.pi) - data_h_fit.values['center'] * np.sin(roi.angle() % 360 / 180 * np.pi)
        pos_y = data_v_fit.values['center'] * np.sin(roi.angle() % 360 / 180 * np.pi) + data_h_fit.values['center'] * np.cos(roi.angle() % 360 / 180 * np.pi)
        fit_dict = {
            "frame": index + 1,
            "Height_OOP": data_v_fit.values['height'],
            "Height_IP": data_h_fit.values['height'],
            "Amplitude_OOP": data_v_fit.values['amplitude'],
            "Amplitude_IP": data_h_fit.values['amplitude'],
            "Center_OOP": pos_x,
            "Center_IP": pos_y,
            "FWHM_OOP": data_v_fit.values['fwhm'],
            "FWHM_IP": data_h_fit.values['fwhm'],
            "ROI_x": roi.pos().x(),
            "ROI_y": roi.pos().y(),
            "ROI_Lx": roi.size().x()*image_scale[0],
            "ROI_Ly": roi.size().y()*image_scale[1],
            "ROI_angle": roi.angle() % 360
        }
        model = TableModel(pd.DataFrame(fit_dict, index=[0]).transpose())
        w.tableView.setModel(model)
        w.tableView.horizontalHeader().setStretchLastSection(True)
        hLine.setPos(fit_dict['Center_IP'])
        vLine.setPos(fit_dict['Center_OOP'])
        try:
            fwhm_roi.setPos([pos_x - data_v_fit.values['fwhm'] / 2., pos_y - data_h_fit.values['fwhm'] / 2.])
            fwhm_roi.setSize([data_v_fit.values['fwhm'], data_h_fit.values['fwhm']])
        except Exception:
            pass
    else:
        plot_v_fit.clear()
        plot_h_fit.clear()
        vLine.hide()
        hLine.hide()
        fwhm_roi.hide()

def setup_evolution():
    global df
    columns = df.select_dtypes(include='number').columns
    #set the columns to w.comboBox
    w.comboBox.addItems(columns)
    #set w.comboBox to be LTS_position by default
    w.comboBox.setCurrentText('LTS_position')
    y_columns = ['Avg', 'Sum', 'CoMx', 'CoMy', 'Pos of xMax','Pos of yMax']
    w.comboBox_2.addItems(y_columns)
    #set labels of x and y axis
def evolution_labels():
    w.widget_6.setLabel('left', w.comboBox_2.currentText())
    w.widget_6.setLabel('bottom', w.comboBox.currentText())
    
w.checkBox_3.stateChanged.connect(setup_evolution)
w.checkBox_3.stateChanged.connect(roi_update_plot)
w.comboBox.currentIndexChanged.connect(evolution_labels)
w.comboBox.currentIndexChanged.connect(roi_update_plot)
w.comboBox_2.currentIndexChanged.connect(roi_update_plot)
w.comboBox_2.currentIndexChanged.connect(evolution_labels)


roi.sigRegionChanged.connect(roi_update_plot)
w.roi_chkbx.stateChanged.connect(roi_show)
w.fit_G.clicked.connect(roi_update_plot)
w.fit_V.clicked.connect(roi_update_plot)
w.fit_chkbx.clicked.connect(roi_update_plot)


def downsample16bit():
    global img_arr, img
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    selected = roi.getArrayRegion(img, w.widget.getImageItem())
    x, y = np.indices(selected.shape)
    x = x + int(roi.pos().x())
    y = y + int(roi.pos().y())
    for img1 in img_arr:
        img1[y, x] = np.uint32(img1[y, x]/50.)
    img_arr = img_arr.copy().astype(np.uint16)

    print(sys.getrefcount(img_arr))
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


w.actionDownsample_to_16bit.triggered.connect(downsample16bit)


def plot_sum():
    global img_arr, img, image_offset, image_scale, z_scale_mode
    
    img = np.sum(img_arr, axis=0)
    if z_scale_mode == 'log':
        img = np.log(img)
    elif z_scale_mode == 'arcsinh':
        img = np.arcsinh(img)
    elif z_scale_mode == 'square':
        img = img**2
    w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
    w.checkBox.setChecked(False)


def plot_sum_range():
    global img_arr, img, image_offset, image_scale
    x1 = int(w.spinBox.value())
    x2 = int(w.spinBox_2.value())
    img = np.sum(img_arr[x1:x2], axis=0)
    w.sum_chkbx.setChecked(False)
    w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)


current_frame = 0
z_scale_mode = 'linear'

def set_z_scale_mode(mode):
    global z_scale_mode, current_frame
    z_scale_mode = mode
    if mode == 'linear':
        w.actionLog.setChecked(False)
        w.actionArcsinh.setChecked(False)
        w.actionSquare.setChecked(False)
    elif mode == 'log':
        w.actionLinear.setChecked(False)
        w.actionArcsinh.setChecked(False)
        w.actionSquare.setChecked(False)
    elif mode == 'arcsinh':
        w.actionLinear.setChecked(False)
        w.actionLog.setChecked(False)
        w.actionSquare.setChecked(False)
    scroll_data(current_frame)
    w.widget.autoRange()
    w.widget.autoLevels()
    w.widget.autoHistogramRange()
    
w.actionLinear.triggered.connect(lambda: set_z_scale_mode('linear'))
w.actionLog.triggered.connect(lambda: set_z_scale_mode('log'))
w.actionArcsinh.triggered.connect(lambda: set_z_scale_mode('arcsinh'))
w.actionSquare.triggered.connect(lambda: set_z_scale_mode('square'))

def set_colormap():
    global colormap, current_frame, img
    colormap = w.comboBox_3.currentText()
    if colormap == 'Grayscale':
        w.widget.setColorMap(pg.colormap.get('CET-L1'))
    elif colormap == 'Rainbow':
        w.widget.setColorMap(pg.colormap.get('CET-R4'))
    elif colormap == 'Hot-Cold':
        w.widget.setColorMap(pg.colormap.get('CET-D1'))
    scroll_data(current_frame)
    
        
w.comboBox_3.currentTextChanged.connect(set_colormap)
    

def scroll_data(i):
    global img, img_arr, df, current_frame, image_offset, image_scale, z_scale_mode, colormap
        
    w.horizontalScrollBar.setMaximum(img_arr.shape[0] - 1)
    if z_scale_mode == 'linear':
        img = img_arr[i]
    elif z_scale_mode == 'log':
        img = np.log(img_arr[i])
    elif z_scale_mode == 'arcsinh':
        img = np.arcsinh(img_arr[i])
    elif z_scale_mode == 'square':
        img = img_arr[i]**2
    
    w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)

    w.label_3.setText("%d of %d" % (i + 1, img_arr.shape[0]))
    if w.roi_chkbx.isChecked() is True:
        roi_update_plot(index=i)
    else:
        pass
    w.sum_chkbx.setChecked(False)
    w.checkBox.setChecked(False)
    vLine_p.setPos(i)
    vLine_t.setPos(i)
    text = ''
    try:
        w.notes_lbl.setText(df['notes'][i])
    except Exception:
        pass
    print('columns in df ', df.columns)
    print('columns in df[i] ',df.columns[i])
    print('length of df ', len(df))
    for column in df.columns:
        if column != 'notes':
            # print(df['delay'][i])
            print(column)
            print(df[column][i])
            col = df[column][i]
            if isinstance(col, numbers.Number):
                b = format(col, '.3g')
            else:
                b = str(col)
            text = text + str(column) + ':\t' + b + '\n'
    model = TableModel(df.iloc[[i]].transpose())
    w.tableView_2.setModel(model)
    w.tableView_2.horizontalHeader().setStretchLastSection(True)
    current_frame = i
    print('scroll data went well, current frame is', current_frame)

set_colormap()

w.horizontalScrollBar.valueChanged.connect(scroll_data)

w.sum_chkbx.clicked.connect(plot_sum)
w.checkBox.clicked.connect(plot_sum_range)


fitted_data = 0
df_fits = 0


class fit_data_thread_class(QThread):
    global fit_dict, img_arr, fitted_data, df, df_fits
    finished = QtCore.pyqtSignal(int)
    updated = QtCore.pyqtSignal(float)

    def __init__(self, isVoigt=False, isFitting=True):
        super(QThread, self).__init__()
        self.df_fit = 0
        scroll_data(0)
        self.len_data = img_arr.shape[0]
        self.fits_list = []
        self.isVoigt = isVoigt
        self.isFitting = isFitting
        if self.isFitting:
            w.fit_chkbx.setChecked(1)

    def run(self):
        global df_fits, image_scale, image_offset
        self.data_v_fit = 0
        self.data_h_fit = 0
        if self.isFitting:
            for self.i, self.img in enumerate(img_arr):
                self.selected = roi.getArrayRegion(self.img, w.widget.getImageItem())
                self.data_v = self.selected.sum(axis=0)
                self.data_h = self.selected.sum(axis=1)
                self.x_data_v = (np.arange(len(self.data_v))+image_offset[0])*image_scale[0]
                self.x_data_v = self.x_data_v + roi.pos()[0]
                self.x_data_h = (np.arange(len(self.data_h))+image_offset[1])*image_scale[1]
                self.x_data_h = self.x_data_h + roi.pos()[1]
                try:
                    if self.isVoigt:
                        self.data_v_fit = fit_voigt(self.data_v, self.x_data_v)
                    else:
                        self.data_v_fit = fit_gauss(self.data_v, self.x_data_v)
                except Exception:
                    print(f'{bcolors.FAIL}Error in frame {self.i} vertical binning{bcolors.ENDC}')
                    pass
                try:
                    if self.isVoigt:
                        self.data_h_fit = fit_voigt(self.data_h, self.x_data_h)
                    else:
                        self.data_h_fit = fit_gauss(self.data_h, self.x_data_h)
                except Exception:
                    print(f'{bcolors.FAIL}Error in frame {self.i} horizontal binning{bcolors.ENDC}')
                    pass
                self.pos_x = self.data_v_fit.values['center'] * np.cos(
                    roi.angle() % 360 / 180 *
                    np.pi) - self.data_h_fit.values['center'] * np.sin(
                        roi.angle() % 360 / 180 * np.pi)
                self.pos_y = self.data_v_fit.values['center'] * np.sin(
                    roi.angle() % 360 / 180 *
                    np.pi) + self.data_h_fit.values['center'] * np.cos(
                        roi.angle() % 360 / 180 * np.pi)
                self.fit_dict = {
                    "frame": self.i + 1,
                    "Height_OOP": self.data_v_fit.values['height'],
                    "Height_IP": self.data_h_fit.values['height'],
                    "Amplitude_OOP": self.data_v_fit.values['amplitude'],
                    "Amplitude_IP": self.data_h_fit.values['amplitude'],
                    "Center_OOP": self.pos_x,
                    "Center_IP": self.pos_y,
                    "FWHM_OOP": self.data_v_fit.values['fwhm'],
                    "FWHM_IP": self.data_h_fit.values['fwhm'],
                    "ROI_x": roi.pos().x(),
                    "ROI_y": roi.pos().y(),
                    "ROI_Lx": roi.size().x()*image_scale[0],
                    "ROI_Ly": roi.size().y()*image_scale[1],
                    "ROI_angle": roi.angle() % 360
                }
                self.fits_list.append(self.fit_dict)
                self.updated.emit((self.i + 1) / len(img_arr) * 100)
            df_fits = pd.DataFrame(self.fits_list)
            self.finished.emit(1)
        else:
            for self.i, self.img in enumerate(img_arr):
                self.selected = roi.getArrayRegion(self.img, w.widget.getImageItem())
                self.roi_avg = np.sum(self.selected)
                self.fit_dict = {
                    "frame": self.i + 1,
                    "Avg Counts": self.roi_avg,
                    "ROI_x": roi.pos().x(),
                    "ROI_y": roi.pos().y(),
                    "ROI_Lx": roi.size().x(),
                    "ROI_Ly": roi.size().y(),
                    "ROI_angle": roi.angle() % 360
                }
                self.fits_list.append(self.fit_dict)
                self.updated.emit((self.i + 1) / len(img_arr) * 100)
            df_fits = pd.DataFrame(self.fits_list)
            self.finished.emit(1)


fit_data_thread = 0
list_of_fit_datasets = []
list_of_fit_datasets_names = []

def data_fitter(isVoigt=False, isFitting=True, dlg = None, roi = roi):
    global df_fits, image_scale, image_offset, img_arr
    # df_fit = 0
    scroll_data(0)
    # len_data = img_arr.shape[0]
    fits_list = []
    data_h_fit = 0 
    data_v_fit = 0
    if isFitting:
        w.fit_chkbx.setChecked(1)
        for i, img in enumerate(img_arr):
            if dlg is not None:
                if dlg.wasCanceled():
                    break
            selected = roi.getArrayRegion(img, w.widget.getImageItem())
            data_v = selected.sum(axis=0)
            data_h = selected.sum(axis=1)
            x_data_v = (np.arange(len(data_v))+image_offset[0])*image_scale[0]
            x_data_v = x_data_v + roi.pos()[0]
            x_data_h = (np.arange(len(data_h))+image_offset[1])*image_scale[1]
            x_data_h = x_data_h + roi.pos()[1]
            try:
                if isVoigt:
                    data_v_fit = fit_voigt(data_v, x_data_v)
                else:
                    data_v_fit = fit_gauss(data_v, x_data_v)
            except Exception:
                show_error(f'Error in frame {i} vertical binning')
                print(f'{bcolors.FAIL}Error in frame {i} vertical binning{bcolors.ENDC}')
                pass
            try:
                if isVoigt:
                    data_h_fit = fit_voigt(data_h, x_data_h)
                else:
                    data_h_fit = fit_gauss(data_h, x_data_h)
            except Exception:
                show_error(f'Error in frame {i} horizontal binning')
                print(f'{bcolors.FAIL}Error in frame {i} horizontal binning{bcolors.ENDC}')
                pass
            pos_x = data_v_fit.values['center'] * np.cos(
                roi.angle() % 360 / 180 *
                np.pi) - data_h_fit.values['center'] * np.sin(
                    roi.angle() % 360 / 180 * np.pi)
            pos_y = data_v_fit.values['center'] * np.sin(
                roi.angle() % 360 / 180 *
                np.pi) + data_h_fit.values['center'] * np.cos(
                    roi.angle() % 360 / 180 * np.pi)
            fit_dict = {
                "frame": i + 1,
                "Height_OOP": data_v_fit.values['height'],
                "Height_IP": data_h_fit.values['height'],
                "Amplitude_OOP": data_v_fit.values['amplitude'],
                "Amplitude_IP": data_h_fit.values['amplitude'],
                "Center_OOP": pos_x,
                "Center_IP": pos_y,
                "FWHM_OOP": data_v_fit.values['fwhm'],
                "FWHM_IP": data_h_fit.values['fwhm'],
                "ROI_x": roi.pos().x(),
                "ROI_y": roi.pos().y(),
                "ROI_Lx": roi.size().x()*image_scale[0],
                "ROI_Ly": roi.size().y()*image_scale[1],
                "ROI_angle": roi.angle() % 360
            }
            fits_list.append(fit_dict)
            if dlg is not None:
                dlg +=1
        df_fits = pd.DataFrame(fits_list)
    else:
        for i, img in enumerate(img_arr):
            selected = roi.getArrayRegion(img, w.widget.getImageItem())
            roi_avg = np.sum(selected)
            fit_dict = {
                "frame": i + 1,
                "Avg Counts": roi_avg,
                "ROI_x": roi.pos().x(),
                "ROI_y": roi.pos().y(),
                "ROI_Lx": roi.size().x(),
                "ROI_Ly": roi.size().y(),
                "ROI_angle": roi.angle() % 360
            }
            fits_list.append(fit_dict)
        df_fits = pd.DataFrame(fits_list)

fit_datasets_dict = []

def show_fits(number = None):
    global df_fits, df, list_of_fit_datasets, filename_opened, fit_number, list_rois_numbers, fit_datasets_dict
    w_prog.close()
    list_of_fit_datasets.append(pd.concat([df, df_fits], axis=1))
    if number is None:
        list_of_fit_datasets_names.append(f"{filename_opened}_{fit_number:03d}")
        w_plot.show()
    else:
        list_of_fit_datasets_names.append(f"{filename_opened}_{fit_number:03d}_ROI{number:03d}")
    fit_datasets_dict.append(list_of_fit_datasets[-1])
    w_plot.listWidget.clear()
    for i, obj in enumerate(list_of_fit_datasets):
        w_plot.listWidget.insertItem(i, f'{list_of_fit_datasets_names[i]}')

def clear_fits():
    global list_of_fit_datasets, list_of_fit_datasets_names, fit_datasets_dict
    list_of_fit_datasets = []
    list_of_fit_datasets_names = []
    w_plot.listWidget.clear()
    fit_datasets_dict = []
    
    
def show_pdgui():
    global fit_datasets_dict
    print(fit_datasets_dict)
    pdgui.show(*fit_datasets_dict)
    
w_plot.Clear_Button.clicked.connect(clear_fits)
w_plot.actionShow_in_PdGUI.triggered.connect(show_pdgui)



df_columns = []
df_row = 0


def list_columns():
    global df_columns, df_row, list_of_fit_datasets
    df_row = w_plot.listWidget.currentRow()
    w_plot.listWidget_2.clear()
    df_columns = list_of_fit_datasets[df_row].columns
    w_plot.listWidget_2.insertItems(0, df_columns)
    w_plot.listWidget_3.clear()
    df_columns = list_of_fit_datasets[df_row].columns
    w_plot.listWidget_3.insertItems(0, df_columns)
    try:
        plot_column()
    except Exception:
        pass

region = pg.LinearRegionItem()
region.setZValue(10)

def plotter_enable_roi():
    if w_plot.actionROI.isChecked():
        w_plot.plot_roi.show()
        w_plot.gridLayout_2.setRowStretch(0, 1)
        w_plot.plot.addItem(region, ignoreBounds=True)
    else:
        w_plot.plot_roi.hide()
        w_plot.gridLayout_2.setRowStretch(0, 0)
        w_plot.plot.removeItem(region)

w_plot.actionROI.triggered.connect(plotter_enable_roi)
w_plot.gridLayout_2.setRowStretch(0, 0)
w_plot.plot_roi.hide()

p1 = 0
p2 = 0
p1 = w_plot.plot.plot([0], [0], pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(255, 0, 0, 128), clear=True)
p3 = w_plot.plot_roi.plot([0], [0], pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(255, 0, 0, 128), clear=True)
p1p = w_plot.plot.getPlotItem()
p_roi = w_plot.plot_roi.getPlotItem()
region.setClipItem(p1p)

def update():
    region.setZValue(10)
    minX, maxX = region.getRegion()
    data = p1.getData()
    p3.setData(data[0], data[1])
    # p_roi.plot(data[0], data[1], pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(255, 0, 0, 128), clear=True)
    p_roi.setXRange(minX, maxX, padding=0)    

region.sigRegionChanged.connect(update)

def updateRegion(window, viewRange):
    rgn = viewRange[0]
    region.setRegion(rgn)

p_roi.sigRangeChanged.connect(updateRegion)


p1.setData(np.arange(5000), np.random.normal(size=5000))

plotter_crosshair_enabled = True
vLine_p1 = pg.InfiniteLine(angle=90, movable=False)
hLine_p1 = pg.InfiniteLine(angle=0, movable=False)
label_p1 = pg.TextItem()
vLine_p_roi = pg.InfiniteLine(angle=90, movable=False)
hLine_p_roi = pg.InfiniteLine(angle=0, movable=False)
label_p_roi = pg.TextItem()


proxy = 0
proxy_roi = 0

def enable_plotter_crosshair():
    global proxy, proxy_roi
    if w_plot.actionCrosshair.isChecked() is True:
        # w_plot.actionCrosshair.setChecked(True)
        p1p.addItem(vLine_p1, ignoreBounds=True)
        p1p.addItem(hLine_p1, ignoreBounds=True)
        p1p.addItem(label_p1, ignoreBounds=True)
        p_roi.addItem(vLine_p_roi, ignoreBounds=True)
        p_roi.addItem(hLine_p_roi, ignoreBounds=True)
        p_roi.addItem(label_p_roi, ignoreBounds=True)
        proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved_p1)
        proxy_roi = pg.SignalProxy(p_roi.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved_p_roi)
    else:
        # w_plot.actionCrosshair.setChecked(False)
        p1p.removeItem(vLine_p1)
        p1p.removeItem(hLine_p1)
        p1p.removeItem(label_p1)
        p_roi.removeItem(vLine_p_roi)
        p_roi.removeItem(hLine_p_roi)
        p_roi.removeItem(label_p_roi)
        proxy = 0
        proxy_roi = 0
    
w_plot.actionCrosshair.triggered.connect(enable_plotter_crosshair)


current_df_row = 0
current_column_name_x = 0
current_column_name = 0
list_of_lines = []
list_of_labels = []

vb_p1 = p1p.vb
vb_p_roi = p_roi.vb

def mouseMoved_p1(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    data = p1.getData()
    if p1p.sceneBoundingRect().contains(pos):
        mousePoint = vb_p1.mapSceneToView(pos)
        x = mousePoint.x()
        y = mousePoint.y()
        if x < (max(data[0])+min(data[0]))/2:
            label_p1.setAnchor((0, 0))
        else:
            label_p1.setAnchor((1, 0))
        if x > 0 and x < max(data[0]):
            # label_p1.setText('cacca'+str(mousePoint.x()))
            label_p1.setHtml("<span style='font-size: 12pt'>x=%0.4g,  <span style='color: green'>y2=%0.4g</span>" % (x, y))
            label_p1.setPos(x, y)        
        vLine_p1.setPos(x)
        hLine_p1.setPos(y)
def mouseMoved_p_roi(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    data = p3.getData()
    if p_roi.sceneBoundingRect().contains(pos):
        mousePoint = vb_p_roi.mapSceneToView(pos)
        x = mousePoint.x()
        y = mousePoint.y()
        if x < (max(data[0])+min(data[0]))/2:
            label_p_roi.setAnchor((0, 0))
        else:
            label_p_roi.setAnchor((1, 0))
        if x > 0 and x < max(data[0]):
            # label_p1.setText('cacca'+str(mousePoint.x()))
            label_p_roi.setHtml("<span style='font-size: 12pt'>x=%0.4g,  <span style='color: green'>y2=%0.4g</span>" % (x, y))
            label_p_roi.setPos(x, y)        
        vLine_p_roi.setPos(x)
        hLine_p_roi.setPos(y)
enable_plotter_crosshair()

def plot_column():
    global df_columns, df_row, list_of_fit_datasets, p1, p2, current_df_row, current_column_name_x, current_column_name, list_of_lines, list_of_labels
    row = w_plot.listWidget_2.currentRow()
    x_row = w_plot.listWidget_3.currentRow()
    column_name = df_columns[row]
    column_name_x = df_columns[x_row]
    # print(df_row, current_df_row, column_name_x, current_column_name_x, column_name, current_column_name)
    if (column_name_x != current_column_name_x or column_name != current_column_name) and df_row == current_df_row:
        p1.clear()
        p1.setData(list_of_fit_datasets[df_row][column_name_x], list_of_fit_datasets[df_row][column_name])
        w_plot.plot.setLabel('left', column_name)
        w_plot.plot.setLabel('bottom', column_name_x)
        column_mean = list_of_fit_datasets[df_row][column_name].mean()
        column_stdDev = list_of_fit_datasets[df_row][column_name].std()
        current_column_name = column_name
        current_column_name_x = column_name_x
        current_df_row = df_row
        # print(df_row, current_df_row, column_name_x, current_column_name_x, column_name, current_column_name)
        if len(list_of_lines) > 0:
            for line, label in zip(list_of_lines, list_of_labels):
                w_plot.plot.removeItem(line)
                w_plot.plot.removeItem(label)
        if len(set(list_of_fit_datasets[df_row]['LTS_loop'])) > 1:
            print(len(set(list_of_fit_datasets[df_row]['LTS_loop'])))
            if column_name_x not in ['LTS_position', 'Delay_ps', 'ROI_angle']:
                #get indexes of list_of_fit_datasets[df_row]['LTS_position'] equals to its first value
                LTS = np.array(list_of_fit_datasets[df_row]['LTS_position'])
                indexes = np.where(LTS == LTS[0])[0]
                # print(indexes, LTS[0], LTS)
                #add a vertical line at positions of indexes
                data_span = list_of_fit_datasets[df_row][column_name].max() - list_of_fit_datasets[df_row][column_name].min()
                for i in indexes:
                    p2 = pg.InfiniteLine(pos=list_of_fit_datasets[df_row][column_name_x][i], angle=90, pen=(255, 0, 255, 255), movable=False)
                    list_of_lines.append(p2)
                    w_plot.plot.addItem(p2)
                    #add an index label at positions of indexes
                    p3 = pg.TextItem(text=f"{list_of_fit_datasets[df_row]['LTS_loop'][i]}", color=(255, 0, 255, 255), anchor=(0.0, 0.5))
                    p3.setPos(list_of_fit_datasets[df_row][column_name_x][i], list_of_fit_datasets[df_row][column_name].min()-data_span*0.1)
                    list_of_labels.append(p3)
                    w_plot.plot.addItem(p3)
                
    else:
        p1.clear()
        p1.setData(list_of_fit_datasets[df_row][current_column_name_x], list_of_fit_datasets[df_row][current_column_name])
        w_plot.plot.setLabel('left', current_column_name)
        w_plot.plot.setLabel('bottom', column_name_x)
        column_mean = list_of_fit_datasets[df_row][column_name].mean()
        column_stdDev = list_of_fit_datasets[df_row][column_name].std()
        current_df_row = df_row
    w_plot.info_lbl.setText(f"Mean: {column_mean:.3g}{new_line}StdDev: {column_stdDev:.3g}{new_line}StdDev%: {column_stdDev/column_mean*100:.3g}%")

    for child1 in params_tree.child('Trace'):
        if child1.name() == 'Line':
            data = child1.value()
            p1.setPen(data)
        if child1.name() == 'Symbol pen':
            data = child1.value()
            p1.setSymbolPen(data)
        if child1.name() == 'Symbol':
            for child2 in child1.children():
                if child2.name() == 'Shape':
                    data = child2.value()
                    p1.setSymbol(data)
                if child2.name() == 'Size':
                    data = child2.value()
                    p1.setSymbolSize(data)
                if child2.name() == 'Color':
                    data = child2.value()
                    p1.setSymbolBrush(data)


# print(PyQt5.QtWidgets.QStyleFactory.keys())


plot_params = [{'name': 'Trace', 'type': 'group', 'children': [{'name': 'Line', 'type': 'pen', 'value': pg.mkPen(None)}, {'name': 'Symbol', 'type': 'group', 'children': [{'name': 'Shape', 'type': 'list', 'values': [
    'o', 't', 't1', 't2', 't3', 's', 'p', 'h', 'star', '+', 'd', 'x']}, {'name': 'Size', 'type': 'float', 'value': 5}, {'name': 'Color', 'type': 'color', 'value': (255, 0, 0, 128)}]}, {'name': 'Symbol pen', 'type': 'pen', 'value': pg.mkPen(None)}]}]
params_tree = Parameter.create(name="params", type='group', children=plot_params)
w_plot.tree_params.setParameters(params_tree, showTop=False)


def tree_change(param, changes):
    for param, change, data in changes:
        if params_tree.childPath(param) == ['Trace', 'Line']:
            p1.setPen(data)
            p3.setPen(data)
        if params_tree.childPath(param) == ['Trace', 'Symbol pen']:
            p1.setSymbolPen(data)
            p3.setSymbolPen(data)
        if params_tree.childPath(param) == ['Trace', 'Symbol', 'Shape']:
            p1.setSymbol(data)
            p3.setSymbol(data)
        if params_tree.childPath(param) == ['Trace', 'Symbol', 'Size']:
            p1.setSymbolSize(data)
            p3.setSymbolSize(data)
        if params_tree.childPath(param) == ['Trace', 'Symbol', 'Color']:
            p1.setSymbolBrush(data)
            p3.setSymbolBrush(data)


params_tree.sigTreeStateChanged.connect(tree_change)

p_compare = 0
p_compare = w_plot.plot_compare
p_compare.addLegend()

test_index = 0

dict_of_compared_plots = {'name': [], 'plot': []}


def add_compare_plot():
    global df_columns, df_row, list_of_fit_datasets, p1, p2, p_compare, test_index, dict_of_compared_plots
    w_curve_name_dialog.hide()
    row = w_plot.listWidget_2.currentRow()
    x_row = w_plot.listWidget_3.currentRow()
    column_name = df_columns[row]
    column_name_x = df_columns[x_row]
    w_plot.plot_compare.setLabel('left', column_name)
    w_plot.plot_compare.setLabel('bottom', column_name_x)

    curve_name = w_curve_name_dialog.lineEdit.text()
    curve = p_compare.plot(list_of_fit_datasets[df_row][column_name_x], list_of_fit_datasets[df_row][column_name], name=curve_name)
    dict_of_compared_plots['name'].append(curve_name)
    dict_of_compared_plots['plot'].append(curve)
    test_index = test_index + 1
    for child1 in params_tree.child('Trace'):
        if child1.name() == 'Line':
            data = child1.value()
            curve.setPen(data)
        if child1.name() == 'Symbol pen':
            data = child1.value()
            curve.setSymbolPen(data)
        if child1.name() == 'Symbol':
            for child2 in child1.children():
                if child2.name() == 'Shape':
                    data = child2.value()
                    curve.setSymbol(data)
                if child2.name() == 'Size':
                    data = child2.value()
                    curve.setSymbolSize(data)
                if child2.name() == 'Color':
                    data = child2.value()
                    curve.setSymbolBrush(data)


def clear_compare_plot():
    global dict_of_compared_plots
    for plot in dict_of_compared_plots['plot']:
        p_compare.removeItem(plot)
    p_compare.clear()
    dict_of_compared_plots = {'name': [], 'plot': []}


def show_curve_name_dialog():
    w_curve_name_dialog.show()
    w_curve_name_dialog.lineEdit.setText(f'{test_index:03d}')


w_plot.actionAdd_to_compare.triggered.connect(show_curve_name_dialog)
w_curve_name_dialog.buttonBox.accepted.connect(add_compare_plot)
w_curve_name_dialog.buttonBox.rejected.connect(w_curve_name_dialog.hide)

w_plot.actionClear.triggered.connect(clear_compare_plot)


def export_df_to_clipboard():
    global df_columns, df_row, list_of_fit_datasets
    list_of_fit_datasets[df_row].to_clipboard(index=False)


def export_df_to_csv():
    global filename_opened, starting_path, df_row, list_of_fit_datasets
    fname = QtWidgets.QFileDialog.getSaveFileName(
        w, "Save dataset to CSV", starting_path+'/'+filename_opened, "CSV Files (*.csv)")
    list_of_fit_datasets[df_row].to_csv(fname[0], index=False)

def export_all_df_to_csv():
    global filename_opened, starting_path, df_row, list_of_fit_datasets, list_of_fit_datasets_names
    path = QtWidgets.QFileDialog.getExistingDirectory(w, "Select directory to save datasets", starting_path)
    for i, dataset in enumerate(list_of_fit_datasets):
        fname = path + '/' + list_of_fit_datasets_names[i] + '.csv'
        print(fname)
        dataset.to_csv(fname, index=False)


def import_df_from_csv():
    global list_of_fit_datasets, starting_path, list_of_fit_datasets_names
    fname = QtWidgets.QFileDialog.getOpenFileName(w_plot, "Select dataset to import", starting_path, "CSV Files (*.csv)")
    list_of_fit_datasets.append(pd.read_csv(fname[0]))
    w_plot.show()
    w_plot.listWidget.clear()
    list_of_fit_datasets_names.append(os.path.splitext(os.path.basename(fname[0]))[0])
    for i, obj in enumerate(list_of_fit_datasets):
        w_plot.listWidget.insertItem(i, f'{list_of_fit_datasets_names[i]}')


def import_df_from_pickle():
    global starting_path, img_arr, df
    fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select dataset to import", starting_path, "Pickle Files (*.pickle)")
    input_df = pd.read_pickle(fname[0], compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    img_arr = np.stack(input_df['images'])
    df = input_df.drop('images', axis=1)
    del input_df
    plot_metadata()
    scroll_data(0)


def import_df_from_pickle_ON_OFF():
    global starting_path, img_arr, df, imgON, imgOFF
    fname = QtWidgets.QFileDialog.getOpenFileName(w, "Select dataset to import", starting_path, "Pickle Files (*.pickle)")
    input_df = pd.read_pickle(fname[0], compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    imgON = np.stack(input_df['imagesON'])
    imgOFF = np.stack(input_df['imagesOFF'])
    img_arr = imgON
    df = input_df.drop('imagesON', axis=1)
    del input_df
    plot_metadata()
    scroll_data(0)

w_plot.listWidget.clicked.connect(list_columns)
w_plot.listWidget_2.clicked.connect(plot_column)
w_plot.listWidget_3.clicked.connect(plot_column)
w_plot.actionClipboard.triggered.connect(export_df_to_clipboard)
w_plot.actionCSV.triggered.connect(export_df_to_csv)
w_plot.actionImportCSV.triggered.connect(import_df_from_csv)
w_plot.actionExport_All.triggered.connect(export_all_df_to_csv)
w.actionImport_Dataset.triggered.connect(import_df_from_pickle)
w.actionImport_Dataset_ON_OFF.triggered.connect(import_df_from_pickle_ON_OFF)



def fit_data(V=False, fit=True):
    global fit_data_thread, fit_number, img_arr, roi, list_rois, list_rois_numbers
    fit_number = fit_number + 1
    # fit_data_thread = fit_data_thread_class(V, fit)
    if len(list_rois) == 0:
        with pg.ProgressDialog("Fitting ROI...", 0, img_arr.shape[0]) as dlg:
            dlg.setWindowTitle("Fitting ROI...")
            dlg.setFixedSize(400, 100)
            data_fitter(V, fit, dlg, roi)
        show_fits()
    else:
        with pg.ProgressDialog("Fitting ROI...", 0, img_arr.shape[0]*len(list_rois)) as dlg:
            dlg.setWindowTitle("Fitting ROI...")
            dlg.setFixedSize(400, 100)
            for i, ro in enumerate(list_rois):
                data_fitter(V, fit, dlg, ro)
                show_fits(i+1)
        w_plot.show()
        

    
list_rois = []
list_rois_numbers = []
list_rois_labels = []
def add_roi():
    global list_rois, list_rois_numbers, list_rois_labels
    font = QtGui.QFont()
    font.setPixelSize(14)
    
    w.roi_chkbx.setChecked(False)
    r = pg.ROI([0, 0], [200, 50], pen=pg.mkPen('m', width=2), hoverPen=pg.mkPen('m', width=4))
    r.setSize([img_arr.shape[2]*image_scale[0]*0.1, img_arr.shape[1]*image_scale[1]*0.1], update=True)
    r.addRotateHandle([0, 0], [0.5, 0.5])
    r.addScaleHandle([1, 1], [0, 0])
    r.setZValue(1e9)
    list_rois.append(r)
    list_rois_numbers.append(len(list_rois))
    #add text label to roi
    t = pg.TextItem(text=f'({len(list_rois)})', color=(255, 0, 255), anchor=(1.1, 1.1))
    t.setPos(r.pos())
    t.setFont(font)
    list_rois_labels.append(t)
    w.widget.addItem(r)
    w.widget.addItem(t)
    for ro in list_rois:
        ro.sigRegionChanged.connect(print_roi_label)
w.pushButton_2.clicked.connect(add_roi)

def remove_all_rois():
    global list_rois, list_rois_numbers, list_rois_labels
    for ro in list_rois:
        w.widget.removeItem(ro)
    for t in list_rois_labels:
        w.widget.removeItem(t)
    list_rois = []
    list_rois_numbers = []
    list_rois_labels = []
    
def remove_last_roi():
    global list_rois, list_rois_numbers, list_rois_labels
    w.widget.removeItem(list_rois[-1])
    w.widget.removeItem(list_rois_labels[-1])
    list_rois.pop()
    list_rois_numbers.pop()
    list_rois_labels.pop()
    
w.pushButton_3.clicked.connect(remove_last_roi)
w.pushButton_4.clicked.connect(remove_all_rois)

def print_roi_label():
    global list_rois, list_rois_numbers, list_rois_labels, image_scale
    for i, ro in enumerate(list_rois):
        list_rois_labels[i].setPos(ro.pos())



w.fit_gauss_btn.clicked.connect(partial(fit_data, False, True))
w.fit_voigt_btn.clicked.connect(partial(fit_data, True, True))
w.actionROI_avg_intensity.triggered.connect(partial(fit_data, True, False))

zero_order = [0, 0]


def set_center_of_image():
    global fit_dict, zero_order
    zero_order = [fit_dict['Center_OOP'], fit_dict['Center_IP']]
    print(zero_order)


def export_img_origin():
    global img, zero_order
    x0 = zero_order[0]
    y0 = zero_order[1]
    exporter = pg.exporters.ImageExporter(w.widget.imageItem)
    exporter.export('tmp.png')
    image = Image.open('tmp.png')
    #size of image
    width, height = image.size
    blank = Image.new('RGB', (2*width, 2*height), (0, 0, 0))
    output = io.BytesIO()
    image = image.convert('RGB')
    cx, cy = int((514-x0)*width/514), int((514-y0)*height/514)
    print(cx, cy)
    blank.paste(image, (cx, cy))
    blank.resize((1028, 1028), Image.ANTIALIAS)
    blank.save(output, "BMP")
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)
    # pd.DataFrame(img).to_clipboard(index=False, header=False)


w.actionSet_000_position.triggered.connect(set_center_of_image)


def export_arr_origin():
    global img


def update_colormap():
    global img, image_offset, image_scale
    if w.greyscale_chkbx.isChecked():
        w.widget.setColorMap(pg.colormap.get('CET-L1'))
        w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
    else:
        w.widget.setColorMap(pg.colormap.get('CET-R4'))
        w.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)


def subtract_first_img():
    global img_arr
    img_arr = -img_arr + img_arr[0]
    scroll_data(0)


w.export_img_btn.clicked.connect(export_img_origin)
w.actionSubtract_first_image.triggered.connect(subtract_first_img)


def export_Dataset():
    global img_arr, df
    images_df = df
    # images_df['images']=img_arr.tolist()
    images_df['images'] = list(img_arr)
    global filename_opened, starting_path
    fname = QtWidgets.QFileDialog.getSaveFileName(w, "Save dataset to pickle", starting_path+'/'+filename_opened, "Pickle Files (*.pickle)")
    images_df.to_pickle(fname[0], compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})


w.actionDataset.triggered.connect(export_Dataset)

def export_Dataset_ON_OFF():
    global img_arr, df, imgON, imgOFF
    images_df = df
    # images_df['images']=img_arr.tolist()
    images_df['imagesON'] = list(imgON)
    images_df['imagesOFF'] = list(imgOFF)
    global filename_opened, starting_path
    fname = QtWidgets.QFileDialog.getSaveFileName(w, "Save dataset to pickle", starting_path+'/'+filename_opened, "Pickle Files (*.pickle)")
    images_df.to_pickle(fname[0], compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})

w.actionDataset_ON_OFF.triggered.connect(export_Dataset_ON_OFF)

# ---------------------------------TIME RESOLVED-----------------------------------------

def group_data_by_delay():
    global df, img_arr, img_arr_2, imgON, imgOFF
    try:
        notes_df = df['notes']
    except Exception:
        pass
    
    delay_list = df['LTS_position'].to_list()

    unique_delays, indices = np.unique(delay_list, return_inverse=True)
    print('indices: ', indices)
    print('unique delays: ', unique_delays)
    # use the indices to group the images by delay and compute their mean
    grouped_images = np.empty((len(unique_delays),) + imgON.shape[1:])
    for i in range(len(unique_delays)):
        grouped_images[i] = np.sum(imgON[indices == i], axis=0, dtype=np.int64)
    img_arr = grouped_images
    imgON = img_arr
    ###
    # n_imgs = number of images in goup_images
    # df['n_shot'] = n_shot for one image * n_imgs
    try:
        grouped_images = np.empty((len(unique_delays),) + imgOFF.shape[1:])
        for i in range(len(unique_delays)):
            grouped_images[i] = np.sum(imgOFF[indices == i], axis=0, dtype=np.int64)
        img_arr_2 = 0
        imgOFF = grouped_images
    except Exception:
        print('No second image')
        
    numeric_cols = df.select_dtypes(include='number')
    df = numeric_cols.groupby(['LTS_position']).mean()
    df.reset_index(inplace=True)
    x['Delay_ps'] = df['LTS_position']*6.66
    # df['Time_for_humans']=str(dtm.fromtimestamp(df['Time']))
    try:
        df['notes'] = notes_df
    except Exception:
        pass
    scroll_data(0)
    plot_metadata()
        


def bin_data(bin):
    global df, img_arr
    try:
        notes_df = df['notes']
    except Exception:
        pass
    length = img_arr.shape[0]
    rep_arr = np.arange(int(length/bin)+1)
    # print(np.repeat(rep_arr,bin))
    df['bin'] = np.repeat(rep_arr, bin)[:length]

    groups, means = npi.group_by(df['bin'].to_numpy()).sum(img_arr)
    img_arr = means
    df = df.groupby(['bin']).mean()
    df.reset_index(inplace=True)
    try:
        df['notes'] = notes_df
    except Exception:
        pass

    df['Time_for_humans'] = [time.asctime(time.localtime(a)) for a in list(np.stack(df['Time'].astype(float)))]
    plot_metadata()
    scroll_data(0)


def slice_data():
    global img_arr, df
    start = w_slice_data.spin_start.value()
    end = w_slice_data.spin_end.value()
    img_arr = img_arr[start:end][:][:]
    df = df.iloc[start:end]
    df.reset_index(inplace=True)
    plot_metadata()
    scroll_data(0)


w_slice_data.buttonBox.accepted.connect(slice_data)


w.actionGroup_by_delay.triggered.connect(group_data_by_delay)
w.action2x2.triggered.connect(partial(bin_data, 2))
w.action3x3.triggered.connect(partial(bin_data, 3))
w.action5x5.triggered.connect(partial(bin_data, 5))
w.action10x10.triggered.connect(partial(bin_data, 10))
w.actionSlice_Data.triggered.connect(w_slice_data.show)

w_plot.actionTime_resolution.triggered.connect(w_timeres.show)


def convolution(x, mu, sigma, gamma, ampl, offset):
    if w_timeres.reverse_chkbx.isChecked():
        ampl = -ampl
    return offset+ampl*gamma/2*np.exp(gamma*(mu - x + gamma*sigma**2/2))*special.erfc((mu+gamma*sigma**2-x)/(np.sqrt(2)*sigma))


p2 = w_plot.plot.plot()


def apply_params_time_resolved():
    global p2, p1
    p0 = [float(w_timeres.mu_edit.text()), float(w_timeres.sigma_edit.text()), float(w_timeres.gamma_edit.text()), float(w_timeres.ampl_edit.text()), float(w_timeres.offset_edit.text())]
    p1_data = p1.getData()
    x_Data = p1_data[0]
    p2.clear()
    p2 = w_plot.plot.plot(x_Data, convolution(x_Data, *p0))
    p2.setPen((255, 255, 0))


curve_time_resolution = 0


def fit_time_resolution():
    global p2, p1, curve_time_resolution
    p0 = [float(w_timeres.mu_edit.text()), float(w_timeres.sigma_edit.text()), float(w_timeres.gamma_edit.text()), float(w_timeres.ampl_edit.text()), float(w_timeres.offset_edit.text())]
    low_bounds = [float(w_timeres.mu_low_edit.text()), float(w_timeres.sigma_low_edit.text()), float(
        w_timeres.gamma_low_edit.text()), float(w_timeres.ampl_low_edit.text()), (w_timeres.offset_low_edit.text())]
    high_bounds = [float(w_timeres.mu_high_edit.text()), float(w_timeres.sigma_high_edit.text()), float(
        w_timeres.gamma_high_edit.text()), float(w_timeres.ampl_high_edit.text()), float(w_timeres.offset_high_edit.text())]

    p1_data = p1.getData()
    x_Data = p1_data[0]
    y_Data = p1_data[1]
    try:
        w_timeres.label_10.setText("")
        popt, pcov = curve_fit(convolution, x_Data, y_Data, p0=p0, bounds=(low_bounds, high_bounds), maxfev=10000)
        p2.clear()
        p2.setData(x_Data, convolution(x_Data, *popt))
        w_timeres.ampl.setText(str(popt[3]))
        w_timeres.mu.setText(str(popt[0]))
        w_timeres.sigma.setText(str(popt[1]))
        w_timeres.gamma.setText(str(popt[2]))
        w_timeres.offset.setText(str(popt[4]))

    except Exception:
        w_timeres.label_10.setText("FIT ERROR!")


def remove_fit_time_resolution():
    global p2
    w_plot.plot.removeItem(p2)


w_timeres.pushButton.clicked.connect(apply_params_time_resolved)
w_timeres.pushButton_2.clicked.connect(fit_time_resolution)
w_timeres.pushButton_3.clicked.connect(remove_fit_time_resolution)


def on_minus_off():
    global img_arr, imgON, imgOFF
    img_arr = (imgON - imgOFF)
    scroll_data(0)
    
w.actionON_OFF.triggered.connect(on_minus_off)

def diff_on_off():
    global img_arr, imgON, imgOFF
    img_arr = (imgON - imgOFF)/imgOFF*np.mean(imgOFF, axis=0)+np.mean(imgOFF, axis=0)
    img_arr[np.isnan(img_arr)] = 0
    img_arr[img_arr == np.inf] = 0
    img_arr[img_arr == -np.inf] = 0
    scroll_data(0)
    
w.actionDiff_On_Off.triggered.connect(diff_on_off)


def show_on_off():
    global img_arr, imgON, imgOFF
    img_arr = (imgON/imgOFF)
    img_arr[np.isnan(img_arr)] = 0
    img_arr[img_arr == np.inf] = 0
    img_arr[img_arr == -np.inf] = 0
    scroll_data(0)

w.actionShow_On_Off.triggered.connect(show_on_off)

def show_on():
    global img_arr, imgON, imgOFF
    img_arr = imgON
    img_arr[np.isnan(img_arr)] = 0
    img_arr[img_arr == np.inf] = 0
    img_arr[img_arr == -np.inf] = 0
    scroll_data(0)

w.actionShow_On.triggered.connect(show_on)
from skimage import filters

def gaussian_smooth():
    global img_arr, imgON, imgOFF
    img_filtered = filters.gaussian(img_arr, sigma=1)
    img_arr = img_filtered
    scroll_data(0)

w.actionGaussian_smooth.triggered.connect(gaussian_smooth)

def gaussian_smooth_ON_OFF():
    global img_arr, imgON, imgOFF
    imgON_filtered = filters.gaussian(imgON, sigma=1)
    imgOFF_filtered = filters.gaussian(imgOFF, sigma=1)
    img_arr = imgON_filtered/imgOFF_filtered

w.actionGaussian_smooth_ON_OFF.triggered.connect(gaussian_smooth_ON_OFF)

def show_off():
    global img_arr, imgON, imgOFF
    img_arr = imgOFF
    img_arr[np.isnan(img_arr)] = 0
    img_arr[img_arr == np.inf] = 0
    img_arr[img_arr == -np.inf] = 0
    scroll_data(0)

w.actionShow_Off.triggered.connect(show_off)

def swap():
    global img_arr, img_arr_2
    text = w.label_7.text()
    if text == 'Pump ON':
        w.label_7.setText('Pump OFF')
    if text == 'Pump OFF':
        w.label_7.setText('Pump ON')
    img_arr, img_arr_2 = img_arr_2, img_arr
    
    
w.actionSwap_On_Off.triggered.connect(swap)


# -----------------------------------RADIAL AVERAGE----------------------------------------


imv_v = w.widget.getView()
line_top = pg.PlotCurveItem(x=[1, 200], y=[1, 200], pen=pg.mkPen('r', width=4))
line_bottom = pg.PlotCurveItem(x=[1, 200], y=[1, 200], pen=pg.mkPen('r', width=4))
imv_v.addItem(line_top, line_bottom)


w.widget.addItem(line_top)
w.widget.addItem(line_bottom)
line_top.hide()
line_bottom.hide()


def open_radial_average(angle=0):
    global line_roi1
    w_radial_average.show()
    global current_index
    w.fit_chkbx.setChecked(True)
    scroll_data(current_index)
    center = (fit_dict["Center_OOP"], fit_dict["Center_IP"])
    # line_bottom.setValue(center)
    angle = w_radial_average.angle_spinbox.value()/180*np.pi
    # angle_top = angle+180
    # angle_bottom = -(angle+180)
    if angle <= np.pi/2:
        line_top.setData(np.array([center[0], np.max([0, center[0]-center[1]/np.tan(angle)])]), np.array([center[1], np.max([0, center[1]-center[0]*np.tan(angle)])]))
        line_bottom.setData(np.array([center[0], np.max([0, center[0]+(514-center[1])/np.tan(-angle)])]), np.array([center[1], np.min([514, center[1]+center[0]*np.tan(angle)])]))
    if angle > np.pi/2:
        line_top.setData(np.array([center[0], np.min([514, center[0]+center[1]*np.tan(angle-np.pi/2)])]), np.array([center[1], np.max([0, center[1]-(514-center[0])*np.tan(np.pi-angle)])]))
        line_bottom.setData(np.array([center[0], np.min([514, center[0]+(514-center[1])*np.tan(angle-np.pi/2)])]), np.array([center[1], np.min([514, center[1]+(514-center[0])*np.tan(np.pi-angle)])]))

    # line_bottom.setData(np.array([center[0],center[0]-center[1]/np.tan(angle)]), np.array([center[1],0]))
    # line_top.setSpan((1,1))
    line_top.show()
    line_bottom.show()


def radial_close(event):
    global current_index
    line_top.hide()
    line_bottom.hide()
    w.fit_chkbx.setChecked(False)
    plot_metadata()
    scroll_data(current_index)


w.actionRadial_Average.triggered.connect(open_radial_average)
w_radial_average.angle_spinbox.valueChanged.connect(open_radial_average)
w_radial_average.closeEvent = radial_close


def radial_profile(data, center=(10, 256), phi_max=180):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    phi = np.abs(np.arctan2(y - center[1], -x + center[0])/(np.pi)*180)
    phi_mask = phi <= phi_max
    r = (r*phi_mask).astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = np.divide(tbin, nr, out=np.zeros_like(tbin), where=nr != 0)
    radialprofile.resize(514, refcheck=False)
    return radialprofile


radial_profiles_list = []
plot_radial = w_radial_average.radial_profile_plot.plot()


def calculate_radial_profiles():
    global img_arr, radial_profiles_list, img
    index = 0
    phi_max = w_radial_average.angle_spinbox.value()
    radial_profiles_list = []
    if w_radial_average.checkBox.isChecked():
        w.fit_chkbx.setChecked(True)
        center = (fit_dict["Center_OOP"], fit_dict["Center_IP"])
        radial_profiles_list.append(radial_profile(img, center, phi_max))
    else:
        for img1 in img_arr:
            w.fit_chkbx.setChecked(True)
            scroll_data(index)
            center = (fit_dict["Center_OOP"], fit_dict["Center_IP"])
            radial_profiles_list.append(radial_profile(img1, center, phi_max))
            index = index + 1
    scroll_profiles(0)


current_index = 0
exclusion = 0


def scroll_profiles(i):
    global radial_profiles_list, current_index, exclusion
    current_index = i
    w_radial_average.horizontalScrollBar.setMaximum(img_arr.shape[0] - 1)
    w_radial_average.label.setText("%d of %d" % (i + 1, img_arr.shape[0]))
    w.horizontalScrollBar.setValue(i)
    exclusion = w_radial_average.spinBox.value()
    x = np.arange(exclusion, len(radial_profiles_list[i]), 1)
    if w_radial_average.subtract_chkbx.isChecked():
        plot_radial.setData(x, radial_profiles_list[i][exclusion:]-radial_profiles_list[w_radial_average.subtract_spinbox.value()][exclusion:])
    else:
        plot_radial.setData(x, radial_profiles_list[i][exclusion:])


def update_exclusion(j):
    global radial_profiles_list, current_index
    plot_radial.setData(radial_profiles_list[current_index][j:])


w_radial_average.horizontalScrollBar.valueChanged.connect(scroll_profiles)
w_radial_average.spinBox.valueChanged.connect(update_exclusion)
w_radial_average.subtract_chkbx.toggled.connect(partial(scroll_profiles, current_index))

w_radial_average.calculate_btn.clicked.connect(calculate_radial_profiles)

# -----------------------------------------------------------------------------------------
imv_v = 0
scat_fcc = 0
scat_fcc_twin = 0
scat_hcp = 0
labels = []

labels_hcp = []
imv_v = w.widget.getView()
scat_fcc = pg.ScatterPlotItem(x=[], y=[])
scat_hcp = pg.ScatterPlotItem(x=[], y=[])
scat_fcc_twin = pg.ScatterPlotItem(x=[], y=[])
imv_v.addItem(scat_fcc)
imv_v.addItem(scat_hcp)
imv_v.addItem(scat_fcc_twin)


def show_pattern():
    global imv_v, scat_fcc, scat_fcc_twin, scat_hcp
    w_pat.show()


def hide_fcc():
    global imv_v, scat_fcc, labels
    for label in labels:
        w.widget.removeItem(label)
    scat_fcc.setData(x=[], y=[])


def hide_fcc_twin():
    global scat_fcc_twin
    scat_fcc_twin.setData(x=[], y=[])


def hide_hcp():
    global scat_hcp
    scat_hcp.setData(x=[], y=[])


def calc_fcc():
    global imv_v, scat_fcc, labels
    angle = float(w_pat.lineEdit_5.text())
    order_fcc = int(w_pat.lineEdit_9.text())
    xf, yf, peaks_fcc = calc_pattern_FCC(r=order_fcc)
    points_FCC = np.stack((xf, yf, np.zeros(len(xf))), axis=1)
    rot = rotation_matrix(angle * np.pi / 180, 0, 0, 1)
    rotated_points_FCC = rotate_list_vectors(list(points_FCC), rot)
    rotated_points_FCC = np.array(rotated_points_FCC)
    offx = float(w_pat.lineEdit.text())
    offy = float(w_pat.lineEdit_2.text())
    FX = float(w_pat.lineEdit_3.text())
    FY = float(w_pat.lineEdit_4.text())
    symbol_fcc = w_pat.lineEdit_7.text()
    size_fcc = float(w_pat.lineEdit_6.text())
    brush_fcc = eval(w_pat.lineEdit_8.text())
    x = rotated_points_FCC[:, 0] * FX + offx
    y = rotated_points_FCC[:, 1] * FY + offy
    x_n = x[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    y_n = y[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    peaks_fcc = [peaks_fcc[i] for i in np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))[0].tolist()]
    scat_fcc.setData(
        x=x_n,
        y=y_n,
        pen=pg.mkPen('w', width=1),
        symbol=symbol_fcc,
        size=size_fcc,
        brush=brush_fcc,
    )
    if len(labels) > 0:
        for label in labels:
            w.widget.removeItem(label)
    labels = []
    font = QFont()
    font.setPixelSize(16)
    font.setBold(True)
    for label in peaks_fcc:
        labels.append(
            pg.TextItem(text=label, color='k', fill=(255, 255, 0, 128)))
    for i, label in enumerate(labels):
        label.setFont(font)
        w.widget.addItem(label)
        label.setPos(x_n[i], y_n[i])


def calc_fcc_twin():
    global imv_v, scat_fcc_twin, labels
    order_fcc = int(w_pat.lineEdit_9.text())
    angle = float(w_pat.lineEdit_5.text())
    xf, yf, peaks_fcc = calc_pattern_FCC(r=order_fcc)
    points_FCC = np.stack((-xf, yf, np.zeros(len(xf))), axis=1)
    rot_twin = rotation_matrix(angle * np.pi / 180, 0, 0, 1)
    rotated_points_FCC_twin = rotate_list_vectors(list(points_FCC), rot_twin)
    rotated_points_FCC_twin = np.array(rotated_points_FCC_twin)
    offx = float(w_pat.lineEdit.text())
    offy = float(w_pat.lineEdit_2.text())
    FX = float(w_pat.lineEdit_3.text())
    FY = float(w_pat.lineEdit_4.text())
    symbol_fcc = w_pat.lineEdit_15.text()
    size_fcc = float(w_pat.lineEdit_14.text())
    brush_fcc = eval(w_pat.lineEdit_16.text())
    x = rotated_points_FCC_twin[:, 0] * FX + offx
    y = rotated_points_FCC_twin[:, 1] * FY + offy
    x_n = x[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    y_n = y[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    scat_fcc_twin.setData(
        x=x_n,
        y=y_n,
        pen=pg.mkPen('w', width=1),
        symbol=symbol_fcc,
        size=size_fcc,
        brush=brush_fcc,
    )


def calc_hcp():
    global imv_v, scat_hcp, labels_hcp
    angle = float(w_pat.lineEdit_5.text())
    order_hcp = int(w_pat.lineEdit_10.text())
    xf_h, yf_h, peaks_hcp = calc_pattern_HCP(r=order_hcp)
    points_HCP = np.stack((xf_h, yf_h, np.zeros(len(xf_h))), axis=1)
    rot = rotation_matrix(angle * np.pi / 180, 0, 0, 1)
    rotated_points_HCP = rotate_list_vectors(list(points_HCP), rot)
    rotated_points_HCP = np.array(rotated_points_HCP)
    offx = float(w_pat.lineEdit.text())
    offy = float(w_pat.lineEdit_2.text())
    FX = float(w_pat.lineEdit_3.text())
    FY = float(w_pat.lineEdit_4.text())
    symbol_hcp = w_pat.lineEdit_17.text()
    size_hcp = float(w_pat.lineEdit_18.text())
    brush_hcp = eval(w_pat.lineEdit_19.text())
    x = rotated_points_HCP[:, 0] * FX + offx
    y = rotated_points_HCP[:, 1] * FY + offy
    x_n = x[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    y_n = y[np.where(np.logical_and(np.logical_and(x >= 0, x <= 514), np.logical_and(y >= 0, y <= 514)))]
    scat_hcp.setData(
        x=x_n,
        y=y_n,
        pen=pg.mkPen('w', width=1),
        symbol=symbol_hcp,
        size=size_hcp,
        brush=brush_hcp,
    )


# w.FCC_btn.clicked.connect(show_pattern)
w_pat.pushButton.clicked.connect(calc_fcc)
w_pat.pushButton_5.clicked.connect(calc_hcp)
w_pat.pushButton_3.clicked.connect(calc_fcc_twin)
w_pat.pushButton_2.clicked.connect(hide_fcc)
w_pat.pushButton_4.clicked.connect(hide_fcc_twin)
w_pat.pushButton_6.clicked.connect(hide_hcp)


#----------------------------------------- Simulator ------------------------------------------

w.actionPattern_simulator.triggered.connect(w_simulator.show)

w_simulator.widget.setColorMap(pg.colormap.get('CET-R4'))
w_simulator.widget.setImage(img, autoRange=False, autoLevels=False, autoHistogramRange=False)

w_simulator.widget.ui.roiBtn.hide()
w_simulator.widget.ui.menuBtn.hide()



@vectorize([float64(float64, float64, float64, float64, float64, float64)])
def dot_product(a1, a2, a3, b1, b2, b3):
    return np.float64(a1 * b1 + a2 * b2 + a3 * b3)


def map_q(detector, k_i):
    matrix = detector - k_i
    return matrix


tt_status = 0



@njit(parallel=True)
def calc_intensity_2(detector_complex, struct, detector, q_map, status, k=np.array([89, 0, 0]), p=0):
    global tt_status
    detector_complex[:, :] = 0
    # progress = np.zeros(detector.shape[1])
    for tt in prange(detector.shape[1]):
        for ee in range(detector.shape[0]):
            S_s = 0
            S_c = 0
            q = q_map[ee, tt]+(random.random()*2-1)*k*p
            # f = f_map[ee, tt]
            for part in struct:
                Q = (dot_product(q[0], q[1], q[2], part[0], part[1], part[2]))
                cos = math.cos(Q)
                sin = math.sin(Q)
                S_s = S_s + sin
                S_c = S_c + cos
            A = math.sqrt(S_s*S_s + S_c*S_c)
            detector_complex[ee, tt] = A
        # progress[tt] = 1
        status.update(1)


@njit(parallel=True)
def q_setup(detector, th, et, k):
    k0 = k[0]
    for ee in prange(detector.shape[0]):
        for tt in range(detector.shape[1]):
            t = th[tt]*np.pi/180
            e = et[ee]*np.pi/180

            k_fx = k0 * np.cos(t) * np.cos(e)
            k_fy = k0 * np.sin(t)
            k_fz = k0 * np.cos(t) * np.sin(e)
            detector[ee, tt] = np.array([k_fx, k_fy, k_fz])
            # f_map[ee, tt] = np.linalg.norm(detector[ee, tt]-k)


def calc_form_factors(element, q):
    form_factors_df = pd.read_csv('./formfactors.csv')
    element = 'C'
    row = form_factors_df[form_factors_df.values == element]
    a1 = row['a1'].to_numpy()
    b1 = row['b1'].to_numpy()
    a2 = row['a2'].to_numpy()
    b2 = row['b2'].to_numpy()
    a3 = row['a3'].to_numpy()
    b3 = row['b3'].to_numpy()
    c = row['c'].to_numpy()
    f_X = c + a1*np.exp(-b1*(q/4/np.pi)**2) + a2*np.exp(-b2*(q/4/np.pi)**2) + a3*np.exp(-b3*(q/4/np.pi)**2)
    f_e = 0.2393*(12-f_X)/q**2
    return f_e


def import_struct(filename):
    df_struct = pd.read_csv(filename, sep="\s+", header=2, names=["atom", 'x', 'y', 'z'])
    result = df_struct[['x', 'y', 'z']].to_numpy()
    atom_type = df_struct[['atom']].to_numpy()
    return result, atom_type


def lambda_e2(E):
    #https://virtuelle-experimente.de/en/elektronenbeugung/wellenlaenge/de-broglie-relativistisch.php
    q = 1.60217662e-19  # C
    E = E * q  # J
    h = 6.62607004e-34  # m^2 kg / s
    m_e = 9.10938356e-31  # kg
    c = 299792458  # m/s
    b = h*c
    a = 2*m_e*c**2
    return b/np.sqrt(a*E + E**2)*1e10


def inv_lambda_e2(L):
    q = 1.60217662e-19  # C
    L = L * 1e-10  # m
    h = 6.62607004e-34  # m^2 kg / s
    m_e = 9.10938356e-31  # kg
    c = 299792458  # m/s
    b = h*c
    a = 2*m_e*c**2
    return 0.5*(-a+np.sqrt(a**2+4*b**2/L**2))/q


def set_wavelength():
    E = w_simulator.doubleSpinBox_13.value()*1000
    if E > 0:
        w_simulator.doubleSpinBox_14.setValue(lambda_e2(E))


def set_energy():
    L = w_simulator.doubleSpinBox_14.value()
    if L > 0:
        w_simulator.doubleSpinBox_13.setValue(inv_lambda_e2(L)/1000)


set_wavelength()

w_simulator.doubleSpinBox_13.editingFinished.connect(set_wavelength)
w_simulator.doubleSpinBox_14.editingFinished.connect(set_energy)


def calc_detector_resolution():
    L = w_simulator.doubleSpinBox_14.value()
    x, y = w_simulator.spinBox.value(), w_simulator.spinBox_2.value()
    angle_e = (w_simulator.doubleSpinBox_2.value() - w_simulator.doubleSpinBox.value())
    angle_t = (w_simulator.doubleSpinBox_4.value() - w_simulator.doubleSpinBox_3.value())
    w_simulator.label_20.setText(f"{angle_e/x:.4g}°/px")
    w_simulator.label_21.setText(f"{angle_t/y:.4g}°/px")
    w_simulator.label_18.setText(f"{angle_t/y:.4g}°/px")
    cx = 2*np.sin(angle_e/2/180*np.pi)/L
    cy = 2*np.sin(angle_t/2/180*np.pi)/L
    w_simulator.label_23.setText(f"{cx/x:.4g}Å⁻¹/px")
    w_simulator.label_24.setText(f"{cy/y:.4g}Å⁻¹/px")
    # w_simulator.label_24.setText(f"{cy:.4g}Å⁻¹")


w_simulator.doubleSpinBox.editingFinished.connect(calc_detector_resolution)
w_simulator.doubleSpinBox_2.editingFinished.connect(calc_detector_resolution)
w_simulator.doubleSpinBox_3.editingFinished.connect(calc_detector_resolution)
w_simulator.doubleSpinBox_4.editingFinished.connect(calc_detector_resolution)
w_simulator.spinBox.editingFinished.connect(calc_detector_resolution)
w_simulator.spinBox_2.editingFinished.connect(calc_detector_resolution)

calc_detector_resolution()

simulated_images = []
gB_state = 0
xyz_file = ''

f = StringIO()
list_theta = []
simulation_metadata = []


class simulation_thread_class(QThread):
    started = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)

    def run(self):
        global simulated_images, gB_state, xyz_file, f, list_theta, simulation_metadata
        self.det_x = w_simulator.spinBox.value()
        self.det_y = w_simulator.spinBox_2.value()
        self.th = np.linspace(w_simulator.doubleSpinBox.value(), w_simulator.doubleSpinBox_2.value(), self.det_x)
        self.et = np.linspace(w_simulator.doubleSpinBox_3.value(), w_simulator.doubleSpinBox_4.value(), self.det_y)
        self.lambda_electron = w_simulator.doubleSpinBox_14.value()/10
        self.k = 2*np.pi*np.array([1, 0, 0])/self.lambda_electron
        self.delta = w_simulator.doubleSpinBox_15.value()/100
        self.detector_x, self.detector_y = np.meshgrid(self.th, self.et)
        self.detector = np.zeros(self.det_y * self.det_x * 3).reshape(self.det_y, self.det_x, 3)
        self.detector_complex = np.zeros(self.det_y * self.det_x).reshape(self.det_y, self.det_x)
        # self.f_map = np.zeros(self.det_y * self.det_x).reshape(self.det_y, self.det_x)
        q_setup(self.detector, self.th, self.et, self.k)
        # w_simulator.widget.setImage(self.f_map, autoRange=False, autoLevels=False, autoHistogramRange=False)
        self.struct, self.atoms = import_struct(xyz_file)
        self.strain_x = w_simulator.doubleSpinBox_9.value()
        self.strain_y = w_simulator.doubleSpinBox_10.value()
        self.strain_z = w_simulator.doubleSpinBox_11.value()
        self.strain = np.array([self.strain_x, self.strain_y, self.strain_z])
        self.struct = self.struct * self.strain
        self.rot = rotation_matrix(30*np.pi/180., 0, 0, 1)
        self.q_map = map_q(self.detector, self.k)
        # self.f_map_C = calc_form_factors('C', self.f_map)
        if gB_state == 0:
            self.list_theta = [w_simulator.doubleSpinBox_5.value()]
        else:
            self.list_theta = list((np.arange(w_simulator.doubleSpinBox_6.value(), w_simulator.doubleSpinBox_7.value(), w_simulator.doubleSpinBox_8.value())))

        self.n_twins = w_simulator.spinBox_3.value()
        self.twins_angle = w_simulator.doubleSpinBox_12.value()
        self.list_twins = list(np.arange(0, self.n_twins, 1)*self.twins_angle*np.pi/180)
        t0 = time.time()
        # self.metadata_dataframe = pd.DataFrame()
        with ProgressBar(total=self.detector.shape[1]*len(self.list_theta)*self.n_twins, ncols=10, file=f, leave=True) as numba_progress:
            self.started.emit(1)
            self.simulated_twins = []
            self.simulated_infos = pd.DataFrame()
            for self.t, self.twin in enumerate(self.list_twins):
                self.simulated_twin = 0
                self.rocking_curve = []
                for self.theta in self.list_theta:
                    self.rot = rotation_matrix(self.twin, 0, 0, 1)
                    self.new_struct = np.array(rotate_list_vectors(self.struct, self.rot))/10
                    self.rot = rotation_matrix(-self.theta*np.pi/180., 0, 1, 0)
                    self.new_struct = np.array(rotate_list_vectors(self.new_struct, self.rot))
                    # print('starting')

                    calc_intensity_2(self.detector_complex, self.new_struct, self.detector, self.q_map, numba_progress, k=self.k, p=self.delta)
                    # print(f.getvalue())
                    # print('finished')
                    self.rocking_curve.append(self.detector_complex.copy())
                    self.infos = {
                        'Energy': w_simulator.doubleSpinBox_13.value(),
                        'Wavelength': w_simulator.doubleSpinBox_14.value(),
                        'Delta': w_simulator.doubleSpinBox_15.value(),
                        'Detector': [w_simulator.spinBox.value(), w_simulator.spinBox_2.value()],
                        'Eta range': [w_simulator.doubleSpinBox.value(), w_simulator.doubleSpinBox_2.value()],
                        '2Theta range': [w_simulator.doubleSpinBox_3.value(), w_simulator.doubleSpinBox_4.value()],
                        'Calibration': [w_simulator.label_23.text(), w_simulator.label_24.text()],
                        'Strain': [w_simulator.doubleSpinBox_9.value(), w_simulator.doubleSpinBox_10.value(), w_simulator.doubleSpinBox_11.value()],
                        'Twinning': [w_simulator.spinBox_3.value(), w_simulator.doubleSpinBox_12.value()],
                        'Single': w_simulator.groupBox_2.isChecked(),
                        'Theta single': w_simulator.doubleSpinBox_5.value(),
                        'Rocking curve': w_simulator.groupBox_3.isChecked(),
                        'Angle type': [a*b for a, b in zip(['Eta', 'Phi', 'Theta'], [w_simulator.radioButton.isChecked(), w_simulator.radioButton_2.isChecked(), w_simulator.radioButton_3.isChecked()])],
                        'Rocking curve range': [w_simulator.doubleSpinBox_6.value(), w_simulator.doubleSpinBox_7.value(), w_simulator.doubleSpinBox_8.value()],
                        'Angle': self.theta,
                    }
                    self.df_infos = pd.DataFrame([self.infos], index=[0])
                    if self.t == 0:
                        #concatenate the infos with itself
                        self.simulated_infos = pd.concat([self.simulated_infos, self.df_infos], ignore_index=True)
                        # self.simulated_infos = self.simulated_infos.append(self.infos, ignore_index=True)
                self.rocking_curve = np.array(self.rocking_curve)
                self.simulated_twins.append(self.rocking_curve)
            self.simulated_twins = np.array(self.simulated_twins)
        self.simulated_twins = np.sum(self.simulated_twins, axis=0)
        list_theta = self.list_theta
        w_simulator.widget.setImage(self.simulated_twins[0], autoRange=False, autoLevels=True, autoHistogramRange=True)
        w_simulator.widget.setLevels(0, 1)
        w_simulator.widget.setHistogramRange(0, 1)
        simulated_images = self.simulated_twins
        simulation_metadata = self.simulated_infos
        w_simulator.horizontalScrollBar.setMaximum(simulated_images.shape[0] - 1)
        time.sleep(1)
        self.finished.emit(1)
        t1 = time.time()
        print(t1-t0)


simulation_thread = simulation_thread_class()


def read_f():
    global f
    val = str(copy.copy(f.getvalue()))
    # p = val[-300:-1].split('%|')
    s = val[-120:-1].strip().replace('\r', '')
    p1 = int(s[s.find('| ')+1:s.rfind('.0/')])
    p2 = int(s[s.find('/')+1:s.rfind('[')])
    # p = s[s.find(']')+1:s.rfind('%|')]

    # print('p1=',p1,'\np2=',p2)
    # print(s)

    w_simulator.progressBar.setFormat("%.02f%%" % (p1/p2*100))
    w_simulator.progressBar.setValue(int(p1/p2*100))


timer_th = QtCore.QTimer()
timer_th.timeout.connect(read_f)
# timer_th.start(250)
simulation_thread.started.connect(partial(timer_th.start, 100))
simulation_thread.finished.connect(timer_th.stop)


def start_simulation():
    global tt_status
    tt_status = 0
    simulation_thread.start()


def stop_simulation():
    global tt_status, simulation_thread
    tt_status = 1
    print(tt_status)
    time.sleep(1)
    simulation_thread.exit()
    tt_status = 0
    # simulation_thread.wait()
    # timer_th.stop()


def scroll_simulated_images(value):
    w_simulator.widget.setImage(simulated_images[value], autoRange=False, autoLevels=False, autoHistogramRange=False)
    w_simulator.label_8.setText(str(value))


w_simulator.horizontalScrollBar.valueChanged.connect(scroll_simulated_images)

w_simulator.pushButton_2.clicked.connect(start_simulation)
w_simulator.pushButton_3.clicked.connect(start_simulation)
w_simulator.pushButton_6.clicked.connect(stop_simulation)
w_simulator.pushButton_5.clicked.connect(stop_simulation)


def activate_single_rocking():
    global gB_state
    if gB_state == 0:
        w_simulator.groupBox_2.setChecked(0)
        w_simulator.groupBox_3.setChecked(1)
        gB_state = 1
    else:
        w_simulator.groupBox_2.setChecked(1)
        w_simulator.groupBox_3.setChecked(0)
        gB_state = 0


def rotate_cw90():
    global simulated_images
    simulated_images = np.rot90(simulated_images, -1, (1, 2))
    w_simulator.horizontalScrollBar.setMaximum(simulated_images.shape[0] - 1)
    w_simulator.widget.setImage(simulated_images[0], autoRange=False, autoLevels=False, autoHistogramRange=False)


def rotate_ccw90():
    global simulated_images
    simulated_images = np.rot90(simulated_images, 1, (1, 2))
    w_simulator.horizontalScrollBar.setMaximum(simulated_images.shape[0] - 1)
    w_simulator.widget.setImage(simulated_images[0], autoRange=False, autoLevels=False, autoHistogramRange=False)


w_simulator.groupBox_2.clicked.connect(activate_single_rocking)
w_simulator.groupBox_3.clicked.connect(activate_single_rocking)
w_simulator.pushButton_7.clicked.connect(rotate_cw90)
w_simulator.pushButton_8.clicked.connect(rotate_ccw90)


def open_xyz_file():
    global starting_path, xyz_file
    fname = QtWidgets.QFileDialog.getOpenFileName(w_simulator, "Select XYZ fil", starting_path, "XYZ Files (*.xyz)")
    xyz_file = fname[0]
    w_simulator.label.setText(os.path.split(fname[0])[1])


w_simulator.pushButton.clicked.connect(open_xyz_file)


def send_sim_to_main():
    global simulated_images, img_arr, df, list_theta, simulation_metadata
    img_arr = simulated_images
    # df = pd.DataFrame()
    # df["theta"] = np.array(list_theta)
    # df['2theta'] = np.array(list_theta)*2
    df = simulation_metadata
    scroll_data(0)


w_simulator.pushButton_4.clicked.connect(send_sim_to_main)

#----------------------------------------------------------------------------------------------


w.actionPlotter.triggered.connect(w_plot.show)

#-------------------------------------CALIBRATION----------------------------------------


def set_offset():
    global image_offset, zero_order, roi, current_frame, fit_dict, image_scale
    w_calibration.doubleSpinBox_2.setValue(-fit_dict['Center_OOP'])
    w_calibration.doubleSpinBox_3.setValue(-fit_dict['Center_IP'])
    zero_order = [fit_dict['Center_OOP'], fit_dict['Center_IP']]
    roi.setPos([0, 0], update=True)
    roi.setSize([img_arr.shape[2]*image_scale[0]*0.1, img_arr.shape[1]*image_scale[1]*0.1], update=True)
    scroll_data(current_frame)


second_point_calibration = [0, 0]
calibration = 0
is_calibrated = 0


def set_second_point():
    global second_point_calibration
    second_point_calibration = [fit_dict['Center_OOP'], fit_dict['Center_IP']]
    # print(second_point_calibration[0]-w_calibration.doubleSpinBox_2.value(), second_point_calibration[1]-w_calibration.doubleSpinBox_3.value())


def set_scale(value):
    global image_scale, image_offset, second_point_calibration
    off = [-w_calibration.doubleSpinBox_2.value(), -w_calibration.doubleSpinBox_3.value()]
    r = np.linalg.norm(np.array(off) - np.array(second_point_calibration))
    w_calibration.doubleSpinBox.setValue(value/r)


def set_calibration():
    global image_scale, calibration, roi, image_offset, zero_order, current_frame, img, is_calibrated, baseplot
    calibration = w_calibration.doubleSpinBox.value()
    image_offset = [w_calibration.doubleSpinBox_2.value()*calibration, w_calibration.doubleSpinBox_3.value()*calibration]
    zero_order = [zero_order[0]*calibration, zero_order[1]*calibration]
    image_scale = [calibration, calibration]
    w.fit_chkbx.setChecked(False)
    roi.setSize([0.1*img.shape[0]*image_scale[0], 0.1*img.shape[1]*image_scale[1]], update=True)
    roi.setPos([-roi.size().x()/2, -roi.size().y()/2], update=True)
    w.widget.setImage(img, autoRange=True, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
    roi_ruler.setSize([0.1*img.shape[0]*image_scale[0], 1e-6], update=True)
    roi_ruler.setPos([0, 0], update=True)
    w.widget.setImage(img, autoRange=True, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
    print_ruler_size()
    baseplot.setXRange(image_offset[0], (image_offset[0]+img.shape[0]*image_scale[0]))
    baseplot.setYRange(image_offset[1], (image_offset[1]+img.shape[1]*image_scale[1]))
    roi_update_plot()

    is_calibrated = 1
    print(is_calibrated)


def reset_calibration():
    global image_scale, image_offset, zero_order, calibration, current_frame, is_calibrated, img
    image_offset = [0, 0]
    image_scale = [1, 1]
    zero_order = [0, 0]
    calibration = 0
    w_calibration.doubleSpinBox.setValue(calibration)
    is_calibrated = 0
    w.widget.setImage(img, autoRange=True, autoLevels=False, autoHistogramRange=False, pos=image_offset, scale=image_scale)
    baseplot.setXRange(image_offset[0], (image_offset[0]+img.shape[0]*image_scale[0]))
    baseplot.setYRange(image_offset[1], (image_offset[1]+img.shape[1]*image_scale[1]))
    roi.setSize([0.1*img.shape[0]*image_scale[0], 0.1*img.shape[1]*image_scale[1]], update=True)
    roi.setPos([0, 0], update=True)
    roi_ruler.setSize([0.1*img.shape[0]*image_scale[0], 1e-6], update=True)
    roi_ruler.setPos([0, 0], update=True)
    roi_update_plot()


def button_box_output(btn):
    if (btn.text()) == 'Restore Defaults':
        reset_calibration()
    if (btn.text()) == 'Apply':
        set_calibration()


w.actionCalibrate_image.triggered.connect(w_calibration.show)
w.actionCalibrate_image.triggered.connect(partial(w.fit_chkbx.setChecked, True))
w.actionCalibrate_image.triggered.connect(roi_update_plot)
w_calibration.pushButton.clicked.connect(set_offset)
w_calibration.pushButton_2.clicked.connect(set_second_point)
w_calibration.buttonBox.clicked.connect(button_box_output)
w_calibration.buttonBox.accepted.connect(w_calibration.hide)
w_calibration.doubleSpinBox_4.valueChanged.connect(set_scale)


#-----------------------------------------------------------------------------------------
#--------------------------------------MEASURE----------------------------------------------
roi_ruler = pg.ROI([0, 0], [200, 1e-6], pen=pg.mkPen('m', width=4), hoverPen=pg.mkPen('m', width=4), movable=False)
roi_ruler.addScaleRotateHandle([0, 0.5], [1, 0.5])
roi_ruler.addScaleRotateHandle([1, 0.5], [0, 0.5])
roi_ruler.addTranslateHandle([0.5, 0.5], [0.5, 0.5])
roi_ruler.setZValue(2e9)
label_distance = pg.TextItem(text='', color=(0, 0, 0), anchor=(0.5, 0.0), angle=0, fill=(255, 255, 255, 128))


def add_roi_ruler():
    baseplot.addItem(roi_ruler)
    roi_ruler.setPos([0, 0], update=True)
    print_ruler_size()
    w.widget.addItem(label_distance)


def remove_roi_ruler():
    baseplot.removeItem(roi_ruler)
    baseplot.removeItem(label_distance)


def measure_check():
    if w.actionMeasure_distance.isChecked():
        add_roi_ruler()
    else:
        remove_roi_ruler()

# w.widget.addItem(label_distance)


def print_ruler_size():
    global roi_ruler, img, is_calibrated, calibration, image_scale, image_offset
    p1 = np.array([roi_ruler.getSceneHandlePositions()[1][1].x(), roi_ruler.getSceneHandlePositions()[1][1].y()])
    p2 = np.array([roi_ruler.getSceneHandlePositions()[0][1].x(), roi_ruler.getSceneHandlePositions()[0][1].y()])
    p1 = QtCore.QPointF(p1[0], p1[1])
    p1 = np.array([w.widget.getImageItem().mapFromScene(p1).x(), w.widget.getImageItem().mapFromScene(p1).y()])*image_scale+image_offset
    p2 = QtCore.QPointF(p2[0], p2[1])
    p2 = np.array([w.widget.getImageItem().mapFromScene(p2).x(), w.widget.getImageItem().mapFromScene(p2).y()])*image_scale+image_offset
    r = roi_ruler.size()[0]
    angle = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    if angle >= np.pi/2:
        angle = angle - np.pi
    if angle <= -np.pi/2:
        angle = angle + np.pi
    label_distance.setPos(((p1+p2)/2)[0], ((p1+p2)/2)[1])
    if is_calibrated:
        label_distance.setText(f'{r:.4g} Å⁻¹, {angle*180/np.pi:.2f}°')
        w.label.setText(f'{r:.4g} Å⁻¹,{1/r:.4g} Å, {angle*180/np.pi:.2f}°')
    else:
        label_distance.setText(f'{r:.2f} px, {angle*180/np.pi:.2f}°')
        w.label.setText(f'{r:.2f} px\n{r*0.075:.2f} mm\n{angle*180/np.pi:.2f}°')
    label_distance.setAngle(-angle*180/np.pi)


w.actionMeasure_distance.triggered.connect(measure_check)
roi_ruler.sigRegionChanged.connect(print_ruler_size)

#------------------------------------------------------------------------------------------

#--------------------------------------3D view----------------------------------------------
w_3dView = gl.GLViewWidget()
w_3dView.setWindowTitle('3D view of experiment')
grid = gl.GLGridItem()
grid.scale(1, 1, 1)
w_3dView.addItem(grid)
w_3dView.opts['distance'] = 2000
w_3dView.opts['fov'] = 1

w_simulator.pushButton_9.clicked.connect(w_3dView.show)

verts_detector = np.array([
    [0, 0, 0],
    [0, 10, 0],
    [0, 10, 10],
    [0, 0, 10],
])
faces_detector = np.array([
    [0, 1, 2],
    [0, 2, 3],

])
colors_detector = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],

])

## Mesh item will automatically compute face normals.
m1 = gl.GLMeshItem(vertexes=verts_detector, faces=faces_detector, faceColors=colors_detector, smooth=False)
m1.translate(10, -5, -5)
# m1.setGLOptions('additive')
w_3dView.addItem(m1)


cyl = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 0.1], length=100)
beam = gl.GLMeshItem(meshdata=cyl, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque', color=(1, 1, 0, 1))
beam.rotate(-90, 0, 1, 0)
beam.translate(10, 0, 0)
w_3dView.addItem(beam)

arrow_count = [1, 2, 3, 4, 5, 6]
arrow_mesh = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 0.4], length=2)
for arrow in arrow_count:
    arr_m = gl.GLMeshItem(meshdata=arrow_mesh, smooth=True, drawEdges=False, shader='shaded', glOptions='opaque', color=(1, 1, 0, 1))
    arr_m.rotate(-90, 0, 1, 0)
    arr_m.translate(-arrow*10, 0, 0)
    w_3dView.addItem(arr_m)


def draw_sample():
    global xyz_file
    scale = 0.1
    #read file CPK_colors.csv and create pandas dataframe
    df = pd.read_csv('CPK_colors.csv')
    array, atoms = import_struct(xyz_file)
    array = array*scale
    md = gl.MeshData.sphere(rows=10, cols=20)
    list_of_atoms = []
    for vec, atom in zip(array, atoms):
        #find color of atom

        color = df.loc[df['Element'] == atom[0], 'RGB Color'].values[0][1:-1].split(',')
        color = [int(i)/255 for i in color]
        color.append(1)
        #convert list to tuple
        color = tuple(color)
        radius = df.loc[df['Element'] == atom[0], 'Radius'].values[0]
        radius = float(radius)
        m = gl.GLMeshItem(meshdata=md, smooth=True, color=color, shader='shaded', glOptions='opaque')
        m.scale(scale*radius, scale*radius, scale*radius)
        m.translate(vec[0], vec[1], vec[2])
        list_of_atoms.append(m)
        w_3dView.addItem(m)


w_simulator.pushButton_9.clicked.connect(draw_sample)


#------------------------------------------------------------------------------------------


def prog_close(event):
    global loading_thread, fit_data_thread
    try:
        loading_thread.terminate()
    except Exception:
        pass
    try:
        fit_data_thread.terminate()
    except Exception:
        pass


w_prog.closeEvent = prog_close


def on_close(event):
    app.closeAllWindows()


w.closeEvent = on_close

app.exec_()
w.close()
