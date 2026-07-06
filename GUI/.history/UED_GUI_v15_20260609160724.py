#Needs to be run on python 3.12.8 !!!!!!!

import sys
sys.path.append("c:/Users/mainUED/Documents/GitHub/UED/GUI/")
import os
from ctypes import * 
from ctypes import cdll
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(BASE_DIR, "DLLs", "gladz", "GlazLib.dll")
print("Loading DLL from:", DLL_PATH)
glaz = cdll.LoadLibrary(DLL_PATH)
#glaz = cdll.LoadLibrary("./DLLs/gladz/GlazLib.dll")  #noqa F405
simulator = 0
from calendar import c
from hashlib import new
import io
import json
from operator import ne
import sys
import time
from datetime import datetime, date
from datetime import datetime as dtm
from datetime import timedelta
from functools import partial
from inspect import modulesbyfile
from threading import Event, Thread, Timer
from time import mktime
from tkinter import font
from urllib.request import urlopen
import pandas as pd
import copy
import h5py
import hdf5plugin
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import requests
import serial
import tifffile
from lmfit.models import (ConstantModel, GaussianModel, LinearModel, LorentzianModel, VoigtModel, QuadraticModel)
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from PIL import Image, ImageTk

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import (QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal)
from pyqtgraph import AxisItem
from pyqtgraph.dockarea import * #noqa F403
from pyqtgraph.Qt import QtCore, QtGui
from scipy import ndimage
from simple_pid import PID
if simulator == 0:
    from DEigerClient3 import DEigerClient
else:
    from DEigerClient3_dummy import DEigerClient
from A7_pulser_v2 import A7_pulser
import pyvisa
from PyQt6.QtWidgets import * #noqa F403
from PyQt6.QtGui import * #noqa F403
# gdagfgvedrf
from thorlabs_mc2000b import MC2000B, Blade

from PyQt6.QtWidgets import QFileDialog
from memory_profiler import profile

import nidaqmx as nd
import nidaqmx.errors
from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, Coupling, DigitalPatternCondition
from nidaqmx.constants import DigitalWidthUnits, Slope, Timescale, TriggerType, WindowTriggerCondition1



os.chdir(sys.path[0])
initial_path = os.getcwd()

base_measurement_path = r'G:\UED_measurements'

pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOptions(antialias=True)


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
    amplitude = float(initial_guess[0])
    center = float(initial_guess[1])
    sigma = float(initial_guess[2])
    gamma = float(initial_guess[3])
    c = float(initial_guess[4])
    # params = model.make_params(amplitude=initial_guess[0], center=initial_guess[1], sigma=initial_guess[2], gamma=initial_guess[3], c=initial_guess[4], b=0, a=0)
    model.set_param_hint('amplitude', value=amplitude, min=0)
    model.set_param_hint('center', value=center, min=0)
    model.set_param_hint('sigma', value=sigma, min=0)
    model.set_param_hint('gamma', value=gamma, min=0)
    model.set_param_hint('c', value=c, min=0)
    model.set_param_hint('b', value=0, min=0)
    y = data
    result = model.fit(y, x=x, max_nfev=400)
    return result


class DateAxisItem(AxisItem):
    """
    A tool that provides a date-time aware axis. It is implemented as an
    AxisItem that interpretes positions as unix timestamps (i.e. seconds
    since 1970).
    The labels and the tick positions are dynamically adjusted depending
    on the range.
    It provides a  :meth:`attachToPlotItem` method to add it to a given
    PlotItem
    """

    # Max width in pixels reserved for each label in axis
    _pxLabelWidth = 80

    def __init__(self, *args, **kwargs):
        AxisItem.__init__(self, *args, **kwargs)
        self._oldAxis = None

    def tickValues(self, minVal, maxVal, size):
        """
        Reimplemented from PlotItem to adjust to the range and to force
        the ticks at "round" positions in the context of time units instead of
        rounding in a decimal base
        """

        maxMajSteps = int(size/self._pxLabelWidth)

        dt1 = datetime.fromtimestamp(minVal)
        dt2 = datetime.fromtimestamp(maxVal)

        dx = maxVal - minVal
        majticks = []

        if dx > 63072001:  # 3600s*24*(365+366) = 2 years (count leap year)
            d = timedelta(days=366)
            for y in range(dt1.year + 1, dt2.year):
                dt = datetime(year=y, month=1, day=1)
                majticks.append(mktime(dt.timetuple()))

        elif dx > 5270400:  # 3600s*24*61 = 61 days
            d = timedelta(days=31)
            dt = dt1.replace(day=1, hour=0, minute=0,
                             second=0, microsecond=0) + d
            while dt < dt2:
                # make sure that we are on day 1 (even if always sum 31 days)
                dt = dt.replace(day=1)
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 172800:  # 3600s24*2 = 2 days
            d = timedelta(days=1)
            dt = dt1.replace(hour=0, minute=0, second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 7200:  # 3600s*2 = 2hours
            d = timedelta(hours=1)
            dt = dt1.replace(minute=0, second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 1200:  # 60s*20 = 20 minutes
            d = timedelta(minutes=10)
            dt = dt1.replace(minute=(dt1.minute // 10) * 10,
                             second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 120:  # 60s*2 = 2 minutes
            d = timedelta(minutes=1)
            dt = dt1.replace(second=0, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 20:  # 20s
            d = timedelta(seconds=10)
            dt = dt1.replace(second=(dt1.second // 10) * 10, microsecond=0) + d
            while dt < dt2:
                majticks.append(mktime(dt.timetuple()))
                dt += d

        elif dx > 2:  # 2s
            d = timedelta(seconds=1)
            majticks = range(int(minVal), int(maxVal))

        else:  # <2s , use standard implementation from parent
            return AxisItem.tickValues(self, minVal, maxVal, size)

        L = len(majticks)
        if L > maxMajSteps:
            majticks = majticks[::int(np.ceil(float(L) / maxMajSteps))]

        return [(d.total_seconds(), majticks)]

    def tickStrings(self, values, scale, spacing):
        """Reimplemented from PlotItem to adjust to the range"""
        ret = []
        if not values:
            return []

        if spacing >= 31622400:  # 366 days
            fmt = "%Y"

        elif spacing >= 2678400:  # 31 days
            fmt = "%Y %b"

        elif spacing >= 86400:  # = 1 day
            fmt = "%b/%d"

        elif spacing >= 3600:  # 1 h
            fmt = "%b/%d-%Hh"

        elif spacing >= 60:  # 1 m
            fmt = "%H:%M"

        elif spacing >= 1:  # 1s
            fmt = "%H:%M:%S"

        else:
            # less than 2s (show microseconds)
            # fmt = '%S.%f"'
            fmt = '[+%fms]'  # explicitly relative to last second

        for x in values:
            try:
                t = datetime.fromtimestamp(x)
                ret.append(t.strftime(fmt))
            except ValueError:  # Windows can't handle dates before 1970
                ret.append('')

        return ret

    def attachToPlotItem(self, plotItem):
        """Add this axis to the given PlotItem
        :param plotItem: (PlotItem)
        """
        self.setParentItem(plotItem)
        viewBox = plotItem.getViewBox()
        self.linkToView(viewBox)
        self._oldAxis = plotItem.axes[self.orientation]['item']
        self._oldAxis.hide()
        plotItem.axes[self.orientation]['item'] = self
        pos = plotItem.axes[self.orientation]['pos']
        plotItem.layout.addItem(self, *pos)
        self.setZValue(-1000)

    def detachFromPlotItem(self):
        """Remove this axis from its attached PlotItem
        (not yet implemented)
        """
        raise NotImplementedError()  # TODO


DCU_IP = '169.254.254.1'
ec = DEigerClient(DCU_IP)
# ec.setDetectorConfig('incident_energy', 40000)  # Ev
# # ec.setDetectorConfig('threshold_energy', 20000)
# ec.setDetectorConfig('trigger_mode', 'ints')


url = "http://169.254.254.1/monitor/api/1.8.0/images/monitor"

# bkg_file = h5py.File(r"bkg\bkg.h5", 'r')
# bkg_img = np.array(bkg_file['data'], dtype=bool)
bkg_img = np.zeros((512,512))
# inverted_bkg = (np.array(bkg_img, dtype=np.int32))*(-1)+1
# inverted_bkg = inverted_bkg*0+1
# print(inverted_bkg)

LTS_position = 0.0
loop_number = 0
ref_image = np.array([])


img = np.zeros((512,512))
img_arr = np.array(img)
img_arr_2 = np.array(img)
# img_arr = np.load('graphite_300K.npy')
# jet = cm.get_cmap('jet', 12)
# colors_list = []
# for i in range(jet.N):
#     colors_list.append(tuple([z * 255 for z in jet(i)[:-1]]))
# cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 12), color=colors_list)


# app = QtGui.QApplication(sys.argv)
app = pg.mkQApp('UED')
app.setStyle('Windows10')



new_main_window = uic.loadUi("./Main_UED_new.ui")
# new_main_window.resize(1920, 1160)
window_leakvalve = uic.loadUi("Leak_valve_main_window.ui")
window_notes = uic.loadUi("notes.ui")
window_settings_detector = uic.loadUi("settings_detector.ui")
new_main_window.showMaximized()

img_1 = new_main_window.main_image
img_1.ui.roiBtn.hide()
img_1.ui.menuBtn.hide()
viewbox_1 = img_1.view
img_1.setColorMap(pg.colormap.get('CET-R4'))

img_2 = new_main_window.difference_image
img_2.ui.roiBtn.hide()
img_2.ui.menuBtn.hide()
viewbox_2 = img_2.view
img_2.setColorMap(pg.colormap.get('CET-D1'))


img_1.setImage(img_arr, autoRange=True, autoLevels=True)
col = 0
row = 0
img_pixel_value = 0


vLine = pg.InfiniteLine(angle=90, movable=False)
hLine = pg.InfiniteLine(angle=0, movable=False)
img_1.addItem(vLine, ignoreBounds=True)
img_1.addItem(hLine, ignoreBounds=True)


def mouseClicked(ev):
    global col, row, img_pixel_value, pixel_info_lbl
    p = QtCore.QPointF(ev.pos()[0], ev.pos()[1])
    data = img_1.image  # or use a self.data member
    nRows, nCols = data.shape
    scenePos = img_1.getImageItem().mapFromScene(p)
    row, col = int(scenePos.y()), int(scenePos.x())
    if (0 <= row < nRows) and (0 <= col < nCols) and str(ev.button()) == 'MouseButton.LeftButton':
        img_pixel_value = data[row, col]
        if lock_cross_chkbx.isChecked() is False:
            vLine.setPos(col+0.5)
            hLine.setPos(row+0.5)
            vLine_cross.setPos(col+0.5)
            hLine_cross.setPos(row+0.5)
            window_cross_v_plt.setData(0.5+np.arange(data.shape[0]), img_arr[row, :])
            window_cross_h.getViewBox().invertX(True)
            window_cross_h_plt.setData(img_arr[:, col], 0.5+np.arange(data.shape[1]))
            pixel_info_lbl.setText("X = %3d \nY = %3d \nI = %d" % (col, row, img_pixel_value))


viewbox_1.scene().sigMouseClicked.connect(mouseClicked)
window_cross_v = new_main_window.vertical_cross_plot
window_cross_v_plt = window_cross_v.plot()
window_cross_v_plt_fit = window_cross_v.plot()
window_cross_v_plt.setData(img_arr[0, :])
window_cross_v.setXLink(viewbox_1)
window_cross_v.showAxis('right')
window_cross_v.hideAxis('left')
vLine_cross = pg.InfiniteLine(angle=90, movable=False)
window_cross_v.addItem(vLine_cross, ignoreBounds=True)
window_cross_h = new_main_window.horizontal_cross_plot
window_cross_h_plt = window_cross_h.plot()
window_cross_h_plt_fit = window_cross_h.plot()
window_cross_h_plt.getViewBox().invertY(True)
window_cross_h_plt_fit.getViewBox().invertY(True)
window_cross_h_plt.setData(img_arr[:, 0], range(img_arr.shape[1]))
window_cross_h.getViewBox().invertX(True)
window_cross_h.setYLink(viewbox_1)
hLine_cross = pg.InfiniteLine(angle=0, movable=False)
window_cross_h.addItem(hLine_cross, ignoreBounds=True)

acquire_btn = new_main_window.live_btn


stop_btn = new_main_window.live_stop_btn
exposure_slider = new_main_window.exposure_slider
exposure_label = new_main_window.exposure_label
live_acquire_btn = new_main_window.live_acquire_btn
millis_btn = new_main_window.millis_btn
micros_btn = new_main_window.micros_btn
expo1000_btn = new_main_window.expo1000_btn
expo500_btn = new_main_window.expo500_btn
expo200_btn = new_main_window.expo200_btn

pixel_info_lbl = new_main_window.pixel_info_lbl
autorange_btn = new_main_window.autorange_btn

folder_btn = new_main_window.folder_btn
notes_btn = new_main_window.notes_btn
ROI_chkbx = new_main_window.ROI_chkbx
fit_chkbx = new_main_window.fit_chkbx
lock_cross_chkbx = new_main_window.lock_cross_chkbx
integral_label = new_main_window.integral_label
integral_label.setFont(QFont("Arial", 16))
# fit_label = new_main_window.fit_label

# energy_scan_btn = QtGui.QPushButton("Energy scan")

exposure = 1000
exposure_divider = 1000.
triggering = True
save_files = False
allow_drawing = True
threshold_energy = 30000

roi = pg.ROI([0, 0], [200, 50], pen=pg.mkPen('m', width=2), hoverPen=pg.mkPen('m', width=4))
roi.addRotateHandle([0, 0], [0.5, 0.5])
roi.addScaleHandle([1, 1], [0, 0])
roi.setZValue(1e9)

def read_run_number():
    global run_number, initial_path
    filepath = os.path.join(initial_path, 'run_number.txt')
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            run_number = int(f.readline())
    else:
        run_number = 0
        with open(filepath, 'w') as f:
            f.write('0')
    new_main_window.label_30.setText('Run number: %06d' % run_number)
    
def update_run_number():
    global run_number, initial_path, images_path
    run_number += 1
    filepath = os.path.join(initial_path, 'run_number.txt')
    with open(filepath, 'w') as f:
        f.write('%d' % run_number)
    new_main_window.label_30.setText('Run number: %06d' % run_number)
    today = datetime.now()
    year = today.year
    month = today.strftime("%m %B")
    day = today.strftime("%d")
    folder_path = os.path.join(base_measurement_path, str(year), str(month), str(day), f'r{run_number:06d}', 'RAW')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    images_path = folder_path


read_run_number()

def ROIupdatePlot():
    global img_1, roi, img_arr, p2, integral_label
    if ROI_chkbx.isChecked():
        img_1.addItem(roi)
        selected = roi.getArrayRegion(img_arr, img_1.getImageItem())
        data_v = selected.sum(axis=0)
        data_h = selected.sum(axis=1)
        x_data_v = np.arange(len(data_v))
        x_data_v = x_data_v + roi.pos()[0]
        x_data_h = np.arange(len(data_h))
        x_data_h = x_data_h + roi.pos()[1]
        if fit_chkbx.isChecked():
            try:
                data_v_fit = fit_gauss(data_v, x_data_v)
                data_h_fit = fit_gauss(data_h, x_data_h)
            except Exception as e:
                print(e)
                pass
            window_cross_v_plt_fit.setData(x_data_v, data_v_fit.best_fit, clear=True, pen=pg.mkPen('y', width=2))
            window_cross_h_plt_fit.setData(data_h_fit.best_fit, x_data_h, clear=True, pen=pg.mkPen('y', width=2))
            # fit_label.setText("Height: "+format(int(data_v_fit.values['height']), '.3g') + "\nCenter X:" + format(format(data_v_fit.values['center'], '5.2f')) + "\nCenter Y:"+format(format(data_h_fit.values['center'], '5.2f'))+"\nFWHM X:"+format(
                # format(data_v_fit.values['fwhm'], '5.2f')) + "\nFWHM Y:"+format(format(data_h_fit.values['fwhm'], '5.2f')) + '\nEccentricity: '+format(data_h_fit.values['fwhm']/data_v_fit.values['fwhm'], '5.3f'))
        else:
            window_cross_h_plt_fit.clear()
            window_cross_v_plt_fit.clear()
            # fit_label.setText("")
        window_cross_v_plt.setData(
            x_data_v, data_v, clear=True, pen=pg.mkPen('m', width=2))

        window_cross_h_plt.setData(
            data_h, x_data_h, clear=True, pen=pg.mkPen('m', width=2))

        integral_label.setText("Integral:\n"+format(data_h.sum(axis=0), ".3g"))


roi.sigRegionChanged.connect(ROIupdatePlot)
ROIupdatePlot()


def print_roi_pos():
    global roi
    if ROI_chkbx.isChecked():
        img_1.addItem(roi)
        ROIupdatePlot()
    else:
        img_1.removeItem(roi)


ROI_chkbx.toggled.connect(print_roi_pos)


def update_roi_diff():
    global img_arr, img_arr_2, roi
    if new_main_window.pushButton_14.isChecked() and ROI_chkbx.isChecked():
        roi_on = roi.getArrayRegion(img_arr, img_1.getImageItem())
        roi_off = roi.getArrayRegion(img_arr_2, img_1.getImageItem())
        # print('in update img, roi_on:', np.sum(roi_on))
        # print('in update img, roi_off:', np.sum(roi_off))
        
        img_2.setImage(roi_on-roi_off, autoRange=False, autoLevels=False, autoHistogramRange=False)
        
def autoscale_roi_diff():
    global img_arr, img_arr_2, roi
    roi_on = roi.getArrayRegion(img_arr, img_1.getImageItem())
    roi_off = roi.getArrayRegion(img_arr_2, img_1.getImageItem())
    min = np.min(roi_on-roi_off)
    max = np.max(roi_on-roi_off)
    biggest = np.max([np.abs(min), np.abs(max)])
    img_2.setLevels(-biggest, biggest)
    img_2.setHistogramRange(-biggest, biggest)

roi.sigRegionChanged.connect(update_roi_diff)
new_main_window.pushButton_15.clicked.connect(autoscale_roi_diff)
def plot_func():
    global img_arr, img_arr_old, allow_drawing, row, col, img_pixel_value, img_arr_2, roi, pixel_info_lbl
    if allow_drawing:
        img_pixel_value = img_arr[row, col]
        pixel_info_lbl.setText("X = %3d \nY = %3d \nI = %d" % (col, row, img_pixel_value))
        img_1.setImage(img_arr, autoRange=False, autoLevels=False, autoHistogramRange=False)
        update_roi_diff()




def autorange_img():
    img_1.autoRange()
    img_1.autoLevels()
    img_1.autoHistogramRange()


class AThread(QThread):
    updated = QtCore.pyqtSignal(int)
    started = QtCore.pyqtSignal(int)

    def run(self):
        global exposure, triggering, quadro_frametime
        self._is_running = True
        while(self._is_running):
            total_exposure = exposure/exposure_divider
            quadro_frametime = total_exposure
            if total_exposure < 2.5e-4:
                total_exposure = 2.5e-4
            if simulator == 0:
                ec.setDetectorConfig('count_time', total_exposure)
                ec.setDetectorConfig('frame_time', total_exposure)
                ec.sendDetectorCommand('arm')
                # count = 0
                while triggering:
                    self.started.emit(1)
                    try:
                        ec.sendDetectorCommand('trigger')
                    except Exception:
                        print(" Trigger error")
                        stop_monitor()
                        ec.sendDetectorCommand('disarm')
                    update_img()
                    self.updated.emit(1)
            else:
                while triggering:
                    self.started.emit(1)
                    time.sleep(total_exposure)
                    update_img()
                    self.updated.emit(1)

    def stop(self):
        time.sleep(5)


thread = AThread()
thread.updated.connect(plot_func)
img_arr_old = img_arr.copy()*0.
# folderpath = os.getcwd()
folderpath = r'G:\UED_measurements'
# folderpath = r"C:/Users/paull/OneDrive - epfl.ch/Documents/ued_test_files/test_01"
new_main_window.folder_path_entry.setText(folderpath)


def select_folder():
    global folderpath
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        new_main_window, 'Select Folder', directory=folderpath)
    new_main_window.folder_path_entry.setText(folderpath)
    os.chdir(folderpath)


def update_folder():
    global folderpath
    folderpath = new_main_window.folder_path_entry.text()
    try:
        os.path.exists(folderpath)
        os.chdir(folderpath)
    except Exception:
        os.mkdir(folderpath)
        os.chdir(folderpath)


def update_exposure_label(value):
    exposure_label.setText(str(value))

quadro_frametime = 1

def update_exposure_value():
    global exposure, triggering, quadro_frametime
    triggering = False
    if simulator == 0:
        ec.sendDetectorCommand('disarm')
    exposure = exposure_slider.value()
    quadro_frametime = exposure
    update_exposure_label(exposure)
    if simulator == 0:
        ec.setDetectorConfig('count_time', exposure/1000.)
        ec.setDetectorConfig('frame_time', quadro_frametime/1000.)
        ec.sendDetectorCommand('arm')
    triggering = True


def set_exposure_slider(exp):
    exposure_slider.setValue(exp)
    update_exposure_value()


expo1000_btn.clicked.connect(partial(set_exposure_slider, 1000))
expo500_btn.clicked.connect(partial(set_exposure_slider, 500))
expo200_btn.clicked.connect(partial(set_exposure_slider, 200))
new_main_window.folder_path_entry.editingFinished.connect(update_folder)


def save_image_h5(filename, array, array_2 = None, data_diode = None):
    global temperature_A, temperature_B, LTS_position, pressure_list, SimStep_x, SimStep_y, SimStep_z, SimStep_th, exposure, sensor_A, sensor_B, threshold_energy, loop_number, pump_power_entry, ref_image, threshold_entry, quadro_frm_entry, shutter_status_2, lakeshore_df, LTS_position_df, quadro_nimages, quadro_frametime, quadro_incidentenergy
    global attocube_encoder_raw, attocube_encoder_deg, run_number, base_measurement_path, list_of_files_per_loop, images_path
    file_path = os.path.join(images_path, filename+'.h5')
    try:
        list_of_files_per_loop.append(file_path)
    except Exception:
        pass
    # print(file_path)
    
    
    file = h5py.File(file_path, "w")
    acquisition_mode = new_main_window.comboBox.currentText()
    dset = file.create_dataset("data", array.shape, 'int32',chunks=array.shape, compression="gzip", compression_opts=1)
    # dset_ref = file.create_dataset("ref_img", ref_image.shape, 'int32', compression="gzip", compression_opts=9)
    dset[:] = array[:]
    
    if array_2 is not None:
        dset2 = file.create_dataset("data2", array_2.shape, 'int32',chunks=array.shape, compression="gzip", compression_opts=1)
        dset2[:] = array_2[:]
    # dset_ref[:] = ref_image[:]
    
    if data_diode is not None:
        dset3 = file.create_dataset("data_diode", data_diode.shape, 'int32',chunks=data_diode.shape, compression="gzip", compression_opts=1)
        dset3[:] = data_diode[:]
    
    if (quadro_nimages == 1) and (acquisition_mode != 'EXTG'):
        dset.attrs['Time'] = format(time.time(), "014.3f")
        dset.attrs['Pressure'] = pressure_list[-1]
        dset.attrs['Temperature_A'] = lakeshore_df['temperature_A'].iloc[-1]
        dset.attrs['Temperature_B'] = lakeshore_df['temperature_B'].iloc[-1]
        dset.attrs['Sensor_A_Ohm'] = lakeshore_df['sensor_A'].iloc[-1]
        dset.attrs['Sensor_B_Ohm'] = lakeshore_df['sensor_B'].iloc[-1]
        dset.attrs['LTS_position'] = LTS_position
    if (quadro_nimages != 1) and (acquisition_mode == 'EXTG'):
        dset.attrs['Time'] = format(time.time(), "014.3f")
        dset.attrs['Pressure'] = pressure_list[-1]
        dset.attrs['Temperature_A'] = lakeshore_df['temperature_A'].iloc[-1]
        dset.attrs['Temperature_B'] = lakeshore_df['temperature_B'].iloc[-1]
        dset.attrs['Sensor_A_Ohm'] = lakeshore_df['sensor_A'].iloc[-1]
        dset.attrs['Sensor_B_Ohm'] = lakeshore_df['sensor_B'].iloc[-1]
        dset.attrs['LTS_position'] = LTS_position
    elif (quadro_nimages != 1) and (acquisition_mode != 'EXTG'):
        dset.attrs['Time'], dset.attrs['Pressure'], dset.attrs['Temperature_A'], dset.attrs['Temperature_B'], dset.attrs['Sensor_A_Ohm'], dset.attrs['Sensor_B_Ohm'], dset.attrs['LTS_position'] = interpolate_PT()
    dset.attrs['Acquisition_mode'] = acquisition_mode
    dset.attrs['Time_for_humans'] = str(dtm.fromtimestamp(time.time()))
    dset.attrs['Simstep_positions'] = [SimStep_x, SimStep_y, SimStep_z, SimStep_th]
    dset.attrs['Exposure_time_ms'] = exposure
    dset.attrs['Frame_time'] = quadro_frametime
    #dset.attrs['threshold_energy'] = get_detector_parameter('threshold_energy')
    dset.attrs['incident_energy'] = quadro_incidentenergy
    dset.attrs['nimages'] = quadro_nimages
    dset.attrs['loop_number'] = loop_number
    dset.attrs['pump_power'] = float(new_main_window.pump_power_entry.text())
    dset.attrs['PHI_RAW'] = attocube_encoder_raw
    dset.attrs['PHI_DEG'] = attocube_encoder_deg
    dset.attrs['pump_area'] = float(new_main_window.pump_area_entry.text())
    dset.attrs['RF_Cavity_phase'] = float(new_main_window.RF_phase_entry.text())
    dset.attrs['RF_Cavity_power'] = float(new_main_window.RF_power_entry.text())
    dset.attrs['notes'] = window_notes.notes.toPlainText()
    dset.attrs['shutter'] = shutter_status_2
    dset.attrs['UV_power'] = float(new_main_window.UV_power_entry.text())
    file.close()
    
def gaussian_array(amplitude, sigma, shape, center_noise=0.001, sigma_noise=0.0002, amplitude_noise=0.002, sine_amp = 10):
    # Generate random noise for the center and sigma parameters
    x_noise = np.random.normal(1, center_noise)
    y_noise = np.random.normal(1, center_noise)
    sigma_noise_val = np.random.normal(1, sigma_noise)
    amplitude_noise_val = np.random.normal(1, amplitude_noise)

    # Calculate the center of the array with added noise
    x_center = (shape[0] // 2) * x_noise
    y_center = (shape[1] // 2) * y_noise
    x_center = x_center+sine_amp*np.sin(time.time()/10)

    # Generate a grid of x and y values centered at the center of the array
    x, y = np.meshgrid(np.arange(shape[0]) - x_center, np.arange(shape[1]) - y_center)
    # Generate the 2D Gaussian peak with the specified amplitude, sigma, and noise
    gaussian = amplitude_noise_val * amplitude * np.exp(-(x**2 + y**2) / (2 * (sigma * sigma_noise_val)**2))

    return gaussian


def update_img():
    global img_arr, save_files, img_arr_old, ask_plot_image, inverted_bkg
    img_arr_old = img_arr
    if simulator == 0:
        try:
            img = Image.open(requests.get(url, stream=True).raw).convert("I")
            img_arr = np.array(img)
            img_arr = np.flipud(np.rot90(img_arr))
        except Exception:
            pass
    else:
        # generate an image 512x512 with a 2d gaussian peak around 256,256, with sigma=10 and amplitude=10000
        img_arr = gaussian_array(10000, 10, (512, 512), center_noise=0.001, sigma_noise=0.0002, amplitude_noise=0.002)
    
    # metadata = get_metadata()
    if save_files:
        filepattern = format(time.time(), "014.3f")
        # filename = filepattern+".tiff"
        save_image_h5(filepattern, img_arr)


acquiring = False


def start_monitor():
    global exposure, triggering, quadro_frametime
    exposure = exposure_slider.value()
    quadro_frametime = exposure
    acquire_btn.setEnabled(False)
    live_acquire_btn.setEnabled(False)
    folder_btn.setEnabled(False)
    stop_btn.setEnabled(True)
    acquire_btn.setStyleSheet('')
    live_acquire_btn.setStyleSheet('')
    stop_btn.setStyleSheet('QPushButton { background-color: red }')
    if simulator == 0:
        ec.setDetectorConfig('count_time', exposure/1000.)
        ec.setDetectorConfig('frame_time', quadro_frametime/1000.)
        ec.setMonitorConfig('mode', 'enabled')
        ec.setMonitorConfig('buffer_size', 10)
        ec.setDetectorConfig('ntrigger', 2000000)
        ec.setDetectorConfig('nimages', 1)
    triggering = True
    thread.start()


def kill_monitor(thread):
    global kill
    if kill:
        thread.terminate()
        if simulator == 0:
            ec.sendDetectorCommand('disarm')
        kill = 0


thread.started.connect(partial(kill_monitor, thread))
kill = 0


def stop_monitor():
    global save_files, triggering, kill
    acquire_btn.setEnabled(True)
    live_acquire_btn.setEnabled(True)
    folder_btn.setEnabled(True)
    stop_btn.setEnabled(False)
    acquire_btn.setStyleSheet('QPushButton { background-color: rgb(85, 255, 0) }')
    live_acquire_btn.setStyleSheet('QPushButton { background-color: rgb(255, 170, 0) }')
    stop_btn.setStyleSheet('')
    save_files = 0
    kill = 1


def acquire_monitor_image():
    global acquiring
    acquiring = True


def start_live_acquire():
    global exposure, save_files, quadro_frametime
    exposure = exposure_slider.value()
    quadro_frametime = exposure
    acquire_btn.setEnabled(False)
    live_acquire_btn.setEnabled(False)
    folder_btn.setEnabled(False)
    stop_btn.setEnabled(True)
    acquire_btn.setStyleSheet('')
    live_acquire_btn.setStyleSheet('')
    stop_btn.setStyleSheet('QPushButton { background-color: red }')
    save_files = True
    update_run_number()
    if simulator == 0:
        ec.setDetectorConfig('count_time', exposure/1000.)
        ec.setDetectorConfig('frame_time', quadro_frametime/1000.)
        ec.setMonitorConfig('mode', 'enabled')
        ec.setMonitorConfig('buffer_size', 10)
        ec.setDetectorConfig('ntrigger', 2000000)
        ec.setDetectorConfig('nimages', 1)
    thread.start()


def update_exposure_divider():
    global exposure_divider
    if millis_btn.isChecked() is True:
        exposure_divider = 1000.
        update_exposure_value()
    if micros_btn.isChecked() is True:
        exposure_divider = 1.0e6
        update_exposure_value()


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


ask_plot_image = False
timer_progress_value = 0


def plot_image_main():
    global img_arr, img_arr_old, ask_plot_image, time_PR_list, pressure_list, time_list, temperature_A, temperature_B, LTS_position, pressure_mean_list, timer_progress_value, temp_busy, lakeshore_df
    # new_main_window.acquisition_progressBar.setValue(int(timer_progress_value))
    if temp_busy == 0:
        if len(time_list) > 0 & len(time_list) < 500:
            try:
                pressure_plot.setData(np.array(time_PR_list), np.array(pressure_list))
                pressure_plot_mean.setData(np.array(time_PR_list), np.array(pressure_mean_list))
                Pressure_lbl.setText("P ="+format(pressure_list[-1], ".2e"))
                TempA_lbl.setText("T A:"+str(lakeshore_df["temperature_A"].iloc[-1])+"K")
                T_A.setData(np.stack(lakeshore_df["time"]), np.stack(lakeshore_df["temperature_A"]))
                TempB_lbl.setText("T B:"+str(lakeshore_df["temperature_B"].iloc[-1])+"K")
                T_B.setData(np.stack(lakeshore_df["time"]), np.stack(lakeshore_df["temperature_B"]))
                Delta_T_lbl.setText("Delta T:"+format(lakeshore_df["temperature_A"].iloc[-1]-lakeshore_df["temperature_B"].iloc[-1], "2.3")+"K")
                pass
            except Exception:
                pass
        if len(pressure_list) > 500:
            try:
                pressure_plot.setData(np.array(time_PR_list[-500:]), np.array(pressure_list[-500:]))
                pressure_plot_mean.setData(np.array(time_PR_list[-500:]), np.array(pressure_mean_list[-500:]))
                Pressure_lbl.setText("P ="+format(pressure_list[-1], ".2e"))
                TempA_lbl.setText("T A:"+str(lakeshore_df["temperature_A"].iloc[-1])+"K")
                T_A.setData(np.stack(lakeshore_df["time"].iloc[-500:]), np.stack(lakeshore_df["temperature_A"].iloc[-500:]))
                TempB_lbl.setText("T B:"+str(lakeshore_df["temperature_B"].iloc[-1])+"K")
                T_B.setData(np.stack(lakeshore_df["time"].iloc[-500:]), np.stack(lakeshore_df["temperature_B"].iloc[-500:]))
                Delta_T_lbl.setText("Delta T:"+format(lakeshore_df["temperature_A"].iloc[-1]-lakeshore_df["temperature_B"].iloc[-1], "2.3")+"K")
                pass
            except Exception:
                pass

    if ROI_chkbx.isChecked() is False:
        window_cross_v_plt.setData(
            0.5+np.arange(img_arr.shape[0]), np.int32(img_arr[row, :]))
        window_cross_h.getViewBox().invertX(True)
        window_cross_h_plt.setData(
            np.int32(img_arr[:, col]), 0.5+np.arange(img_arr.shape[1]))
    else:
        ROIupdatePlot()
    if len(lakeshore_df) > 20:
        dt = lakeshore_df['time'].iloc[-1]-lakeshore_df["time"].iloc[-20]
        dA = lakeshore_df['temperature_A'].iloc[-1]-lakeshore_df['temperature_A'].iloc[-20]
        dB = lakeshore_df['temperature_B'].iloc[-1]-lakeshore_df['temperature_B'].iloc[-20]
        Temp_RA_lbl.setText(f"Rate A: {dA/dt*60:+5.2f}K/min")
        Temp_RB_lbl.setText(f"Rate B: {dB/dt*60:+5.2f}K/min")
        # Delta_T_lbl.setText("Delta T:"+format(temperature_A[-1]-temperature_B[-1], "2.3")+"K")
    ask_plot_image = False


timer_img = QtCore.QTimer()
timer_img.timeout.connect(plot_image_main)

timer_img.start(100)


acquire_btn.clicked.connect(start_monitor)
stop_btn.clicked.connect(stop_monitor)
exposure_slider.valueChanged.connect(update_exposure_label)
exposure_slider.sliderReleased.connect(update_exposure_value)
folder_btn.clicked.connect(select_folder)
live_acquire_btn.clicked.connect(start_live_acquire)
millis_btn.toggled.connect(update_exposure_divider)
micros_btn.toggled.connect(update_exposure_divider)
notes_btn.clicked.connect(window_notes.show)
autorange_btn.clicked.connect(autorange_img)

SimStep_updatestatus = 1
Simstep_updating = False


simstep_connect_btn = new_main_window.simstep_connect_btn
simstep_disconnect_btn = new_main_window.simstep_disconnect_btn
# leakvalve_win_btn = new_main_window.leakvalve_win_btn


move_abs_x_btn = new_main_window.move_abs_x_btn
move_abs_x_entry = new_main_window.move_abs_x_entry
pos_x_lbl = new_main_window.pos_x_lbl
rel_stp_x_entry = new_main_window.rel_stp_x_entry
move_rel_x_min_btn = new_main_window.move_rel_x_min_btn
move_rel_x_min_h_btn = new_main_window.move_rel_x_min_h_btn
move_rel_x_plu_btn = new_main_window.move_rel_x_plu_btn
move_rel_x_plu_h_btn = new_main_window.move_rel_x_plu_h_btn
stop_x_btn = new_main_window.stop_x_btn
ref_x_btn = new_main_window.ref_x_btn

move_abs_y_btn = new_main_window.move_abs_y_btn
move_abs_y_entry = new_main_window.move_abs_y_entry
pos_y_lbl = new_main_window.pos_y_lbl
rel_stp_y_entry = new_main_window.rel_stp_y_entry
move_rel_y_min_btn = new_main_window.move_rel_y_min_btn
move_rel_y_min_h_btn = new_main_window.move_rel_y_min_h_btn
move_rel_y_plu_btn = new_main_window.move_rel_y_plu_btn
move_rel_y_plu_h_btn = new_main_window.move_rel_y_plu_h_btn
stop_y_btn = new_main_window.stop_y_btn
ref_y_btn = new_main_window.ref_y_btn

move_abs_z_btn = new_main_window.move_abs_z_btn
move_abs_z_entry = new_main_window.move_abs_z_entry
pos_z_lbl = new_main_window.pos_z_lbl
rel_stp_z_entry = new_main_window.rel_stp_z_entry
move_rel_z_min_btn = new_main_window.move_rel_z_min_btn
move_rel_z_min_h_btn = new_main_window.move_rel_z_min_h_btn
move_rel_z_plu_btn = new_main_window.move_rel_z_plu_btn
move_rel_z_plu_h_btn = new_main_window.move_rel_z_plu_h_btn
stop_z_btn = new_main_window.stop_z_btn
ref_z_btn = new_main_window.ref_z_btn

move_abs_th_btn = new_main_window.move_abs_th_btn
move_abs_th_entry = new_main_window.move_abs_th_entry
pos_th_lbl = new_main_window.pos_th_lbl
rel_stp_th_entry = new_main_window.rel_stp_th_entry
move_rel_th_min_btn = new_main_window.move_rel_th_min_btn
move_rel_th_min_h_btn = new_main_window.move_rel_th_min_h_btn
move_rel_th_plu_btn = new_main_window.move_rel_th_plu_btn
move_rel_th_plu_h_btn = new_main_window.move_rel_th_plu_h_btn
stop_th_btn = new_main_window.stop_th_btn
ref_th_btn = new_main_window.ref_th_btn


ser_simstep = serial.Serial()
ser_simstep.port = 'COM5'
ser_simstep.baudrate = 9600
ser_simstep.timeout = 0
ser_simstep.parity = serial.PARITY_EVEN
ser_simstep.bytesize = 7
ser_simstep.stopbits = 1
ser_simstep.xonxoff = 0


SimStep_x = 0
SimStep_y = 0
SimStep_z = 0
SimStep_th = 0


def update_SimStep_pos(axis):
    global SimStep_updatestatus, Simstep_updating, SimStep_x, SimStep_y, SimStep_z, SimStep_th
    if Simstep_updating:
        if int(axis) == 1:
            starting_pos = pos_x_lbl.text()
            ser_simstep.write(b'1oc\r')
            time.sleep(0.05)
            output = ""
            try:
                output = (int(ser_simstep.read(100)[7:16]))
            except Exception:
                pass
            else:
                pass
            pos_x_lbl.setText(str(output))
            SimStep_x = output
            if SimStep_updatestatus:
                if str(starting_pos) == str(output):
                    SimStep_updatestatus = 0
                    timer_x.stop()
                else:

                    SimStep_updatestatus = 1
                    timer_x.start(500)

        if int(axis) == 2:
            starting_pos = pos_y_lbl.text()
            ser_simstep.write(b'2oc\r')
            time.sleep(0.05)
            output = ""
            try:
                output = (int(ser_simstep.read(100)[7:16]))
            except Exception:
                pass
            else:
                pass
            pos_y_lbl.setText(str(output))
            SimStep_y = output
            if SimStep_updatestatus:
                if str(starting_pos) == str(output):
                    SimStep_updatestatus = 0
                    timer_y.stop()
                else:
                    SimStep_updatestatus = 1
                    timer_y.start(500)
        if int(axis) == 3:
            starting_pos = pos_z_lbl.text()
            ser_simstep.write(b'3oc\r')
            time.sleep(0.05)
            output = ""
            try:
                output = (int(ser_simstep.read(100)[7:16]))
            except Exception:
                pass
            else:
                pass
            pos_z_lbl.setText(str(output))
            SimStep_z = output
            if SimStep_updatestatus:
                if str(starting_pos) == str(output):
                    SimStep_updatestatus = 0
                    timer_z.stop()
                else:
                    SimStep_updatestatus = 1
                    timer_z.start(500)
        if int(axis) == 4:
            starting_pos = pos_th_lbl.text()
            ser_simstep.write(b'4oc\r')
            time.sleep(0.05)
            output = ""
            try:
                output = (int(ser_simstep.read(100)[7:16]))
            except Exception:
                pass
            else:
                pass
            pos_th_lbl.setText(str(output))
            SimStep_th = output
            if SimStep_updatestatus:
                if str(starting_pos) == str(output):
                    SimStep_updatestatus = 0
                    timer_th.stop()
                else:
                    SimStep_updatestatus = 1
                    timer_th.start(500)


timer_x = QtCore.QTimer()
timer_x.timeout.connect(partial(update_SimStep_pos, 1))
timer_x.start(500)
timer_y = QtCore.QTimer()
timer_y.timeout.connect(partial(update_SimStep_pos, 2))
timer_y.start(500)
timer_z = QtCore.QTimer()
timer_z.timeout.connect(partial(update_SimStep_pos, 3))
timer_z.start(500)
timer_th = QtCore.QTimer()
timer_th.timeout.connect(partial(update_SimStep_pos, 4))
timer_th.start(500)


def SimStep_move_relative(axis, direction, full):
    global SimStep_updatestatus, Simstep_updating
    stepsize = 1
    if int(axis) == 1:
        stepsize = rel_stp_x_entry.text()
        Simstep_updating = True
    if int(axis) == 2:
        stepsize = rel_stp_y_entry.text()
        Simstep_updating = True
    if int(axis) == 3:
        stepsize = rel_stp_z_entry.text()
        Simstep_updating = True
    if int(axis) == 4:
        stepsize = rel_stp_th_entry.text()
        Simstep_updating = True
    if full == 0:
        stepsize = str(int(int(stepsize)/2.))
    command = (axis.encode()+b'mr'+direction.encode()+stepsize.encode()+b'\r')
    ser_simstep.write(command)
    time.sleep(0.05)
    ser_simstep.read(100)
    if SimStep_updatestatus == 0:
        SimStep_updatestatus = 1
        update_SimStep_pos(axis)
    return True


def SimStep_move_absolute(axis):
    global SimStep_updatestatus, Simstep_updating
    new_position = 1
    if int(axis) == 1:
        new_position = move_abs_x_entry.text()
        Simstep_updating = True
    if int(axis) == 2:
        new_position = move_abs_y_entry.text()
        Simstep_updating = True
    if int(axis) == 3:
        new_position = move_abs_z_entry.text()
        Simstep_updating = True
    if int(axis) == 4:
        new_position = move_abs_th_entry.text()
        Simstep_updating = True
    command = (axis.encode()+b'ma'+str(new_position).encode()+b'\r')
    ser_simstep.write(command)
    time.sleep(0.05)
    ser_simstep.read(100)
    if SimStep_updatestatus == 0:
        SimStep_updatestatus = 1
        update_SimStep_pos(axis)
    return True


def SimStep_connect():
    global Simstep_updating
    ser_simstep.open()
    simstep_disconnect_btn.setEnabled(True)
    simstep_disconnect_btn.setStyleSheet('QPushButton { background-color: red }')
    simstep_connect_btn.setEnabled(False)
    simstep_connect_btn.setStyleSheet('')
    # SimStep_updatestatus = 1
    Simstep_updating = True
    update_SimStep_pos(1)
    update_SimStep_pos(2)
    update_SimStep_pos(3)
    update_SimStep_pos(4)


def SimStep_disconnect():
    timer_x.stop()
    timer_y.stop()
    timer_z.stop()
    timer_th.stop()
    ser_simstep.close()
    simstep_disconnect_btn.setEnabled(False)
    simstep_disconnect_btn.setStyleSheet('')
    simstep_connect_btn.setEnabled(True)
    simstep_connect_btn.setStyleSheet('QPushButton { background-color: rgb(85, 255, 0) }')


def SimStep_STOP(axis):
    ser_simstep.write(str(axis).encode()+b'ab\r')
    time.sleep(0.1)
    ser_simstep.read(100)
    ser_simstep.write(str(axis).encode()+b'RS\r')
    time.sleep(0.1)
    ser_simstep.read(100)


def SimStep_set_reference(axis):
    global SimStep_updatestatus
    ser_simstep.write(str(axis).encode()+b"cp0\r")
    time.sleep(0.1)
    ser_simstep.read(100)
    if SimStep_updatestatus == 0:
        SimStep_updatestatus = 1
        update_SimStep_pos(axis)


simstep_connect_btn.clicked.connect(SimStep_connect)
simstep_disconnect_btn.clicked.connect(SimStep_disconnect)

move_rel_x_min_btn.clicked.connect(partial(SimStep_move_relative, '1', '-', 1))
move_rel_x_plu_btn.clicked.connect(partial(SimStep_move_relative, '1', '+', 1))
move_rel_y_min_btn.clicked.connect(partial(SimStep_move_relative, '2', '-', 1))
move_rel_y_plu_btn.clicked.connect(partial(SimStep_move_relative, '2', '+', 1))
move_rel_z_min_btn.clicked.connect(partial(SimStep_move_relative, '3', '-', 1))
move_rel_z_plu_btn.clicked.connect(partial(SimStep_move_relative, '3', '+', 1))
move_rel_th_min_btn.clicked.connect(partial(SimStep_move_relative, '4', '-', 1))
move_rel_th_plu_btn.clicked.connect(partial(SimStep_move_relative, '4', '+', 1))

move_rel_x_min_h_btn.clicked.connect(partial(SimStep_move_relative, '1', '-', 0))
move_rel_x_plu_h_btn.clicked.connect(partial(SimStep_move_relative, '1', '+', 0))
move_rel_y_min_h_btn.clicked.connect(partial(SimStep_move_relative, '2', '-', 0))
move_rel_y_plu_h_btn.clicked.connect(partial(SimStep_move_relative, '2', '+', 0))
move_rel_z_min_h_btn.clicked.connect(partial(SimStep_move_relative, '3', '-', 0))
move_rel_z_plu_h_btn.clicked.connect(partial(SimStep_move_relative, '3', '+', 0))
move_rel_th_min_h_btn.clicked.connect(partial(SimStep_move_relative, '4', '-', 0))
move_rel_th_plu_h_btn.clicked.connect(partial(SimStep_move_relative, '4', '+', 0))

stop_x_btn.clicked.connect(partial(SimStep_STOP, 1))
stop_y_btn.clicked.connect(partial(SimStep_STOP, 2))
stop_z_btn.clicked.connect(partial(SimStep_STOP, 3))
stop_th_btn.clicked.connect(partial(SimStep_STOP, 4))

ref_x_btn.clicked.connect(partial(SimStep_set_reference, 1))
ref_y_btn.clicked.connect(partial(SimStep_set_reference, 2))
ref_z_btn.clicked.connect(partial(SimStep_set_reference, 3))
ref_th_btn.clicked.connect(partial(SimStep_set_reference, 4))

move_abs_x_btn.clicked.connect(partial(SimStep_move_absolute, '1'))
move_abs_y_btn.clicked.connect(partial(SimStep_move_absolute, '2'))
move_abs_z_btn.clicked.connect(partial(SimStep_move_absolute, '3'))
move_abs_th_btn.clicked.connect(partial(SimStep_move_absolute, '4'))


quadro_init_btn = new_main_window.quadro_init_btn
quadro_namepattern_entry = new_main_window.quadro_namepattern_entry
quadro_namenumber_entry = new_main_window.quadro_namenumber_entry
quadro_acquire_btn = new_main_window.quadro_acquire_btn


# def get_detector_status():
    # link_status = "http://169.254.254.1/detector/api/1.8.0/status/state"
    # f = urlopen(link_status)
    # myfile = f.read()
    # y = json.loads(myfile)


def get_detector_parameter(par):
    link = 'http://169.254.254.1/detector/api/1.8.0/config/'+par
    f = urlopen(link)
    myfile = f.read()
    y = json.loads(myfile)
    return y['value']


def initialize_detector():
    ec.sendDetectorCommand('initialize')


# def set_filewriter():
#     ec.setFileWriterConfig('mode', 'enabled')
#     # quadro_setFW_lbl.setText("Enabled")
#     ec.sendFileWriterCommand('clear')


get_quadro_status = False
image_acquired = 0

quadro_nimages = 1

quadro_incidentenergy = 40000

def set_quadro_settings():
    global quadro_nimages, exposure, quadro_frametime, quadro_incidentenergy
    quadro_nimages = int(new_main_window.lineEdit_2.text())
    trigger_mode = new_main_window.comboBox.currentText().lower()
    filenumber = int(quadro_namenumber_entry.text())
    filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
    exposure = float(new_main_window.lineEdit_6.text())
    quadro_frametime = float(new_main_window.lineEdit_7.text())
    quadro_incidentenergy = int(new_main_window.spinBox.value())
    ec.sendDetectorCommand('disarm')
    ec.setDetectorConfig('trigger_mode', 'ints')
    ec.setDetectorConfig('incident_energy', quadro_incidentenergy)  # Ev
    ec.setDetectorConfig('ntrigger', int(new_main_window.lineEdit.text()))  # ntrigger must be set to 1 for extg mode
    ec.setDetectorConfig('nimages', quadro_nimages)  # nimage must be even for extg mode
    ec.setFileWriterConfig('nimages_per_file', int(new_main_window.lineEdit_3.text()))
    ec.setDetectorConfig('nexpi', int(new_main_window.lineEdit_4.text()))  # expi, number of total triggers
    ec.setDetectorConfig('trigger_mode', trigger_mode)  # external trigger gatted mode
    ec.setDetectorConfig('counting_mode', 'normal')
    # ec.setDetectorConfig('retrigger_applied',False)  # doesn't exist
    if trigger_mode == 'extg':
        ec.setDetectorConfig('extg_mode', 'double')  # this is to set pump probe
        ec.setDetectorConfig('countrate_correction_applied', False)
        #set_GladzPD() ## so that the photodiode acquire the same window as quadro
        initialize_NI6251()
    ec.setMonitorConfig('mode', new_main_window.comboBox_2.currentText().lower())
    ec.sendFileWriterCommand('clear')
    ec.setMonitorConfig('buffer_size', int(new_main_window.spinBox_2.value()))
    # string 'False' to boolean False
    if new_main_window.comboBox_4.currentText() == 'False':
        ec.setDetectorConfig('countrate_correction_applied', False)
    else:
        ec.setDetectorConfig('countrate_correction_applied', True)  # must be set to false for extg mode
    # # frame time and count time should be set to the real values as close as possible
    ec.setDetectorConfig('count_time', exposure/1000)
    ec.setDetectorConfig('frame_time', quadro_frametime/1000)
    ec.setFileWriterConfig('name_pattern', filename)
    ec.setFileWriterConfig('compression_enabled', True)
    ec.setFileWriterConfig('mode', new_main_window.comboBox_3.currentText().lower())
    ec.sendDetectorCommand('disarm')


new_main_window.pushButton_6.clicked.connect(set_quadro_settings)
new_main_window.pushButton_8.clicked.connect(lambda: ec.sendDetectorCommand('disarm'))


class BThread(QThread):
    updated = QtCore.pyqtSignal(int)
    started = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)

    def run(self):
        update_run_number()
        global img_arr, img_arr_2, save_files, img_arr_old, exposure, image_acquired, quadro_frametime, quadro_nimages, acquisition_start_time, acquisition_end_time
        if new_main_window.comboBox.currentText() == 'INTS':

            filenumber = int(quadro_namenumber_entry.text())
            filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
            n_frames = int(new_main_window.lineEdit_2.text())
            
            if simulator == 0:
                ec.setFileWriterConfig('name_pattern', filename)
                image_acquired = 0
                ec.sendDetectorCommand('arm')
                self.started.emit(1)
                acquisition_start_time = time.time()
                ec.sendDetectorCommand('trigger')
                ec.sendDetectorCommand('disarm')
                acquisition_end_time = time.time()
                time.sleep(1.5)
                self.finished.emit(1)
                quadro_namenumber_entry.setText(str(filenumber + 1))
                url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
                hf = None
                while hf is None:
                    try:
                        hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
                        print("the try went through")
                    except Exception:
                        pass
                if n_frames == 1:
                    print('the h5 file when acquire 1img : ',hf['entry']['data']['data'])
                    img_arr = np.array(hf['entry']['data']['data'][0], dtype=np.int64)
                    print(hf['entry']['data']['data'])
                    img_arr_original = np.flipud(np.rot90(img_arr))
                    self.updated.emit(1)
                    save_image_h5(filename, img_arr_original)
                    quadro_acquire_btn.setEnabled(True)
                    hf = None
                else:
                    img_arr3d = img_arr = np.array(hf['entry']['data']['data'], dtype=np.int64)
                    img_arr = img_arr3d[-1].copy()
                    # vec = np.argwhere(img_arr > 4e9)
                    # img_arr[vec[:, 0], vec[:, 1]] = 0
                    img_arr = np.flipud(np.rot90(img_arr))
                    self.updated.emit(1)
                    save_image_h5(filename, img_arr3d)
                    quadro_acquire_btn.setEnabled(True)
            else:
                self.started.emit(1)
                acquisition_start_time = time.time()
                time.sleep(quadro_nimages*quadro_frametime/1000)
                if quadro_nimages == 1:
                    img_arr = np.random.randint(0, 100, (512, 512))
                    quadro_namenumber_entry.setText(str(filenumber + 1))
                    self.updated.emit(1)
                    acquisition_end_time = time.time()
                    save_image_h5(filename, img_arr)
                else:
                    img_arr3d = np.random.randint(0, 100, (quadro_nimages, 512, 512))
                    img_arr = img_arr3d[-1].copy()
                    quadro_namenumber_entry.setText(str(filenumber + 1))
                    self.updated.emit(1)
                    acquisition_end_time = time.time()
                    save_image_h5(filename, img_arr3d)
            image_acquired = 1
            quadro_acquire_btn.setEnabled(True)
            
            
        if new_main_window.comboBox.currentText() == 'EXTG':
            t0 = time.time()
            filenumber = int(quadro_namenumber_entry.text())
            filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
            ec.setFileWriterConfig('name_pattern', filename)
            if simulator == 0:
                pulser.channel[2].set_source('TRIG')
                #glaz.startMeasurement()
                nd_task.start()
                time.sleep(0.1)
                ec.sendDetectorCommand('arm')
                pulser.channel[2].set_source('OFF')
                a = ec.fileWriterFiles()
                # print(a['value'])
                # print(filename+'_data_000001.h5')
                # while a['value'] == None or a['value'] == [] or a['value'][0] != filename+'_data_000001.h5':
                while filename+'_data_000001.h5' not in a['value']:
                    a = ec.fileWriterFiles()
                    # print(a['value'])
                    # print(filename+'_data_000001.h5')
                    if np.abs(t0-time.time()) > 10:
                        # print('RESET THE TRIGGER')
                        pulser.channel[2].set_source('TRIG')
                        pulser.channel[2].set_source('OFF')
                ec.sendDetectorCommand('disarm')
                quadro_namenumber_entry.setText(str(filenumber + 1))
                pulser.channel[2].set_source('TRIG')
                url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
                hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
                sort_imgs(hf)
                save_image_h5(filename, img_arr, img_arr_2)
            else:
                img_arr = np.random.randint(0, 100, size=(512, 512))
                img_arr_2 = np.random.randint(-100, 0, size=(512, 512))
                quadro_namenumber_entry.setText(str(filenumber + 1))
                plot_func()
                save_image_h5(filename, img_arr, img_arr_2)
            quadro_acquire_btn.setEnabled(True)

thread_1 = BThread()
thread_1.updated.connect(plot_func)


class CThread(QThread):
    updated = QtCore.pyqtSignal(int)
    started = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)

    def run(self):
        # update_run_number()
        global img_arr, img_arr_2, save_files, img_arr_old, exposure, image_acquired, quadro_frametime, quadro_nimages, acquisition_start_time, acquisition_end_time, running
        while running:
                
            if new_main_window.comboBox.currentText() == 'EXTG':
                t0 = time.time()
                filenumber = int(quadro_namenumber_entry.text())
                filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
                ec.setFileWriterConfig('name_pattern', filename)
                if simulator == 0:
                    pulser.channel[2].set_source('TRIG')
                    #glaz.startMeasurement()
                    nd_task.start()
                    time.sleep(0.1)
                    ec.sendDetectorCommand('arm')
                    pulser.channel[2].set_source('OFF')
                    a = ec.fileWriterFiles()
                    # print(a['value'])
                    # print(filename+'_data_000001.h5')
                    # while a['value'] == None or a['value'] == [] or a['value'][0] != filename+'_data_000001.h5':
                    while filename+'_data_000001.h5' not in a['value']:
                        a = ec.fileWriterFiles()
                        # print(a['value'])
                        # print(filename+'_data_000001.h5')
                        if np.abs(t0-time.time()) > 10:
                            print('RESET THE TRIGGER')
                            pulser.channel[2].set_source('TRIG')
                            pulser.channel[2].set_source('OFF')
                    ec.sendDetectorCommand('disarm')
                    quadro_namenumber_entry.setText(str(filenumber + 1))
                    pulser.channel[2].set_source('TRIG')
                    url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
                    hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
                    sort_imgs(hf)
                    # save_image_h5(filename, img_arr, img_arr_2)
                    self.updated.emit(1)
                else:
                    img_arr = gaussian_array(10000, 10, (512, 512), center_noise=0.001, sigma_noise=0.0002, amplitude_noise=0.002)
                    img_arr_2 = gaussian_array(10000, 10, (512, 512), center_noise=0., sigma_noise=0., amplitude_noise=0., sine_amp = 0)
                    quadro_namenumber_entry.setText(str(filenumber + 1))
                    time.sleep(0.3)
                    self.updated.emit(1)
                    # save_image_h5(filename, img_arr, img_arr_2)
                # quadro_acquire_btn.setEnabled(True)

thread_2 = CThread()
thread_2.updated.connect(plot_func)

def ext_live_acquire():
    global running, scan_cnt
    status = new_main_window.pushButton_13.isChecked()
    if status is True:
        #change color of button to red
        new_main_window.pushButton_13.setStyleSheet("background-color: red")
        new_main_window.pushButton_13.setText('STOP')
        running = True
        thread_2.start()
    else:
        #change color of button to green
        new_main_window.pushButton_13.setStyleSheet("background-color: lime")
        new_main_window.pushButton_13.setText('Live EXTG')
        running = False
        time.sleep(3*scan_cnt/1000)
        thread_2.terminate()
        thread_2.wait()
        # print('thread terminated')

new_main_window.pushButton_13.clicked.connect(ext_live_acquire)


def sort_imgs(hf):
    global img_arr, img_arr_2, roi, img_1, data_diode
    # print('delay is ', delay_img)
    # t0 = time.time()
    print('IN SORT_IMGS')
    data = hf['entry']['data']['data'][()]
    # print('all the data, hf[entry][data][data][()] : ', data)
    even = data[0]
    odd = data[1]
    # even_roi = np.sum(roi.getArrayRegion(even, img_1.getImageItem()))
    # odd_roi = np.sum(roi.getArrayRegion(odd, img_1.getImageItem()))
    # even = np.zeros((512, 512))
    # odd = np.ones((512, 512))
    #data_diode, scan_out = Gladz_data()
    data_diode, scan_out = ni6251_data()
    print('number of shot from the NI6251 : ', scan_out)
    
    # print(data_diode[0:15])
    # print(scan_out)
    NI6251_plot(data_diode[0:16])
    print('data from the NI6251 : ', data_diode)
    # print(np.mean(data_diode[0::2]), np.mean(data_diode[1::2]))
    if data_diode[1] > data_diode[0]:  # if odd are pump on
        print('pump is on in odd')
        # print('sum data_diode even : ', np.sum(data_diode[::2]))
        # print('sum data_diode odd : ', np.sum(data_diode[::1]))
        
        img_arr = np.flipud(np.rot90(odd))
        img_arr_2 = np.flipud(np.rot90(even))
        
        # save_image_h5(odd, even, data_diode, scan_out, delay_img)
    else:  # if even are pump on
        print('pump is on in even')
        # print('sum data_diode even : ', np.sum(data_diode[::2]))
        # print('sum data_diode odd : ', np.sum(data_diode[::1]))
        img_arr = np.flipud(np.rot90(even))
        img_arr_2 = np.flipud(np.rot90(odd))
        # print(odd_roi-even_roi)
        # print('pump is off in odd')
        # save_image_h5(even, odd, data_diode, scan_out, delay_img)
    # img_arr = np.random.randint(0, 100, (512, 512))
    time.sleep(0.5)
    # plot_func()
    # print('time to unpack and sort the image is ', time.time()-t0)



# class timer_thread(QThread):
#     def run(self):
#         global timer_progress_value
#         eee = float(new_main_window.quadro_frm_entry.text())*float(new_main_window.quadro_nfrm_entry.text())
#         timer_progress_value = 0
#         steps = 10*eee/1000
#         stepsize = 10000/steps
#         while timer_progress_value <= 10000:
#             timer_progress_value = timer_progress_value + stepsize
#             time.sleep(0.1)
#         quadro_acquire_btn.setEnabled(True)


# progress_acquisition_thread = timer_thread()

# thread_1.started.connect(progress_acquisition_thread.start)

acquisition_start_time = 0


def set_start_time():
    global acquisition_start_time
    acquisition_start_time = time.time()


acquisition_end_time = 0


def set_end_time():
    global acquisition_end_time, acquisition_start_time, quadro_frametime, quadro_nimages
    # acquisition_end_time = acquisition_start_time+get_detector_parameter('count_time')*get_detector_parameter('nimages')
    acquisition_end_time = acquisition_start_time + quadro_frametime * quadro_nimages


def interpolate_PT():
    global acquisition_start_time, acquisition_end_time, pressure_list, time_PR_list
    global temperature_A, temperature_B, time_list, sensor_A, sensor_B, lakeshore_df, LTS_position_df
    global quadro_nimages, exposure, quadro_frametime
    t0 = acquisition_start_time
    t1 = acquisition_end_time
    tA_list = np.stack(lakeshore_df["temperature_A"]).copy()
    tB_list = np.stack(lakeshore_df["temperature_B"]).copy()
    sA_list = np.stack(lakeshore_df["sensor_A"]).copy()
    sB_list = np.stack(lakeshore_df["sensor_B"]).copy()
    tT_list = np.stack(lakeshore_df["time"]).copy()
    pr_list = np.array(pressure_list.copy())
    t_pr_list = np.array(time_PR_list.copy())
    LTS_list = np.stack(LTS_position_df['LTS_position']).copy()
    t_LTS_list = np.stack(LTS_position_df['time']).copy()
    indexes_P = np.where((t_pr_list > t0-1) & (t_pr_list < t1+1))
    indexes_T = np.where((tT_list > t0-1) & (tT_list < t1+1))
    indexes_LTS = np.where((t_LTS_list > t0-1) & (t_LTS_list < t1+1))

    # new_time = np.linspace(t0, t1, int(get_detector_parameter('count_time')*get_detector_parameter('nimages')+1))
    
    # new_time = np.arange(int(get_detector_parameter('nimages')))*get_detector_parameter('count_time')+t0
    new_time = np.arange(quadro_nimages)*quadro_frametime/1000+t0
    print(f"new time {new_time}")
    print(t_pr_list-t0)
    pressure_interpolated = np.interp(new_time, t_pr_list[indexes_P], pr_list[indexes_P])
    temperature_A_interpolated = np.interp(new_time, tT_list[indexes_T], tA_list[indexes_T])
    temperature_B_interpolated = np.interp(new_time, tT_list[indexes_T], tB_list[indexes_T])
    sensor_A_interpolated = np.interp(new_time, tT_list[indexes_T], sA_list[indexes_T])
    sensor_B_interpolated = np.interp(new_time, tT_list[indexes_T], sB_list[indexes_T])
    LTS_interpolated = np.interp(new_time, t_LTS_list[indexes_LTS], LTS_list[indexes_LTS])
    # print(f"{acquisition_start_time}\n{acquisition_end_time}\n{np.array2string(new_time, precision = 13, floatmode = 'fixed')}\n{pressure_interpolated}")
    # print(f"{acquisition_start_time}\n{acquisition_end_time}\n{np.array2string(t_pr_list[indexes], precision = 13, floatmode = 'fixed')}\n{pr_list[indexes]}")
    # dictionary = {'pressure': pr_list[indexes_P], 'time_PR': t_pr_list[indexes_P]}
    # dataframe = pd.DataFrame(dictionary)
    # dataframe.to_csv('test.csv', index=False, header=True)
    return new_time, pressure_interpolated, temperature_A_interpolated, temperature_B_interpolated, sensor_A_interpolated, sensor_B_interpolated, LTS_interpolated


thread_1.started.connect(set_start_time)
thread_1.finished.connect(set_end_time)
# thread_1.updated.connect(print_pressure_list)


def acquire_image_quadro():
    quadro_acquire_btn.setEnabled(False)
    thread_1.start()
    # thread_1.wait()
    thread_1.quit()


quadro_init_btn.clicked.connect(initialize_detector)
quadro_acquire_btn.clicked.connect(acquire_image_quadro)


axis_x_rbtn = new_main_window.axis_x_rbtn
axis_y_rbtn = new_main_window.axis_y_rbtn
axis_z_rbtn = new_main_window.axis_z_rbtn
axis_th_rbtn = new_main_window.axis_th_rbtn

scan_start_entry = new_main_window.scan_start_entry
scan_stop_entry = new_main_window.scan_stop_entry
scan_step_entry = new_main_window.scan_step_entry
scan_start_btn = new_main_window.scan_start_btn
scan_stop_btn = new_main_window.scan_stop_btn
scan_act_pos = new_main_window.scan_act_pos

scan_axis = 4


def select_scan_axis():
    global scan_axis
    if axis_x_rbtn.isChecked():
        scan_axis = 1
    if axis_y_rbtn.isChecked():
        scan_axis = 2
    if axis_z_rbtn.isChecked():
        scan_axis = 3
    if axis_th_rbtn.isChecked():
        scan_axis = 4


def SimStep_move_rocking(axis, stepsize):
    global SimStep_updatestatus, Simstep_updating
    command = (str(axis).encode()+b'mr'+str(stepsize).encode()+b'\r')
    ser_simstep.write(command)
    time.sleep(0.05)
    ser_simstep.read(100)
    # if SimStep_updatestatus==0:
    Simstep_updating = 1
    SimStep_updatestatus = 1
    update_SimStep_pos(axis)
    return True


def SimStep_move_abs_rocking(axis, position):
    global SimStep_updatestatus
    command = (str(axis).encode()+b'ma'+str(position).encode()+b'\r')
    ser_simstep.write(command)
    time.sleep(0.05)
    ser_simstep.read(100)
    if SimStep_updatestatus == 0:
        SimStep_updatestatus = 1


class Scan_Thread(QThread):
    updated = QtCore.pyqtSignal(int)
    update_lbl = QtCore.pyqtSignal(int)

    def run(self):
        global rock_running, scan_axis, scan_act_pos
        start_pos = int(scan_start_entry.text())
        end_pos = int(scan_stop_entry.text())
        stepsize = int(scan_step_entry.text())
        print(scan_axis)
        SimStep_move_abs_rocking(str(scan_axis), start_pos-50)
        print('I moved to abs')
        time.sleep(0.5)
        SimStep_move_abs_rocking(str(scan_axis), start_pos)
        print('I moved to abs a second time')
        time.sleep(2)
        for i in np.arange(int(start_pos), int(end_pos), int(stepsize)):
            print(' I am in the loop pi doop')
            self.update_lbl.emit(i)
            rock_acquire_image(i)
            self.updated.emit(1)
            print(i)
            SimStep_move_rocking(str(scan_axis), str(stepsize))
            time.sleep(1)


def update_scan_lbl(i):
    scan_act_pos.setText(str(i))


def rock_acquire_image(i):
    global img_arr, save_files, img_arr_old
    #filename = str(quadro_namepattern_entry.text())
    filenumber = int(quadro_namenumber_entry.text())
    filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
    
    # expo = float(quadro_cnt_entry.text())/1000.
    # frametime = float(quadro_frm_entry.text())/1000.
    # n_frames = int(quadro_nfrm_entry.text())
    # ec.setDetectorConfig('count_time', expo)
    # ec.setDetectorConfig('frame_time', frametime)
    # ec.setDetectorConfig('ntrigger', 1)
    # ec.setDetectorConfig('nimages', int(quadro_nfrm_entry.text()))
    # ec.setFileWriterConfig('nimages_per_file', int(quadro_nfrm_entry.text()))
    ec.setFileWriterConfig('name_pattern', filename)
    ec.sendDetectorCommand('arm')
    ec.sendDetectorCommand('trigger')
    ec.sendDetectorCommand('disarm')
    time.sleep(0.5)
    timestamp = ''
    # if quadro_namepattern_rbtn.isChecked():
        # timestamp = format(time.time(), "014.3f")
    url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
    hf = None
    while hf is None:
        try:
            hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
        except Exception:
            pass
    img_arr = np.array(hf['entry']['data']['data'][0], dtype=np.uint32)
    vec = np.argwhere(img_arr > 4e9)
    img_arr[vec[:, 0], vec[:, 1]] = 0
    img_arr = np.flipud(np.rot90(img_arr))
    # new_filename = filename+"_"+format(i, "04")+"_"+timestamp+".tiff"
    save_image_h5(filename+"_"+format(i, "04")+"_"+timestamp, img_arr)


scanning_thread = Scan_Thread()
scanning_thread.updated.connect(plot_func)
scanning_thread.update_lbl.connect(update_scan_lbl)


def SimStep_rocking_curve_start():
    global rock_running
    rock_running = True
    scanning_thread.start()


def SimStep_rocking_curve_stop():
    global rock_running
    scanning_thread.terminate()
    rock_running = False
    SimStep_disconnect()
    SimStep_connect()


axis_x_rbtn.toggled.connect(select_scan_axis)
axis_y_rbtn.toggled.connect(select_scan_axis)
axis_z_rbtn.toggled.connect(select_scan_axis)
axis_th_rbtn.toggled.connect(select_scan_axis)
scan_start_btn.clicked.connect(SimStep_rocking_curve_start)
scan_stop_btn.clicked.connect(SimStep_rocking_curve_stop)



ser_lakeshore = serial.Serial()       #must be something related to temperature sensor ?
ser_lakeshore.port = 'COM9'
ser_lakeshore.baudrate = 57600
ser_lakeshore.timeout = 0
ser_lakeshore.parity = serial.PARITY_ODD
ser_lakeshore.bytesize = 7
ser_lakeshore.stopbits = 1
ser_lakeshore.xonxoff = 0
ser_lakeshore.rtscts = 0

if simulator == 0:
    ser_lakeshore.open()

data = np.array([])
pw1 = new_main_window.temperature_plot
axis = DateAxisItem(orientation='bottom')
axis.attachToPlotItem(pw1.getPlotItem())
T_A = pw1.plot()
T_A.setData(data)
T_A.setPen((180, 0, 0), width=2)
T_B = pw1.plot()
T_B.setData(data)
T_B.setPen((0, 0, 180), width=2)
pw1.setBackground([195, 195, 195, 255])
pen = pg.mkPen(color=(0, 0, 0), width=1)
pw1.plotItem.getAxis('left').setPen(pen)
pw1.plotItem.getAxis('bottom').setPen(pen)
pw1.plotItem.getAxis('left').setTextPen(pen)
pw1.plotItem.getAxis('bottom').setTextPen(pen)
pw1.setLabel('left', 'Temperature', units='K')
pw1.setLabel('bottom', 'Time', units=None, unitPrefix=None)
pw1.showGrid(x=True, y=True, alpha=0.5)

TempA_lbl = new_main_window.TempA_lbl
TempB_lbl = new_main_window.TempB_lbl
Temp_RA_lbl = new_main_window.Temp_RA_lbl
Temp_RB_lbl = new_main_window.Temp_RB_lbl
Delta_T_lbl = new_main_window.Delta_T_lbl
folder_LS_btn = new_main_window.folder_LS_btn
logging_LS_rbtn = new_main_window.logging_LS_rbtn
TempA_chkbx = new_main_window.TempA_chkbx
TempB_chkbx = new_main_window.TempB_chkbx
# sol_heater_btn = new_main_window.sol_heater_btn


def hide_traceA():
    if TempA_chkbx.isChecked() is True:
        pw1.addItem(T_A)
    else:
        pw1.removeItem(T_A)


def hide_traceB():
    if TempB_chkbx.isChecked() is True:
        pw1.addItem(T_B)
    else:
        pw1.removeItem(T_B)


TempA_chkbx.toggled.connect(hide_traceA)
TempB_chkbx.toggled.connect(hide_traceB)

folderpath_temperature = os.getcwd()


def select_folder_LS():
    global folderpath_temperature
    folderpath_temperature = QtWidgets.QFileDialog.getExistingDirectory(
        new_main_window, 'Select Folder', directory=folderpath_temperature)
    logging_LS_rbtn.setChecked(True)


folder_LS_btn.clicked.connect(select_folder_LS)


temperature_A = []
temperature_B = []
time_list = []
sensor_A = []
sensor_B = []
lakeshore_df = pd.DataFrame()
temp_busy = 0


def temp_acquire():
    global temperature_A, temperature_B, time_list, sensor_A, sensor_B, temp_busy, lakeshore_df
    temp_busy = 1
    t = time.time()
    if (len(time_list) < 21000):
        time_list.append(t)
    if simulator == 0:
        ser_lakeshore.write(b'KRDG? a\n')
        time.sleep(0.06)
        res = ser_lakeshore.read(20)
        res = float(res[:-2].decode('utf-8'))
        ser_lakeshore.write(b'SRDG? a\n')
        time.sleep(0.06)
        res1 = ser_lakeshore.read(20)
        res1 = float(res1[:-2].decode('utf-8'))

        ser_lakeshore.write(b'KRDG? b\n')
        time.sleep(0.06)
        res2 = ser_lakeshore.read(20)
        res2 = float(res2[:-2].decode('utf-8'))
        ser_lakeshore.write(b'SRDG? b\n')
        time.sleep(0.06)
        res3 = ser_lakeshore.read(20)
        res3 = float(res3[:-2].decode('utf-8'))
    else:
        res = np.random.uniform(295.0, 305.0)
        res1 = np.random.uniform(33.0, 35.0)
        res2 = np.random.uniform(295.0, 305.0)
        res3 = np.random.uniform(33.0, 35.0)
        


    if len(lakeshore_df) < 21000:
        df_t = pd.DataFrame({'time': t, 'temperature_A': res, 'sensor_A': res1, 'temperature_B': res2, 'sensor_B': res3}, index=[0])
        lakeshore_df = pd.concat([lakeshore_df, df_t], ignore_index=True)
        lakeshore_df.reset_index(drop=True, inplace=True)
    # print(lakeshore_df)
    else:
        lakeshore_df.drop(index=0, inplace=True)
        df_t = pd.DataFrame({'time': t, 'temperature_A': res, 'sensor_A': res1, 'temperature_B': res2, 'sensor_B': res3}, index=[0])
        lakeshore_df = pd.concat([lakeshore_df, df_t], ignore_index=True)
        lakeshore_df.reset_index(drop=True, inplace=True)
        # remove first row from pandas dataframe

    temp_busy = 0
    if logging_LS_rbtn.isChecked() is True:
        temp_log()


def temp_log():
    global lakeshore_df
    logfile = open(folderpath_temperature+"\\"+"temperature_log.txt", "a+")
    logfile.write("%014.3f\t%07.4f\t%07.4f\t%07.4f\t%07.4f\n" % (lakeshore_df['time'].iloc[-1], lakeshore_df['temperature_A'].iloc[-1],
                  lakeshore_df['temperature_B'].iloc[-1], lakeshore_df['sensor_A'].iloc[-1], lakeshore_df['sensor_B'].iloc[-1]))


ser_pressure = serial.Serial()
ser_pressure.port = 'COM10'
ser_pressure.baudrate = 9600
ser_pressure.timeout = 0.1

if simulator == 0:
    ser_pressure.open()
    time.sleep(1)


pw2 = new_main_window.pressure_plot


axis_pr = DateAxisItem(orientation='bottom')
axis_pr.attachToPlotItem(pw2.getPlotItem())
pressure_plot = pw2.plot()
pressure_plot.setData(data)
pressure_plot.setPen((0, 0, 180), width=2)
pressure_plot_avg = pw2.plot()
pressure_plot_avg.setData(data)
pressure_plot_avg.setPen((0, 180, 0), width=2)
pressure_plot_mean = pw2.plot()
pressure_plot_mean.setData(data)
pressure_plot_mean.setPen((180, 0, 0), width=2)
pw2.setBackground([195, 195, 195, 255])
pen = pg.mkPen(color=(0, 0, 0), width=1)
pw2.plotItem.getAxis('left').setPen(pen)
pw2.plotItem.getAxis('bottom').setPen(pen)
pw2.plotItem.getAxis('left').setTextPen(pen)
pw2.plotItem.getAxis('bottom').setTextPen(pen)
pw2.setLogMode(False, False)
pw2.setLabel('left', 'Pressure', units='mBar')
pw2.setLabel('bottom', 'Time', units=None, unitPrefix=None)
pw2.showGrid(x=True, y=True, alpha=0.5)
pw2.getAxis("left").enableAutoSIPrefix(False)
pw2.enableAutoRange(enable=False)
pw2.getAxis("left").setRange(-6, -9)
pw2.enableAutoRange(axis="x", enable=True)
pw2.enableAutoRange(axis="y", enable=True)

Pressure_lbl = new_main_window.Pressure_lbl
folder_PR_btn = new_main_window.folder_PR_btn
logging_PR_rbtn = new_main_window.logging_PR_rbtn

folderpath_pressure = os.getcwd()


def select_folder_PR():
    global folderpath_pressure
    folderpath_pressure = QtWidgets.QFileDialog.getExistingDirectory(
        new_main_window, 'Select Folder', directory=folderpath_pressure)
    logging_PR_rbtn.setChecked(True)


folder_PR_btn.clicked.connect(select_folder_PR)


pressure_list = []
pressure_acq_list = []
pressure_mean_list = []

time_PR_list = []
time_acq_PR_list = []


class pressure_acquire_class(QThread):
    def run(self):
        timer_pressure = QtCore.QTimer()
        timer_pressure.timeout.connect(pressure_acquire)
        timer_pressure.start(333)


pressure_acquire_thread = pressure_acquire_class()
pressure_acquire_thread.start()


def pressure_acquire():
    global pressure_list, time_PR_list, pw2, pressure_mean_list
    if simulator == 0:
        ser_pressure.write(b'r\n')
        time.sleep(0.01)
        res1 = 0
        res1 = ser_pressure.read(8)
        res1 = int(res1.decode('utf-8'))
        press = 4.99e-10*np.exp(2.981215e-3*res1)
    else:
        press = np.random.normal(1e-6, 1e-9)

    try:
        time_pr = time.time()

        if (len(pressure_list) < 21000):
            pressure_list.append(press)
            time_PR_list.append(time_pr)
            if len(pressure_list) > 12:
                pressure_mean_list.append(running_mean(np.array(pressure_list[-11:]), 10)[-1])
            else:
                pressure_mean_list.append(0)
        else:
            if len(pressure_list) > 0:
                pressure_list.pop(0)
                time_PR_list.pop(0)
                pressure_mean_list.pop(0)
            pressure_list.append(press)
            time_PR_list.append(time_pr)
            pressure_mean_list.append(running_mean(np.array(pressure_list[-11:]), 10)[-1])
        if logging_PR_rbtn.isChecked() is True:
            pressure_log()

    except Exception:
        print(res1)
        print("pressure error")
        ser_pressure.read(20)
        time.sleep(1)
        pass


class pressure_acquire_class(QThread):
    def run(self):
        timer_pressure = QtCore.QTimer()
        timer_pressure.timeout.connect(pressure_acquire)
        timer_pressure.start(333)


class perpetualTimer():

    def __init__(self, t, hFunction):
        self.t = t
        self.hFunction = hFunction
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.hFunction()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()


t_pressure = perpetualTimer(0.25, pressure_acquire)
t_temperature = perpetualTimer(0.25, temp_acquire)
t_temperature.start()
t_pressure.start()


def pressure_log():
    global pressure_list
    logfile = open(folderpath_pressure+"\\"+"pressure_log.txt", "a+")
    logfile.write("%014.3f\t%.3e\n" % (time.time(), pressure_list[-1]))


chopper = MC2000B('COM12')
# chopper = 0

def chopper_start():
    chopper.enable = True
    
def chopper_stop():
    chopper.enable = False
    
    

new_main_window.chopper_start_btn.clicked.connect(chopper_start)
new_main_window.chopper_stop_btn.clicked.connect(chopper_stop)
chopper_safety_box = new_main_window.chopper_safety_box 


shutter_connect_btn = new_main_window.shutter_connect_btn
shutter_disconnect_btn = new_main_window.shutter_disconnect_btn
shutter_enable_btn = new_main_window.shutter_enable_btn

# pump_frame = pg.LayoutWidget()
# pump_frame.setFixedSize(100, 80)
# pump_power_lbl = QtGui.QLabel("Pump power \n[mW]")
# pump_power_entry = QtGui.QLineEdit("0")

# pump_frame.addWidget(pump_power_lbl, 0, 0)
# pump_frame.addWidget(pump_power_entry, 1, 0)

shutter_status = 0


def shutter_connect():
    shutter_disconnect_btn.setEnabled(True)
    shutter_disconnect_btn.setStyleSheet(
        'QPushButton { background-color: red }')
    shutter_connect_btn.setEnabled(False)
    shutter_connect_btn.setStyleSheet('')
    lib_shutter.SC_Open(serialNumber_shutter)
    lib_shutter.SC_StartPolling(serialNumber_shutter, c_int(200))
    time.sleep(3)
    lib_shutter.SC_ClearMessageQueue(serialNumber_shutter)
    cycles_set = c_int(2)
    lib_shutter.SC_SetOperatingState(serialNumber_shutter, cycles_set)


def shutter_disconnect():
    lib_shutter.SC_ClearMessageQueue(serialNumber_shutter)
    lib_shutter.SC_StopPolling(serialNumber_shutter)
    lib_shutter.SC_Close(serialNumber_shutter)
    shutter_disconnect_btn.setEnabled(False)
    shutter_disconnect_btn.setStyleSheet('')
    shutter_connect_btn.setEnabled(True)
    shutter_connect_btn.setStyleSheet('QPushButton { background-color: rgb(85, 255, 0) }')


def shutter_open_close():
    global shutter_status
    if shutter_status == 0:
        shutter_enable_btn.setStyleSheet(
            'QPushButton { background-color: red }')
        shutter_status = 1
        open_SHUTTER()
    else:
        shutter_enable_btn.setStyleSheet(
            'QPushButton { background-color: rgb(85, 255, 0) }')
        shutter_status = 0
        close_SHUTTER()


shutter_status_2 = 0


def open_SHUTTER():
    global shutter_status_2
    cycles_set = c_int(1)
    shutter_status_2 = 1
    lib_shutter.SC_SetOperatingState(serialNumber_shutter, cycles_set)


def close_SHUTTER():
    global shutter_status_2

    cycles_set = c_int(2)
    lib_shutter.SC_SetOperatingState(serialNumber_shutter, cycles_set)
    shutter_status_2 = 0


path_shutter_DLL = os.getcwd()
lib_shutter = cdll.LoadLibrary(".\\DLLs\\Thorlabs.MotionControl.TCube.Solenoid.dll")
serialNumber_shutter = b"85836441"
lib_shutter.TLI_BuildDeviceList()


shutter_connect_btn.clicked.connect(shutter_connect)
shutter_disconnect_btn.clicked.connect(shutter_disconnect)
shutter_enable_btn.clicked.connect(shutter_open_close)


lib_LTS = cdll.LoadLibrary(".\\DLLs\\Thorlabs.MotionControl.IntegratedStepperMotors.dll")
serialNumber_LTS = b"45828660"
steps_LTS = 25600  # for 800 nm
# serialNumber_LTS = b"45852296"
# steps_LTS = 409600  # for 400 nm
lib_LTS.TLI_BuildDeviceList()

LTS_connect_btn = new_main_window.LTS_connect_btn
LTS_disconnect_btn = new_main_window.LTS_disconnect_btn
LTS_status_lbl = new_main_window.LTS_status_lbl
LTS_position_lbl = new_main_window.LTS_position_lbl
LTS_jogUp_btn = new_main_window.LTS_jogUp_btn
LTS_jogDw_btn = new_main_window.LTS_jogDw_btn
LTS_jogSt_entry = new_main_window.LTS_jogSt_entry
LTS_moveABS_btn = new_main_window.LTS_moveABS_btn
LTS_moveABS_entry = new_main_window.LTS_moveABS_entry
LTS_scanStart_btn = new_main_window.LTS_scanStart_btn
LTS_scanStop_btn = new_main_window.LTS_scanStop_btn
pump_on_off_chkbx = new_main_window.pump_on_off_chkbx
end_at_loop_chkbx = new_main_window.end_at_loop_chkbx

LTS_progressbar = new_main_window.LTS_progressbar
LTS_scanStart_entry = new_main_window.LTS_scanStart_entry
LTS_scanStop_entry = new_main_window.LTS_scanStop_entry
LTS_scanStep_entry = new_main_window.LTS_scanStep_entry
LTS_scanLoop_entry = new_main_window.LTS_scanLoop_entry
LTS_loadFile_btn = new_main_window.LTS_loadFile_btn
LTS_scanFromFile_rbtn = new_main_window.LTS_scanFromFile_rbtn


def LTS_connect():
    global serialNumber_LTS, steps_LTS
    if simulator == 0:
        LTS_disconnect_btn.setEnabled(True)
        LTS_disconnect_btn.setStyleSheet('QPushButton { background-color: red }')
        LTS_connect_btn.setEnabled(False)
        LTS_connect_btn.setStyleSheet('')
        serialNumber_LTS = b"45828660"
        multiplier = 1
        if new_main_window.rbtn_800.isChecked():
            serialNumber_LTS = b"45828660"
            steps_LTS = 25600  # for 800 nm
            multiplier = 1
        elif new_main_window.rbtn_400.isChecked():
            serialNumber_LTS = b"45852296"
            steps_LTS = 409600  # for 400 nm  
            multiplier = 30  
        
        lib_LTS.ISC_Open(serialNumber_LTS)
        lib_LTS.ISC_StartPolling(serialNumber_LTS, c_int(200))
        time.sleep(3)
        lib_LTS.ISC_EnableChannel(serialNumber_LTS)
        lib_LTS.ISC_ClearMessageQueue(serialNumber_LTS)
        speed = 10
        acceleration = 4
        lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(multiplier*speed*steps_LTS), c_int(multiplier*acceleration*steps_LTS))  # for 800nm
        # lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(100*steps_LTS), c_int(40*steps_LTS))  # for 400nm
        LTS_updating_thread.start()
    else:
        LTS_disconnect_btn.setEnabled(True)
        LTS_disconnect_btn.setStyleSheet('QPushButton { background-color: red }')
        LTS_connect_btn.setEnabled(False)
        LTS_connect_btn.setStyleSheet('')
        LTS_updating_thread.start()
        


def LTS_disconnect():
    LTS_updating_thread.terminate()
    # timer_LTS.stop()
    LTS_connect_btn.setEnabled(True)
    LTS_connect_btn.setStyleSheet('QPushButton { background-color: rgb(85, 255, 0) }')
    LTS_disconnect_btn.setEnabled(False)
    LTS_disconnect_btn.setStyleSheet('')
    lib_LTS.ISC_ClearMessageQueue(serialNumber_LTS)
    lib_LTS.ISC_DisableChannel(serialNumber_LTS)
    lib_LTS.ISC_StopPolling(serialNumber_LTS)
    lib_LTS.ISC_Close(serialNumber_LTS)


LTS_position = 0.0
LTS_ismoving = False


def update_LTS_pos():
    global LTS_position, LTS_ismoving
    pos_old = LTS_position
    LTS_position = lib_LTS.ISC_GetPosition(serialNumber_LTS)/steps_LTS  # for 800nm
    # LTS_position = lib_LTS.ISC_GetPosition(serialNumber_LTS)/25600. #for 800nm
    if LTS_ismoving is True and abs(pos_old - LTS_position) == 0.:
        time.sleep(0.25)
        LTS_ismoving = False


LTS_position_df = pd.DataFrame()


class update_LTS_Thread(QThread):
    updated = QtCore.pyqtSignal(int)

    def run(self):
        global LTS_position, LTS_position_df
        
        while (1 > 0):
            # LTS_position = lib_LTS.ISC_GetPosition(serialNumber_LTS)/25600.#for 800nm
            if simulator == 0:
                LTS_position = lib_LTS.ISC_GetPosition(serialNumber_LTS)/steps_LTS  # for 400nm
            else:
                LTS_position = LTS_position
            if len(LTS_position_df) < 21000:
                df_t = pd.DataFrame({'time': time.time(), 'LTS_position': LTS_position}, index=[0])
                LTS_position_df = pd.concat([LTS_position_df, df_t], ignore_index=True)
                LTS_position_df.reset_index(drop=True, inplace=True)
            else:
                LTS_position_df.drop(index=0, inplace=True)
                df_t = pd.DataFrame({'time': time.time(), 'LTS_position': LTS_position}, index=[0])
                LTS_position_df = pd.concat([LTS_position_df, df_t], ignore_index=True)
                LTS_position_df.reset_index(drop=True, inplace=True)
            self.updated.emit(1)
            time.sleep(0.25)


LTS_T0_offset = 0


def set_LTS_T0():
    global LTS_position, LTS_T0_offset
    if new_main_window.t0_btn.isChecked() is True:
        LTS_T0_offset = LTS_position
    else:
        LTS_T0_offset = 0


def update_LTS_position_lbl():
    global LTS_position, LTS_T0_offset
    # LTS_position_lbl.setText(format(LTS_position, "+9.4f"))
    LTS_position_lbl.setText(f"{LTS_position:+9.4f}\n{(LTS_position-LTS_T0_offset)*6.666:+8.2f}")


def calc_LTS_move_time():
    v0 = float(new_main_window.FS_speed_entry.text())
    a = float(new_main_window.FS_accel_entry.text())
    x0 = float(new_main_window.FS_start_entry.text())
    xf = float(new_main_window.FS_stop_entry.text())

    t1 = v0/a
    x1 = 0.5*a*t1*t1
    x2 = 2*x1
    dx = (xf-x0-x2)
    t2 = dx/v0
    new_main_window.FS_time_lbl.setText(f"{2*t1+t2:5.1f}")
    lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(int(a*steps_LTS)), c_int(int(v0*steps_LTS)))


calc_LTS_move_time()
new_main_window.FS_speed_entry.editingFinished.connect(calc_LTS_move_time)
new_main_window.FS_accel_entry.editingFinished.connect(calc_LTS_move_time)
new_main_window.FS_start_entry.editingFinished.connect(calc_LTS_move_time)
new_main_window.FS_stop_entry.editingFinished.connect(calc_LTS_move_time)

new_main_window.t0_btn.clicked.connect(set_LTS_T0)

LTS_updating_thread = update_LTS_Thread()

LTS_updating_thread.updated.connect(update_LTS_position_lbl)


scan_command = False


def move_abs_LTS(pos_to):
    global LTS_ismoving, scan_command, LTS_position
    LTS_ismoving = True
    global serialNumber_LTS
    # deviceUnit = c_int(int(25600 * pos_to)) #for 800nm
    deviceUnit = c_int(int(steps_LTS * pos_to))
    if simulator == 0:
        lib_LTS.ISC_MoveToPosition(serialNumber_LTS, deviceUnit)
    else:
        
        LTS_position = pos_to


def move_abs_LTS_buttonfunc():
    pos_to = float(LTS_moveABS_entry.text())
    move_abs_LTS(pos_to)


def JOG_UP():
    global LTS_ismoving, LTS_position
    LTS_ismoving = True
    pos_to = float(LTS_jogSt_entry.text())
    # deviceUnit = c_int(int(25600 * pos_to)) #for 800nm
    deviceUnit = c_int(int(steps_LTS * pos_to))  # for 800nm
    if simulator == 0:
        lib_LTS.ISC_MoveRelative(serialNumber_LTS, deviceUnit)
    else:
        LTS_position = LTS_position + pos_to


def JOG_DOWN():
    global LTS_ismoving, LTS_position
    LTS_ismoving = True
    pos_to = float(LTS_jogSt_entry.text())
    # deviceUnit = c_int(int(-25600 * pos_to)) #for 800nm
    deviceUnit = c_int(int(-steps_LTS * pos_to))  # for 800nm
    if simulator == 0:
        lib_LTS.ISC_MoveRelative(serialNumber_LTS, deviceUnit)
    else:
        LTS_position = LTS_position - pos_to


delays_from_file = np.zeros(10)


def load_delay_file():
    global delays_from_file
    # options = QtWidgets.QFileDialog.Options()
    # options |= QtWidgets.QFileDialog.DontUseNativeDialog
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(new_main_window, "QFileDialog.getOpenFileName()", "", "Text Files (*.txt)")
    if fileName:
        delays_from_file = np.loadtxt(fileName)


def start_timer():
    global total_time, time_moving_stage, time_acquiring, time_sorting, number_of_acquired_images, time_saving
    print('LTS scan finished in %.02f s' % total_time)
    print('time moving stage: %.02f s' % time_moving_stage)
    print('time acquiring: %.02f s' % time_acquiring)
    print('time_acquiring per image: %.02f s' % (time_acquiring/number_of_acquired_images))
    print('time sorting: %.02f s' % time_sorting)
    print('time saving: %.02f s' % time_saving)
    print('percentage moving stage: %.02f %%' % (time_moving_stage/total_time*100))
    print('percentage acquiring: %.02f %%' % (time_acquiring/total_time*100))
    print('percentage sorting: %.02f %%' % (time_sorting/total_time*100))
    print('percentage saving: %.02f %%' % (time_saving/total_time*100))
    timer_img.start(100)
    LTS_status_lbl.setText("Idle")


def LTS_moving():
    LTS_status_lbl.setText("Moving...")


def LTS_acquiring():
    LTS_status_lbl.setText("Acquiring...")


def shutter_on_color():
    shutter_enable_btn.setStyleSheet('QPushButton { background-color: red }')


def shutter_off_color():
    shutter_enable_btn.setStyleSheet('QPushButton { background-color: rgb(85, 255, 0) }')
    close_SHUTTER()


class energy_Scan_Thread(QThread):
    updated = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)

    def run(self):
        global threshold_energy
        ec.sendFileWriterCommand('clear')
        ec.setDetectorConfig('photon_energy', 40000)
        for energy in range(10000, 40000, 250):

            ec.setDetectorConfig('threshold_energy', energy)
            threshold_energy = energy
            LTS_acquire_img()
            print(energy)
            self.updated.emit(1)
        self.finished.emit(1)
        ec.setDetectorConfig('threshold_energy', 10000)


energy_scanning_thread = energy_Scan_Thread()


def energy_scan_startBtn_func():
    global allow_drawing
    timer_img.stop()
    energy_scanning_thread.start()


energy_scanning_thread.updated.connect(plot_func)
energy_scanning_thread.updated.connect(plot_image_main)
energy_scanning_thread.finished.connect(start_timer)

time_moving_stage = 0
time_acquiring = 0
time_sorting =0
time_saving = 0
number_of_acquired_images = 0
total_time = 0

def check_chopper():
    if chopper.refoutfreq == 0:
        chopper.enable=False
        chopper.enable=True
    

class LTS_Scan_Thread(QThread):
    updated = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(int)
    moving = QtCore.pyqtSignal(int)
    acquiring = QtCore.pyqtSignal(int)
    shutter_on = QtCore.pyqtSignal(int)
    shutter_off = QtCore.pyqtSignal(int)
    end_loop = QtCore.pyqtSignal(int)
    end_at_loop_chkbx.setChecked(True)

    def run(self):
        #print the time in date and hour format
        update_run_number()
        print(f'\n=============== {time.strftime("%d/%m/%Y %H:%M:%S")} ===============')
        print('LTS scan started\n')
        t0 = time.time()
        global delays_from_file, timer_img, loop_number, exposure, LTS_position,lts_prog, current_loop, imgs_on, imgs_off, time_moving_stage, time_acquiring, time_sorting, number_of_acquired_images, total_time, time_saving
        global list_of_files_per_loop, imgs_on, imgs_off, loop_delays
        imgs_on = 0
        imgs_off = 0
        loop_delays = []
        loop_max = int(LTS_scanLoop_entry.text())
        if LTS_scanFromFile_rbtn.isChecked() is False:
            all_delays = np.arange(delay_start, delay_end, delay_step)
            delay_start = float(LTS_scanStart_entry.text())
            delay_end = float(LTS_scanStop_entry.text())
            delay_step = float(LTS_scanStep_entry.text())
            bar_max = abs((delay_end-delay_start)/delay_step*(loop_max))
        else:
            all_delays = delays_from_file
            bar_max = abs(len(all_delays)*(loop_max))
        # exposure = float(quadro_cnt_entry.text())
        if pump_on_off_chkbx.isChecked():
            close_SHUTTER()
            self.shutter_off.emit(1)
        prog = 1
        time_moving_stage = 0
        time_acquiring = 0
        time_sorting = 0
        number_of_acquired_images = 0
        time_saving = 0
        for loop in range(1, loop_max+1, 1):
            list_of_files_per_loop = []
            imgs_on = 0
            imgs_off = 0
            current_loop = loop
            loop_number = format(time.time(), "014.3f")
            for delay in all_delays:
                if chopper_safety_box.isChecked():
                    check_chopper()
                # print('start')
                t_s = time.time()
                move_abs_LTS(delay)
                while True:
                    time.sleep(0.03)
                    self.moving.emit(1)
                    if np.abs(LTS_position - delay) < 0.001:
                        time.sleep(0.05)
                        break
                print('delay stage moved in %.02f s' % (time.time() - t_s))
                time_moving_stage = time_moving_stage + time.time() - t_s
                self.acquiring.emit(1)
                LTS_acquire_img()
                # print('The image was acquired')
                LTS_append_loop_images()
                # print('the image was appended')
                if pump_on_off_chkbx.isChecked():
                    open_SHUTTER()
                    self.shutter_on.emit(1)
                    time.sleep(0.1)
                    LTS_acquire_img()
                    self.acquiring.emit(1)
                    close_SHUTTER()
                    self.shutter_off.emit(1)
                lts_prog = float(prog/bar_max*100)
                # LTS_set_prog()
                prog = prog + 1
                self.updated.emit(prog)
                # print('progress bar was updated')
                number_of_acquired_images = number_of_acquired_images + 1
            ec.sendFileWriterCommand('clear')
            generate_master_file_h5()
            LTS_average_loops()
            self.end_loop.emit(1)
            
        global allow_drawing
        allow_drawing = True
        self.finished.emit(1)
        total_time = time.time() - t0
        

        if simulator == 0:
            self.shutter_off.emit(1)
            close_SHUTTER()

def generate_master_file_h5():
    global list_of_files_per_loop,run_number, base_measurement_path, images_path, run_number
    
    
    with h5py.File(os.path.join(images_path, f'r{run_number:06}_master.h5') , "a") as master_file:
        loop_count = len(master_file.keys())  # Get the current loop count

        for i, files in enumerate([list_of_files_per_loop], start=loop_count + 1):
            # Create a group for each loop
            loop_group = master_file.create_group(f"loop{i:04}")

            for file_name in files:
                # print(file_name)
                # Open each single file
                with h5py.File(file_name, "r") as single_file:
                    # Create a subgroup in the current loop group with the same name as the single file
                    #from file_name take only the name of the file without the path
                    file_name = os.path.basename(file_name)
                    subgroup = loop_group.create_group(file_name)

                    # Link datasets from the single file to the subgroup in the master file
                    for dataset_name in single_file:
                        single_dataset = single_file[dataset_name]
                        link = h5py.ExternalLink(file_name, single_dataset.name)
                        subgroup[dataset_name] = link
                        # subgroup[dataset_name] = h5py.SoftLink(single_dataset)

                        # Include the dataset in the subgroup directly as a reference to the ExternalLink
                        # subgroup.create_dataset(dataset_name, data=h5py.ExternalLink(file_name, single_dataset.name))

    # print("Files added to the master file successfully.")



LTS_scanning_Thread = LTS_Scan_Thread()

def LTS_set_prog(ev):
    global lts_prog
    LTS_progressbar.setValue(int(lts_prog)*100)
    LTS_progressbar.setFormat("%.02f %%" % lts_prog)

LTS_scanning_Thread.updated.connect(plot_func)
LTS_scanning_Thread.updated.connect(plot_image_main)
LTS_scanning_Thread.updated.connect(LTS_set_prog)
LTS_scanning_Thread.finished.connect(start_timer)
LTS_scanning_Thread.moving.connect(LTS_moving)
LTS_scanning_Thread.acquiring.connect(LTS_acquiring)
LTS_scanning_Thread.shutter_on.connect(shutter_on_color)
LTS_scanning_Thread.shutter_off.connect(shutter_off_color)

current_loop = 1

imgs_on = 0
imgs_off = 0
img_diff_avg = 0
loop_delays = []

def LTS_append_loop_images():
    global img_arr, img_arr_2, LTS_position, loop_number, imgs_on, imgs_off, current_loop, loop_delays, img_diff_avg
    if current_loop ==1:
        loop_delays.append(LTS_position)
    #if img_on_avg is not an array:
    if type(imgs_on) is int:
        #define img_on_avg as empty array with 3 dimensions as [0,m,n] and append img_arr
        imgs_on = np.zeros((1, img_arr.shape[0], img_arr.shape[1]))
        imgs_on[0,:,:] = img_arr
        imgs_off = np.zeros((1, img_arr.shape[0], img_arr.shape[1]))
        imgs_off[0,:,:] = img_arr_2
    else:
        imgs_on = np.append(imgs_on, [img_arr], axis=0)
        imgs_off = np.append(imgs_off, [img_arr_2], axis=0)

def LTS_average_loops():
    global imgs_on, imgs_off, loop_delays, images_path, run_number, current_loop
    images_path_2 = os.path.dirname(images_path)
    #remove last folder from path
    images_path_2 = os.path.join(images_path_2, "AVG")
    #create the AVG folder if it doesn't exist
    if not os.path.exists(images_path_2):
        os.mkdir(images_path_2)
    
    #create the h5 file if it doesn't exist
    if not os.path.exists(os.path.join(images_path_2, f"r{run_number:06}_averages.h5")):
        with h5py.File(os.path.join(images_path_2, f"r{run_number:06}_averages.h5"), "w") as f:
            dset = f.create_dataset("imgs_on", data=imgs_on, compression="gzip", compression_opts=1)
            dset2 = f.create_dataset("imgs_off", data=imgs_off, compression="gzip", compression_opts=1)
            #save the number of loops in the file
            f.attrs["N_loops"] = current_loop
            f.attrs["LTS_position"] = loop_delays
            f.attrs["run_number"] = run_number
            f.attrs['Delay_ps'] = np.array(loop_delays)*6.666
            avg_plot(dset[:], dset2[:], loop_delays)
    #open an h5 file as read-write mode and create a dataset with the name "imgs_on". The file must be editable and not cleared if reopened
    else:   
        with h5py.File(os.path.join(images_path_2, f"r{run_number:06}_averages.h5"), "a") as f:
            dset = f["imgs_on"]
            dset2 = f["imgs_off"]
            #sum the new images to the old ones
            dset = dset[:] + imgs_on[:]
            dset2 = dset2 + imgs_off
            #save the number of loops in the file
            #create a new attribute in the dataset with the name "N_loops" and the value of current_loop
            
            
            
            f.attrs["N_loops"] = current_loop
            f.attrs["LTS_position"] = loop_delays
            f.attrs["run_number"] = run_number
            f.attrs['Delay_ps'] = np.array(loop_delays)*6.666
            
            avg_plot(dset[:], dset2[:], loop_delays)
        

    
        

# LTS_average_scan()

# @profile
def LTS_acquire_img():
    global LTS_position, loop_number
    global img_arr, save_files, img_arr_old, img_1, ask_plot_image, img_arr_2, data_diode
    global time_acquiring, time_sorting, time_saving
    acquisition_mode = new_main_window.comboBox.currentText()
    if acquisition_mode == "INTS":
            filenumber = int(quadro_namenumber_entry.text())
            filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
            n_frames = int(new_main_window.lineEdit_2.text())

            ec.setFileWriterConfig('name_pattern', filename)
            ec.sendDetectorCommand('arm')
            ec.sendDetectorCommand('trigger')
            ec.sendDetectorCommand('disarm')
            time.sleep(0.5)
            quadro_namenumber_entry.setText(str(filenumber + 1))
            url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
            hf = None
            if simulator == 0:
                while hf is None:
                    try:
                        hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
                    except Exception:
                        hf = 1
                        pass

            if n_frames == 1:
                if simulator == 0:
                    img_arr = np.array(hf['entry']['data']['data'][0], dtype=np.int64)
                else:
                    img_arr = np.random.randint(0, 100, size=(512, 512))
                    time.sleep(1)
            else:
                if simulator == 0:
                    img_arr = np.sum(np.array(hf['entry']['data']['data'], dtype=np.int64), axis=0)
            # vec = np.argwhere(img_arr > 4e9)
            # img_arr[vec[:, 0], vec[:, 1]] = 0
            img_arr = np.flipud(np.rot90(img_arr))
            save_image_h5(filename, img_arr)
    elif acquisition_mode == 'EXTG':
        t0 = time.time()
        # print('creation of filenumber')
        filenumber = int(quadro_namenumber_entry.text())
        filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
        # print('set filename in hte QUADRO')
        ec.setFileWriterConfig('name_pattern', filename)
        if simulator == 0:
            # print('TIRG the pulser')
            pulser.channel[2].set_source('TRIG')
            t_a = time.time()
            # print('glaz start measurement')
            #glaz.startMeasurement()
            nd_task.start()
            # print('arm the quadro')
            # time.sleep(0.1)
            ec.sendDetectorCommand('arm')
            # print('Off trigger the pulser')
            pulser.channel[2].set_source('OFF')
            # print('look for the file')
            a = ec.fileWriterFiles()
            # print(a['value'])
            # print(filename+'_data_000001.h5')
            # while a['value'] == None or a['value'] == [] or a['value'][0] != filename+'_data_000001.h5':
            # print('enter the while loop')
            while filename+'_data_000001.h5' not in a['value']:
                a = ec.fileWriterFiles()
                # print(a['value'])
                # print(filename+'_data_000001.h5')
                if np.abs(t0-time.time()) > 10:
                    print('RESET THE TRIGGER')
                    pulser.channel[2].set_source('TRIG')
                    pulser.channel[2].set_source('OFF')
            # print('while loop exited')
            time_acquiring += time.time() - t_a
            print('acquiring time: ', time.time() - t_a)
            # print('disarm the quadro')
            ec.sendDetectorCommand('disarm')
            # print('change the data number')
            quadro_namenumber_entry.setText(str(filenumber + 1))
            # print('set the pulser to TRIG')
            pulser.channel[2].set_source('TRIG')
            # print('extract the h5 file')
            url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
            extract = True
            while extract:
                try:
                    hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
                    extract = False
                except Exception:
                    print('DAMN THERE WAS AN ISSUE WHILE EXTRACTING THE DATA')
            # hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
            t_sort = time.time()
            # print('start sorting the image')
            sort_imgs(hf)
            time_sorting += time.time() - t_sort
            t_saving = time.time()
            # print('save the image')
            save_image_h5(filename, img_arr, img_arr_2, data_diode)
            time_saving += time.time() - t_saving
        else:
            img_arr = np.random.randint(0, 100, size=(512, 512))
            img_arr_2 = np.random.randint(-100, 0, size=(512, 512))
            quadro_namenumber_entry.setText(str(filenumber + 1))
            # print('painting image')
            # plot_func()
            save_image_h5(filename, img_arr, img_arr_2)
            time.sleep(0.1)
        

def LTS_acquire_img_fast():
    global LTS_position, loop_number, ref_image
    global img_arr, save_files, img_arr_old, img_1, ask_plot_image

    # get_detector_status()
    # if quadro_status_lbl.text() == "ready":
    #     ec.sendDetectorCommand('disarm')
    # get_detector_status()
# if quadro_status_lbl.text() == "idle":
    filenumber = int(quadro_namenumber_entry.text())
    filename = str(quadro_namepattern_entry.text()) + format(filenumber, "05")
    # expo = 1/1000.
    # frametime = 1/1000.
    n_frames = 1
    # ec.setDetectorConfig('count_time', expo)
    # ec.setDetectorConfig('frame_time', frametime)
    # ec.setDetectorConfig('ntrigger', 1)
    # ec.setDetectorConfig('nimages', int(quadro_nfrm_entry.text()))
    # ec.setFileWriterConfig(
        # 'nimages_per_file', int(quadro_nfrm_entry.text()))
    ec.setFileWriterConfig('name_pattern', filename)
    ec.sendDetectorCommand('arm')
    ec.sendDetectorCommand('trigger')
    ec.sendDetectorCommand('disarm')
    time.sleep(0.5)
    quadro_namenumber_entry.setText(str(filenumber + 1))
    # timestamp = ''
    # timestamp = format(time.time(), "014.3f")

    url = 'http://169.254.254.1/data/'+filename+"_data_000001.h5"
    hf = None
    while hf is None:
        try:
            hf = h5py.File(io.BytesIO(requests.get(url).content), "r+")
        except Exception:
            pass
    if n_frames == 1:
        img_arr = np.array(hf['entry']['data']['data'][0], dtype=np.int64)
    else:
        img_arr = np.sum(
            np.array(hf['entry']['data']['data'], dtype=np.int64), axis=0)
    vec = np.argwhere(img_arr > 4e9)
    img_arr[vec[:, 0], vec[:, 1]] = 0
    img_arr = np.flipud(np.rot90(img_arr))
    # new_filename = filename+"_"+timestamp+".tiff"
    # metadata = get_metadata()
    ref_image = img_arr


def LTS_scan_startBtn_func():
    global allow_drawing
    timer_img.stop()
    end_at_loop_chkbx.setChecked(False)
    LTS_scanning_Thread.start()


def LTS_scan_stopBtn_func():
    global allow_drawing
    allow_drawing = True
    timer_img.start(100)
    LTS_scanning_Thread.terminate()


def LTS_scan_stop_at_loop_func():
    if end_at_loop_chkbx.isChecked() is True:
        global allow_drawing
        allow_drawing = True
        timer_img.start(100)
        LTS_scanning_Thread.terminate()
        LTS_status_lbl.setText("Idle")


def check_for_LTS_move(delay):
    global LTS_position
    while True:
        if LTS_position == delay:
            break


LTS_connect_btn.clicked.connect(LTS_connect)
LTS_disconnect_btn.clicked.connect(LTS_disconnect)
LTS_moveABS_btn.clicked.connect(move_abs_LTS_buttonfunc)
LTS_jogUp_btn.clicked.connect(JOG_UP)
LTS_jogDw_btn.clicked.connect(JOG_DOWN)
LTS_scanStart_btn.clicked.connect(LTS_scan_startBtn_func)
LTS_scanStop_btn.clicked.connect(LTS_scan_stopBtn_func)
LTS_loadFile_btn.clicked.connect(load_delay_file)
LTS_scanning_Thread.end_loop.connect(LTS_scan_stop_at_loop_func)


class LTS_Scan_Thread_Fast(QThread):
    def run(self):
        global image_acquired
        start_pos = float(new_main_window.FS_start_entry.text())
        end_pos = float(new_main_window.FS_stop_entry.text())
        loop_number = int(new_main_window.FS_loops_entry.text())
        v0 = float(new_main_window.FS_speed_entry.text())
        a = float(new_main_window.FS_accel_entry.text())
        for loop in range(loop_number):
            lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(int(4*steps_LTS)), c_int(int(10*steps_LTS)))
            move_abs_LTS(start_pos)
            while True:
                time.sleep(0.3)
                # self.moving.emit(1)
                if np.abs(LTS_position - start_pos) < 0.001:
                    time.sleep(0.5)
                    break
            lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(int(a*steps_LTS)), c_int(int(v0*steps_LTS)))
            move_abs_LTS(end_pos)
            image_acquired = 0
            # print(f'start_fast_acquire {loop}')
            acquire_image_quadro()
            while image_acquired == 0:
                time.sleep(0.1)
            lib_LTS.ISC_SetVelParams(serialNumber_LTS, c_int(int(4*steps_LTS)), c_int(int(10*steps_LTS)))


LTS_Scan_Thread_Fast_t = LTS_Scan_Thread_Fast()

# def lts_start_fast_scan():
#     LTS_Scan_Thread_Fast_t.start()
new_main_window.FS_start_btn.clicked.connect(LTS_Scan_Thread_Fast_t.start)
# new_main_window.FS_start_btn.clicked.connect(LTS_Scan_Thread_Fast_t.start)

# def get_metadata():
#     global time_PR_list, pressure_list, temperature_A, temperature_B, LTS_position, shutter_status_2, sensor_A, sensor_B
#     expo = float(quadro_cnt_entry.text())/1000.
#     return dict(timestamp=str(time_PR_list[-1]), pressure=str(pressure_list[-1]), temperature_sample=str(temperature_A[-1]), temperature_finger=str(temperature_B[-1]), delay=LTS_position, shutter=shutter_status_2, exposure_time=expo, sensor_A_ohm=sensor_A[-1], sensor_B_ohm=sensor_B[-1])

attocube_encoder_raw = 0
attocube_encoder_deg = 0
attocube_encoder_ref = 0

# if simulator == 0:
#     from pylablib.devices import Attocube
#     atc1 = Attocube.ANC300("COM7")
#     new_main_window.volt_LCD.display(atc1.get_voltage(1))
#     new_main_window.freq_LCD.display(atc1.get_frequency(1))
#     atc1.enable_axis(1, mode='stp')


# def attocube_get_capacity():
#     cap = atc1.get_capacitance(1, measure=True)
#     new_main_window.cap_LCD.display(cap)


# def attocube_set_frequency():
#     atc1.set_frequency(1, float(new_main_window.freq_entry.text()))
#     new_main_window.freq_LCD.display(atc1.get_frequency(1))


# def attocube_set_volt():
#     atc1.set_voltage(1, float(new_main_window.volt_entry.text()))
#     new_main_window.volt_LCD.display(atc1.get_voltage(1))


# def attocube_jog_up_start():
#     atc1.jog(1, '-')


# def attocube_jog_up_stop():
#     atc1.stop()


# def attocube_jog_dw_start():
#     atc1.jog(1, '+')


# def attocube_jog_dw_stop():
#     atc1.stop()


# ser_attocube_encoder = serial.Serial()
# ser_attocube_encoder.port = 'COM8'
# ser_attocube_encoder.baudrate = 115200
# ser_attocube_encoder.timeout = 0
# ser_attocube_encoder.parity = serial.PARITY_NONE
# ser_attocube_encoder.bytesize = 8
# ser_attocube_encoder.stopbits = 1
# ser_attocube_encoder.xonxoff = 0
# ser_attocube_encoder.rtscts = 0


# def attocube_encoder_connect():
#     ser_attocube_encoder.open()


# def attocube_encoder_disconnect():
#     ser_attocube_encoder.close()


# def attocube_encoder_read():
#     global attocube_encoder_raw, attocube_encoder_deg, attocube_encoder_ref
#     if(ser_attocube_encoder.isOpen()):
#         ser_attocube_encoder.write(b'r\n')
#         time.sleep(0.05)
#         try:
#             res = float(ser_attocube_encoder.read(7))
#             new_main_window.label_3.setText((format(res, ".4f")))
#             attocube_encoder_raw = res
#             attocube_encoder_deg = res*360.17-155.02+attocube_encoder_ref
#             new_main_window.label_4.setText(
#                 (format(attocube_encoder_deg, "6.2f")))
#         except Exception:
#             pass


# def attocube_encoder_set_ref():
#     global attocube_encoder_ref, attocube_encoder_deg
#     attocube_encoder_ref = -attocube_encoder_deg


# def attocube_encoder_reset_ref():
#     global attocube_encoder_ref
#     attocube_encoder_ref = 0.


# timer_encoder = QtCore.QTimer()
# timer_encoder.timeout.connect(attocube_encoder_read)

# timer_encoder.start(50)

# new_main_window.cap_btn.clicked.connect(attocube_get_capacity)
# new_main_window.set_freq_btn.clicked.connect(attocube_set_frequency)
# new_main_window.set_volt_btn.clicked.connect(attocube_set_volt)
# new_main_window.jog_up_btn.pressed.connect(attocube_jog_up_start)
# new_main_window.jog_up_btn.released.connect(attocube_jog_up_stop)
# new_main_window.jog_down_btn.pressed.connect(attocube_jog_dw_start)
# new_main_window.jog_down_btn.released.connect(attocube_jog_dw_stop)

# new_main_window.pushButton.clicked.connect(attocube_encoder_connect)
# new_main_window.pushButton_2.clicked.connect(attocube_encoder_disconnect)
# new_main_window.pushButton_3.clicked.connect(attocube_encoder_set_ref)
# new_main_window.pushButton_4.clicked.connect(attocube_encoder_reset_ref)


################ LEAK VALVE ########################

# new_main_window.actionLeak_valve.triggered.connect(window_leakvalve.show)

# ser_leak = serial.Serial()
# ser_leak.port = 'COM11'
# ser_leak.baudrate = 9600
# ser_leak.timeout = 0
# ser_leak.parity = serial.PARITY_NONE
# ser_leak.bytesize = 8
# ser_leak.stopbits = 1
# ser_leak.xonxoff = 0

# # if simulator == 0:
# #     ser_leak.open()

# leakvalve_position = 0
# window_leakvalve.pos_label.setText("0")


# def move_valve(dir):
#     global leakvalve_position
#     steps = int(window_leakvalve.steps_entry.text())
#     if dir > 0:
#         command = f"O{steps}#"
#         ser_leak.write(command.encode())
#         time.sleep(0.05)
#         ser_leak.read(100)
#         leakvalve_position = leakvalve_position + steps

#     else:
#         command = f"C{steps}#"
#         ser_leak.write(command.encode())
#         time.sleep(0.05)
#         ser_leak.read(100)
#         leakvalve_position = leakvalve_position - steps
#     window_leakvalve.pos_label.setText(f"{leakvalve_position}")


# def set_leak_speed():
#     speed = int(window_leakvalve.speed_entry.text())
#     delay = int(speed/2)
#     command = f"S{delay}#"
#     ser_leak.write(command.encode())
#     time.sleep(0.05)
#     ser_leak.read(100)

# if simulator == 0:
#     set_leak_speed()


# def leakvalve_set_sero():
#     global leakvalve_position
#     leakvalve_position = 0
#     window_leakvalve.pos_label.setText(f"{0}")


# def leakvalve_close_full():
#     global leakvalve_position
#     if leakvalve_position > 0:
#         command = f"C{leakvalve_position}#"
#         ser_leak.write(command.encode())
#         time.sleep(0.05)
#         ser_leak.read(100)
#     leakvalve_position = 0
#     window_leakvalve.pos_label.setText(f"{0}")


# window_leakvalve.speed_entry.editingFinished.connect(set_leak_speed)
# window_leakvalve.open_btn.clicked.connect(partial(move_valve, 1))
# window_leakvalve.close_btn.clicked.connect(partial(move_valve, -1))
# window_leakvalve.zero_btn.clicked.connect(leakvalve_set_sero)
# window_leakvalve.close_full_btn.clicked.connect(leakvalve_close_full)


#############################################################################


win_sol = uic.loadUi("MainWindow_SOL.ui")


def show_sol_win():
    win_sol.show()

new_main_window.actionSOL_Heater.triggered.connect(show_sol_win)
# .clicked.connect(show_sol_win)


port_sol = 'COM'+win_sol.lineEdit.text()

ser_sol = serial.Serial()
ser_sol.port = port_sol
ser_sol.baudrate = 9600
ser_sol.timeout = 0
ser_sol.parity = serial.PARITY_NONE
ser_sol.bytesize = 8
ser_sol.stopbits = serial.STOPBITS_ONE
ser_sol.xonxoff = 0
ser_sol.rtscts = False

cur_freq = 0
cur_pwr = 0
emission = 0


def update_sol():
    global cur_freq, cur_pwr, lakeshore_df
    # power in percentage
    ser_sol.write(b'8800000088\r')
    time.sleep(0.05)
    res = ser_sol.read(20)
    res2 = '0x'+str(res[2:-3])[2:-1]
    cur_pwr = int(int(res2, base=16)/40)
    win_sol.lcdNumber_2.display(cur_pwr)

    # frequency in Hz
    ser_sol.write(b'9A0000009A\r')
    time.sleep(0.05)
    res = ser_sol.read(20)
    res2 = '0x'+str(res[2:-3])[2:-1]
    cur_freq = int(res2, base=16)
    win_sol.lcdNumber.display(cur_freq/1000)

    win_sol.label_5.setText(str(lakeshore_df['temperature_A'].iloc[-1]))


def sol_power_up():
    global cur_pwr
    cur_pwr = cur_pwr + 1
    ser_sol.write(b'880200008A\r')
    time.sleep(0.05)
    _ = ser_sol.read(20)
    update_sol()


def sol_power_down():
    global cur_pwr
    cur_pwr = cur_pwr - 1
    ser_sol.write(b'8801000089\r')
    time.sleep(0.05)
    _ = ser_sol.read(20)
    update_sol()


def sol_power_up_fast():
    global cur_pwr
    pwr = cur_pwr + 10
    if (0 <= pwr <= 100):
        cur_pwr = pwr
        sol_set_power(pwr)
    update_sol()


def sol_power_down_fast():
    global cur_pwr




















def sol_power_up_pid(val):
    global cur_pwr
    pwr = cur_pwr + val
    if (0 <= pwr <= 100):
        cur_pwr = pwr
        sol_set_power(pwr)
    update_sol()


def sol_power_down_pid(val):
    global cur_pwr
    pwr = cur_pwr - val
    if (0 <= pwr <= 100):
        cur_pwr = pwr
        sol_set_power(pwr)
    update_sol()


def sol_set_power(pwr):
    p_hex = hex(pwr)
    val = '0x0400'+str(p_hex[2:].zfill(2))
    val = val[2:].zfill(6)
    p1 = val[0:2]
    p2 = val[2:4]
    p3 = val[4:6]
    xor = hex(int('0x88', 16))
    for p in [p1, p2, p3]:
        xor = hex(int(xor, 16) ^ int(p, 16))
    com = val+xor[2:].zfill(2)
    com_bytes = str.encode('88'+com.upper()+'\r')
    ser_sol.write(com_bytes)
    time.sleep(0.05)
    _ = ser_sol.read(20)
    update_sol()


def sol_cw():
    ser_sol.write(b'8C0000008C\r')
    time.sleep(0.05)
    _ = ser_sol.read(20)
    sol_set_freq(100000)
    update_sol()


def sol_qs():
    ser_sol.write(b'8C0100008D\r')
    time.sleep(0.05)
    _ = ser_sol.read(20)
    update_sol()


def connect_sol():
    ser_sol.open()
    win_sol.pushButton_8.setEnabled(False)
    win_sol.pushButton_8.setStyleSheet("")
    win_sol.pushButton_9.setEnabled(True)
    win_sol.pushButton_9.setStyleSheet("background-color: rgb(255, 0, 0);")
    win_sol.radioButton_2.toggle()
    sol_qs()
    update_sol()


def disconnect_sol():
    ser_sol.close()
    win_sol.pushButton_8.setEnabled(True)
    win_sol.pushButton_8.setStyleSheet("background-color: rgb(0, 255, 0);")
    win_sol.pushButton_9.setEnabled(False)
    win_sol.pushButton_9.setStyleSheet("")


def sol_set_freq(freq=cur_freq):

    val = hex(int(freq))[2:].zfill(6)
    p1 = val[0:2]
    p2 = val[2:4]
    p3 = val[4:6]
    xor = hex(int('0xA6', 16))
    for p in [p1, p2, p3]:
        xor = hex(int(xor, 16) ^ int(p, 16))
    com = val+xor[2:].zfill(2)
    com_bytes = str.encode('A6'+com.upper()+'\r')
    ser_sol.write(com_bytes)
    time.sleep(0.05)
    _ = ser_sol.read(20)


def sol_freq_up():
    global cur_freq
    freq = cur_freq + 1000
    if (1000 <= freq <= 100000):
        cur_freq = freq
        sol_set_freq(freq)
    update_sol()


def sol_freq_down():
    global cur_freq
    freq = cur_freq - 1000
    if (1000 <= freq <= 100000):
        cur_freq = freq
        sol_set_freq(freq)
    update_sol()


def sol_freq_up_fast():
    global cur_freq
    freq = cur_freq + 10000
    if (1000 <= freq <= 100000):
        cur_freq = freq
        sol_set_freq(freq)
    update_sol()


def sol_freq_down_fast():
    global cur_freq
    freq = cur_freq - 10000
    if (1000 <= freq <= 100000):
        cur_freq = freq
        sol_set_freq(freq)
    update_sol()


def sol_toggle_emission():
    global emission
    if emission:
        emission = 0
        ser_sol.write(b'8400000084\r')
        time.sleep(0.05)
        _ = ser_sol.read(20)
        win_sol.pushButton_4.setStyleSheet("")
    else:
        emission = 1
        ser_sol.write(b'8401000085\r')
        time.sleep(0.05)
        _ = ser_sol.read(20)
        win_sol.pushButton_4.setStyleSheet(
            "background-color: rgb(255, 170, 0);")


pid_copy = 0


def mapRange(value, inMin, inMax, outMin, outMax):
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


class pid_thread_class(QThread):

    def run(self):
        global cur_pwr, pid_copy, lakeshore_df
        v = lakeshore_df['temperature_A'].iloc[-1]
        # started = 0
        # start_pwr = cur_pwr
        P = float(win_sol.lineEdit_4.text())
        Int = float(win_sol.lineEdit_5.text()) #noqa E741
        D = float(win_sol.lineEdit_6.text())
        print('PID = ', P, Int, D)
        if pid_copy == 0:
            pid = PID(P, Int, D, output_limits=(1, 100), setpoint=float(win_sol.lineEdit_3.text()), sample_time=0.5)
            pid_copy = copy.deepcopy(pid)
        else:
            pid = pid_copy
        pid.set_auto_mode(True, last_output=cur_pwr)
        # pid.output_limits = (int(cur_pwr*0.9), int(cur_pwr*1.1))
        t = 0.01
        high = cur_pwr*1.1
        low = cur_pwr*0.9
        print(low, high)
        while True:
            if (low > 1):
                low = low-t
            else:
                low = 1
            if (high < 100):
                high = high+t
            else:
                high = 100
            pid.output_limits = (low, high)
            control = int(pid(v))
            v = lakeshore_df['temperature_A'].iloc[-1]
            sol_set_power(int(mapRange(control, 1, 100, low, high)))
            pid_copy = copy.deepcopy(pid)
            time.sleep(0.5)
            t = t + 0.01


pid_thread = pid_thread_class()

rate = 0


class sol_ramp_thread(QThread):

    def run(self):
        global cur_pwr, pid_copy, rate, lakeshore_df
        rate = float(win_sol.lineEdit_2.text())
        # startpoint_T = temperature_A[-1]
        print('I AM RUNING')
        startpoint_T = float(win_sol.lineEdit_3.text())
        v = lakeshore_df['temperature_A'].iloc[-1]
        P = float(win_sol.lineEdit_4.text())
        Int = float(win_sol.lineEdit_5.text()) #noqa E741
        D = float(win_sol.lineEdit_6.text())
        print('PID = ', P, Int, D)
        # started = 0
        high = cur_pwr*1.1
        low = cur_pwr*0.9
        t = 0.01
        pid = PID(P, Int, D, output_limits=(low, high), setpoint=startpoint_T, sample_time=1)
        pid.set_auto_mode(True)
        while True:
            if (low > 1):
                low = low-t
            else:
                low = 1
            if (high < 50):
                high = high+t
            else:
                high = 50
            win_sol.label_10.setText(format(startpoint_T, ".2f"))
            startpoint_T = startpoint_T+rate/60
            pid.setpoint = (startpoint_T)
            pid.output_limits = (low, high)
            control = int(pid(v))
            print('control = ', control)
            v = lakeshore_df['temperature_A'].iloc[-1]
            sol_set_power(control)
            pid_copy = copy.deepcopy(pid)

            time.sleep(0.5)
            t = t + 0.01


def update_sol_rate():
    global rate
    rate = float(win_sol.lineEdit_2.text())


win_sol.lineEdit_2.editingFinished.connect(update_sol_rate)

# class sol_ramp_thread(QThread):
#     def run(self):
#         rate = float(win_sol.lineEdit_2.text())
#         startpoint_T = temperature_A[-1]
#         v = temperature_A[-1]
#         P = float(win_sol.lineEdit_4.text())
#         I = float(win_sol.lineEdit_5.text())
#         D = float(win_sol.lineEdit_6.text())
#         while True:
#             win_sol.label_10.setText(format(startpoint_T, ".2f"))
#             startpoint_T = startpoint_T+rate/60

#             pid = PID(P, I, D, output_limits=(0, 100), setpoint=startpoint_T, sample_time=1)
#             pid.set_auto_mode(True)
#             control = int(pid(v))
#             v = temperature_A[-1]
#             sol_set_power(control)
#             time.sleep(0.5)


sol_ramp = sol_ramp_thread()


def sol_ramp_start_cmd():
    sol_ramp.start()
    win_sol.pushButton_14.setEnabled(True)
    win_sol.pushButton_3.setEnabled(False)


def sol_ramp_stop_cmd():
    sol_ramp.terminate()
    win_sol.pushButton_14.setEnabled(False)
    win_sol.pushButton_3.setEnabled(True)


def pid_start_cmd():
    pid_thread.start()
    win_sol.pushButton_15.setEnabled(True)
    win_sol.pushButton_7.setEnabled(False)


def pid_stop_cmd():
    pid_thread.terminate()
    win_sol.pushButton_15.setEnabled(False)
    win_sol.pushButton_7.setEnabled(True)


def reset_pid():
    global pid_copy
    pid_copy = 0


win_sol.pushButton_8.clicked.connect(connect_sol)
win_sol.pushButton_9.clicked.connect(disconnect_sol)
win_sol.pushButton_2.clicked.connect(sol_power_up)
win_sol.pushButton_6.clicked.connect(sol_power_down)
win_sol.radioButton.toggled.connect(sol_cw)
win_sol.radioButton_2.toggled.connect(sol_qs)
win_sol.pushButton.clicked.connect(sol_freq_up)
win_sol.pushButton_5.clicked.connect(sol_freq_down)
win_sol.pushButton_12.clicked.connect(sol_freq_up_fast)
win_sol.pushButton_13.clicked.connect(sol_freq_down_fast)
win_sol.pushButton_10.clicked.connect(sol_power_up_fast)
win_sol.pushButton_11.clicked.connect(sol_power_down_fast)
win_sol.pushButton_4.clicked.connect(sol_toggle_emission)
win_sol.pushButton_7.clicked.connect(pid_start_cmd)
win_sol.pushButton_15.clicked.connect(pid_stop_cmd)
win_sol.pushButton_3.clicked.connect(sol_ramp_start_cmd)
win_sol.pushButton_14.clicked.connect(sol_ramp_stop_cmd)
win_sol.lineEdit_4.editingFinished.connect(reset_pid)
win_sol.lineEdit_5.editingFinished.connect(reset_pid)
win_sol.lineEdit_6.editingFinished.connect(reset_pid)


new_main_window.actionDetector.triggered.connect(window_settings_detector.show)


def apply_settings_detector():
    ph = window_settings_detector.photon_spin.value()
    # th = window_settings_detector.threshold_spin.value()
    #ec.setDetectorConfig('photon_energy', int(ph))
    ec.setDetectorConfig('incident_energy', int(ph))  # changed
    #ec.setDetectorConfig('threshold_energy', int(th))
    window_settings_detector.close()


window_settings_detector.buttonBox.rejected.connect(window_settings_detector.close)
window_settings_detector.buttonBox.accepted.connect(apply_settings_detector)


############################################# Pulser #############################################
class FourChannelPulser:
    def __init__(self):
        self.channels = [0, 0, 0, 0]
        
    def set_source(self, mode):
        pass
    
if simulator == 0:
    rm = pyvisa.ResourceManager()
    resourceList = rm.list_resources()
    print(resourceList)

    index = 5 #sept 2025: used to be 6, changed pulser box and is now 5 (ie com15)
    print("Requested index:", index, "valid 0–", len(resourceList)-1)
    pulser = A7_pulser(resourceList[index])
else:
    pulser = FourChannelPulser()
    




def set_pulser():
    channelA = {"source": new_main_window.comboBox_5.currentText(), "divider": int(new_main_window.lineEdit_13.text()),
                "delay": float(new_main_window.lineEdit_10.text()), "width": float(new_main_window.lineEdit_11.text())}  # for controlling the glazPD
    channelB = {"source": new_main_window.comboBox_6.currentText(), "divider": int(new_main_window.lineEdit_13.text()),
                "delay": float(new_main_window.lineEdit_14.text()), "width": float(new_main_window.lineEdit_15.text())}  # for controlling the chopper
    channelC = {"source": new_main_window.comboBox_7.currentText(), "divider": int(new_main_window.lineEdit_17.text()),
                "delay": float(new_main_window.lineEdit_18.text()), "width": float(new_main_window.lineEdit_19.text())}  # for inhibiting the SRS
    channelD = {"source": new_main_window.comboBox_8.currentText(), "divider": int(new_main_window.lineEdit_21.text()),
                "delay": float(new_main_window.lineEdit_22.text()), "width": float(new_main_window.lineEdit_23.text())}  # trigger for SRS
    for i, channel in enumerate([channelA, channelB, channelC, channelD]):
        pulser.channel[i].set_source(channel['source'])
        pulser.channel[i].set_divider(channel['divider'])
        pulser.channel[i].set_delay(channel['delay'])
        pulser.channel[i].set_width(channel['width'])
    pulser.channel[2].set_source('TRIG')


new_main_window.pushButton_10.clicked.connect(set_pulser)

####################################################################################################

############################################ Gladz-PD / NI6251 ##############################################


def initialize_NI6251():
    sample_rate = 500_000 #Hz
    #samples = 500
    global nd_task, samples
    samples = int(new_main_window.lineEdit_4.text()) #samples and scan_cnt same thing
    nd_task = nd.Task()
    try:
        nd_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", terminal_config=TerminalConfiguration.DIFF)
        nd_task.timing.cfg_samp_clk_timing(sample_rate, sample_mode = AcquisitionType.FINITE, 
                                        samps_per_chan = samples) 
        nd_task.triggers.start_trigger.cfg_anlg_window_start_trig(4, 2, trigger_source="/Dev1/APFI0", 
                                        trigger_when=WindowTriggerCondition1.ENTERING_WINDOW)
    except nidaqmx.errors.DaqError as e:
        print("DAQmx error:", e)                 # human message
        print("details:", e.args)                # tuple with details
        print("code:", getattr(e, "error_code", None))
    
scan_cnt =  int(new_main_window.lineEdit_4.text())
scan_cnt_out = c_int(0)
pd_number = c_int(1)  # this has to match the xml config file
pd_channel = c_int(2)  # channel 1 to get reflected beam


def set_NI6251():
    print("NI6251 setted and initialized")

def ni6251_data():
    """Read `scan_cnt` samples from the NI task `nd_task` into a NumPy array.

    Returns (data_array, n_read). Tries to use nidaqmx's fast reader to fill
    a preallocated NumPy buffer; falls back to `nd_task.read()` if necessary.
    """
    global nd_task, scan_cnt
    data = np.empty(scan_cnt, dtype=np.double)
    n_read = 0
    try:
        # Prefer the stream reader which can fill a preallocated NumPy buffer
        from nidaqmx.stream_readers import AnalogSingleChannelReader
        reader = AnalogSingleChannelReader(nd_task.in_stream)
        reader.read_many_sample_double(data, number_of_samples_per_channel=scan_cnt, timeout=10.0)
        n_read = scan_cnt
    except Exception as e:
        # Fallback: single blocking read (may return list or ndarray)
        try:
            data_sec = nd_task.read(scan_cnt)
            arr = np.asarray(data_sec, dtype=np.double)
            if arr.size >= scan_cnt:
                data[:] = arr[:scan_cnt]
                n_read = scan_cnt
            else:
                data[:arr.size] = arr
                n_read = arr.size
        except Exception as e2:
            print("ni6251_data read error:", e, e2)
            n_read = 0
    # stop the task if applicable (safe to ignore errors)
    try:
        nd_task.stop()
    except Exception:
        pass
    return data, n_read


niplt_wdg = new_main_window.NI6251_plot
niplt = niplt_wdg.plot()


def NI6251_plot(data=None):
    """Plot data from a USB-6251 (uses `ni6251_data()` if available).

    If `data` is not provided, attempts to read from the configured `nd_task`.
    """
    global nd_task, samples, scan_cnt
    # Acquire data if not provided
    if data is None:
        try:
            # preferred helper if present
            data, n = ni6251_data()
        except Exception:
            try:
                # fallback: direct read
                raw = nd_task.read(samples if 'samples' in globals() else scan_cnt)
                data = np.asarray(raw, dtype=np.double)
            except Exception as e:
                print("NI6251_plot read error:", e)
                return
    # Plot (mirror gladz_plot behavior)
    try:
        niplt.setData(data, clear=True)
    except Exception as e:
        print("NI6251_plot plot error:", e)

#old version
#new_main_window.pushButton_7.clicked.connect(initialize_Gladz)
#new_main_window.pushButton_9.clicked.connect(set_GladzPD)
#new_main_window.pushButton_11.clicked.connect(gladz_plot)

new_main_window.pushButton_7.clicked.connect(initialize_NI6251)
new_main_window.pushButton_9.clicked.connect(set_NI6251)
new_main_window.pushButton_11.clicked.connect(NI6251_plot)


####################################################################################################

# ############################################# AVG PLOT ###############################################

avgplt_wdg = new_main_window.average_plot
avgplt = avgplt_wdg.plot()
#set pen magenta
avgplt.setPen((200, 0, 200))
#set points
avgplt.setSymbol('o')
#set x axis label
avgplt_wdg.setLabel('bottom', 'LTS position', 'mm')
#set y axis label
avgplt_wdg.setLabel('left', 'Sum', 'cnts')


def avg_plot(img_on, img_off, x_axis):
    global roi, img_1, avg_on, avg_off, avg_x_axis
    avg_x_axis = x_axis
    if ROI_chkbx.isChecked():
        avg_on = []
        avg_off = []
        for i in range(img_on.shape[0]):
            avg_on.append(roi.getArrayRegion(img_on[i], img_1.getImageItem()).mean())
            avg_off.append(roi.getArrayRegion(img_off[i], img_1.getImageItem()).mean())
        avg_on = np.array(avg_on)
        avg_off = np.array(avg_off)
        if new_main_window.pushButton_16.isChecked():
            avg = avg_on - avg_off
        else:
            avg = avg_on
        avgplt.setData(avg_x_axis, avg, clear=True)

def update_avg_plot():
    global avg_on, avg_off, avg_x_axis
    try:
        if new_main_window.pushButton_16.isChecked() == False:
            avg = avg_on - avg_off
        else:
            avg = avg_on
        avgplt.setData(avg_x_axis, avg, clear=True)
    except Exception as e:
        pass   
    

new_main_window.pushButton_16.clicked.connect(update_avg_plot)


def on_close(event):
    app.closeAllWindows()


new_main_window.closeEvent = on_close

app.exec()
t_pressure.cancel()
t_temperature.cancel()
new_main_window.close()

