import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_human = pd.read_csv('human_sum1.csv',header = None)
data_wall = pd.read_csv('wall_sum.csv', header = None)
data_obj1 = pd.read_csv('obj_sum.csv', header = None)

human_rms_value = []
human_var_value = []
human_cov_value = []
human_fft_value = []

wall_rms_value = []
wall_var_value = []
wall_cov_value = []
wall_fft_value = []

obj1_rms_value = []
obj1_var_value = []
obj1_cov_value = []
obj1_fft_value = []

#Equations for calculating the RMS, Variance, Covariance
#rms = np.sqrt(np.mean(row**2))
#var = np.var(row)
#cov = np.cov(row,bias=True)
#count_columns = len(data_human.columns)
#data_human_col = data_human.iloc[:,0]
#print(data_human_col)

for column in data_human:
    human_rms = np.sqrt(np.mean(data_human[column]**2))
    human_var = np.var(data_human[column])
    human_cov = np.cov(data_human[column], bias=False)
    human_fft = max(abs(np.fft.fft(np.abs(data_human[column]))))

    human_rms_value.append(human_rms)
    human_var_value.append(human_var)
    human_cov_value.append(human_cov)
    human_fft_value.append(human_fft)

for column in data_wall:
    wall_rms = np.sqrt(np.mean(data_wall[column] ** 2))
    wall_var = np.var(data_wall[column])
    wall_cov = np.cov(data_wall[column], bias=False)
    wall_fft = max(abs(np.fft.fft(np.abs(data_wall[column]))))

    wall_rms_value.append(wall_rms)
    wall_var_value.append(wall_var)
    wall_cov_value.append(wall_cov)
    wall_fft_value.append(wall_fft)

for column in data_obj1:
    obj1_rms = np.sqrt(np.mean(data_obj1[column] ** 2))
    obj1_var = np.var(data_obj1[column])
    obj1_cov = np.cov(data_obj1[column], bias=False)
    obj1_fft = max(abs(np.fft.fft(np.abs(data_obj1[column]))))

    obj1_rms_value.append(obj1_rms)
    obj1_var_value.append(obj1_var)
    obj1_cov_value.append(obj1_cov)
    obj1_fft_value.append(obj1_fft)
#print out the value to observing
# FFT formular
#t = data_obj1.iloc[:,1]
#sp = np.fft.fft(np.abs(t))
#freq = np.fft.fftfreq(t.shape[-1])
#print(max(abs(sp)))
#plt.plot(freq,sp.real, freq, sp.imag)
#plt.show()


# dictionary of lists
dict_human = {'RMS': human_rms_value, 'Variance': human_var_value, 'FFT Peak': human_fft_value}
dict_wall =  {'RMS': wall_rms_value, 'Variance': wall_var_value, 'FFT Peak': wall_fft_value}
dict_obj = {'RMS': obj1_rms_value, 'Variance': obj1_var_value, 'FFT Peak': obj1_fft_value}
# import to csv file
pd.DataFrame(dict_human).to_csv('human.csv')
pd.DataFrame(dict_wall).to_csv('wall.csv')
pd.DataFrame(dict_obj).to_csv('obj.csv')

# Print out the value of the feature
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
from random import randint

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        # Add Background colour to white
        self.graphWidget.setBackground('w')
        # Add Title
        self.graphWidget.setTitle("Cov Value", color="b", size="30pt")
        # Add Axis Labels
        styles = {"color": "#f00", "font-size": "20px"}
        self.graphWidget.setLabel("left", "Value", **styles)
        self.graphWidget.setLabel("bottom", "Sample", **styles)
        # Add legend
        self.graphWidget.addLegend()
        # Add grid
        #self.graphWidget.showGrid(x=True, y=True)
        # Set Range
        #self.graphWidget.setXRange(0, 10, padding=0)
        #self.graphWidget.setYRange(20, 55, padding=0)

        #self.plot( [data_human], " Input Value", 'r')
        self.plot(human_cov_value, "human",'r')
        self.plot(wall_cov_value, "wall", 'm')
        self.plot(obj1_cov_value, "obj1", 'b')

    def plot(self, y, plotname, color):
        pen = pg.mkPen(color=color)
        self.graphWidget.plot( y, name=plotname, pen=pen, symbol = None, symbolBrush=(color))

def main():
        app = QtWidgets.QApplication(sys.argv)
        main = MainWindow()
        main.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()