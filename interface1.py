from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from operator import itemgetter
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import redpitaya_scpi as scpi
import numpy as np
import pickle
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Red Pitaya signal")

        # setGeometry(left, top, width, height)
        #self.setGeometry(100, 100, 600, 400)
        rp_s = scpi.scpi(sys.argv[1])

        # creating the pause button on toolbar
        toolbar = QToolBar("My main toolbar")
        self.addToolBar(toolbar)
        self.button_action = QAction("Pause", self)
        self.button_action.setStatusTip("Pausing the DAQ process")
        self.button_action.triggered.connect(self.pauseDAQ)
        self.button_action.setCheckable(True)
        toolbar.addAction(self.button_action)
        self.setStatusBar(QStatusBar(self))

        # configure the timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(300)

        # when the timer timeout, call out the function (update_plot_data)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()

        # set up the graph widget and the layout
        self.graphWidget = pg.PlotWidget()
        self.graphWidget2 = pg.PlotWidget()
        #self.setCentralWidget(self.graphWidget)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.graphWidget)
        hbox.addWidget(self.graphWidget2)

        self.label = QLabel()
        self.label.setFont(QtGui.QFont("Arial", 36, QtGui.QFont.Black))

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.label)
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)
        # Setting the layout

        # setting the graphWidget properties
        # setting the range for the y-axis
        self.graphWidget.setYRange(-1.4,1.4,padding =0)
        self.graphWidget2.setYRange(-1.4, 1.4, padding=0)
        # show grid on the graph
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget2.showGrid(x=True, y=True)
        # set the background of the graph
        self.graphWidget.setBackground('w')
        self.graphWidget2.setBackground('w')

        # receive the data from channel 1 of the Red Pitaya
        rp_s.tx_txt('ACQ:DEC 64')
        rp_s.tx_txt('ACQ:TRIG:LEVEL -21000')
        rp_s.tx_txt('ACQ:START')
        rp_s.tx_txt('ACQ:TRIG EXT_NE')
        rp_s.tx_txt('ACQ:TRIG:DLY 8192')

        #get the data at channel 1 (ultrasonic sensor)
        rp_s.tx_txt('ACQ:SOUR1:DATA?')

        buff_string = rp_s.rx_txt()
        buff_string = buff_string.strip('ERR!{}\n\r').replace("  ", "").split(',')
        buff = list(map(float, buff_string))


        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line = self.graphWidget.plot(buff, pen=pen)
        self.data_line2 = self.graphWidget2.plot(buff, pen=pen)
        self.update()
        self.show()

    def update_plot_data(self):
        rp_s = scpi.scpi(sys.argv[1])
        #seting the trigger for receiving the data from ultrasonic sensor
        rp_s.tx_txt('ACQ:DEC 64')
        rp_s.tx_txt('ACQ:TRIG:LEVEL -21000')
        rp_s.tx_txt('ACQ:START')
        rp_s.tx_txt('ACQ:TRIG EXT_NE')
        rp_s.tx_txt('ACQ:TRIG:DLY 8192')
        rp_s.tx_txt('ACQ:SOUR1:DATA?')
        buff_string = rp_s.rx_txt()
        buff_string = buff_string.strip('ERR!{}\n\r').replace("  ", "").split(',')
        self.buff = list(map(float, buff_string))
        # cut off the input sgnale (3000 samples)
        self.signal = self.buff[3000:]
        # find the max value of the reflective signal
        self.max_index = self.signal.index(max(self.signal))
        startPoint = self.max_index - 650
        self.reflectSignal = self.signal[startPoint:(startPoint+3000)]

        if max(self.signal)>0.07:
            sample_data = np.array(self.reflectSignal)
            rms = np.sqrt(np.mean(sample_data ** 2))
            var = np.var(sample_data)
            cov = np.cov(sample_data, bias=True)
            fft = max(abs(np.fft.fft(np.abs(sample_data))))
            # set up some variables for data process
            self.feed_rms = []
            self.feed_var = []
            self.feed_fft = []
            self.feed_data = []

            self.feed_rms.append(rms)
            self.feed_var.append(var)
            self.feed_fft.append(fft)
            self.feed_data = self.feed_rms + self.feed_var + self.feed_fft
            #print(self.feed_data)
            filename = 'knn_python.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict([self.feed_data])
            self.label.setText("Receiving Signal")
            if result[0] == 'wall':
                self.label.setText("Object Detected")
                self.label.setStyleSheet("background-color: blue")
            elif result[0] == 'obj':
                self.label.setText("Wall Detected")
                self.label.setStyleSheet("background-color: red")
            elif result[0] == 'human':
                self.label.setText("Human Detected")
                self.label.setStyleSheet("background-color: yellow")
        else:
            self.label.setText("No object detected")
            self.label.setStyleSheet("background-color: white")
        # Update the data.
        self.data_line.setData(self.buff)
        self.data_line2.setData(self.reflectSignal)


    def pauseDAQ(self, s):
        # if the pause button is pressed, change the backgorund color and pause the timer
        if s == True:
            self.timer.stop()
            self.statusBar().showMessage("Paused",5000)
            self.button_action.setStatusTip("Resume the DAQ process")
        #if the pause button is not pressed, change the backgorund back to default and start the timer
        else:
            self.timer.start()
            self.statusBar().showMessage("Running", 5000)


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())