from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import redpitaya_scpi as scpi

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Red Pitaya signal")
        rp_s = scpi.scpi(sys.argv[1])
        self.timer = QtCore.QTimer()
        self.timer.setInterval(300)
        #call the function (update_plot_data)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        # setting pause function
        self.is_paused = False
        # setting the range for the y-axis
        self.graphWidget.setYRange(-1.4,1.4,padding =0)
        # show grid on the graph
        self.graphWidget.showGrid(x=True,y=True)
        # receive the data from channel 1 of the Red Pitaya
        #rp_s.tx_txt('ACQ:DEC 64')
        #rp_s.tx_txt('ACQ:TRIG:LEVEL -100')

        rp_s.tx_txt('ACQ:START')
        rp_s.tx_txt('ACQ:TRIG EXT_NE')
        rp_s.tx_txt('ACQ:TRIG:DLY 8192')
        #get the data at channel 1 (ultrasonic sensor)
        rp_s.tx_txt('ACQ:SOUR1:DATA?')
        buff_string = rp_s.rx_txt()
        buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
        buff = list(map(float, buff_string))
        #set the background of the graph
        self.graphWidget.setBackground('w')

        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line = self.graphWidget.plot(buff, pen=pen)

    def update_plot_data(self):
        rp_s = scpi.scpi(sys.argv[1])
        #rp_s.tx_txt('ACQ:DEC 64')
        #rp_s.tx_txt('ACQ:TRIG:LEVEL -100')
        rp_s.tx_txt('ACQ:START')
        rp_s.tx_txt('ACQ:TRIG EXT_NE')
        rp_s.tx_txt('ACQ:TRIG:DLY 8192')
        rp_s.tx_txt('ACQ:SOUR1:DATA?')
        buff_string = rp_s.rx_txt()
        buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
        buff = list(map(float, buff_string))

        self.data_line.setData(buff)  # Update the data.
        #print(buff_string)

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())