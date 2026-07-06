#da runnare in Python 3.10.20 (conda)
import nidaqmx as nd
import matplotlib.pyplot as plt
import numpy as np
import time

from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration, Coupling, DigitalPatternCondition
from nidaqmx.constants import DigitalWidthUnits, Slope, Timescale, TriggerType, WindowTriggerCondition1

CLOCK_TERMINAL = "/Dev1/PFI0" 
sample_rate = 1000 #Hz
samples = 1000
num_secs = 0.1
data_bulk = []

with nd.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0", terminal_config=TerminalConfiguration.DIFF)
    task.timing.cfg_samp_clk_timing(rate=1000.0, source=CLOCK_TERMINAL,
    active_edge=Edge.RISING,
    sample_mode=AcquisitionType.FINITE,
    samps_per_chan=samples,
    )
    # DAQmx Start Code
    task.start()
    #for i in range(1, int(num_secs*sample_rate/samples)+1):
    for i in range(1, 1000):
        # DAQmx Read Code
        data_sec = task.read(number_of_samples_per_channel=samples, timeout=10.0)
        print(data_sec)
        # Append data
        data_bulk.append(data_sec)

    # DAQmx Stop task
    task.stop()

meas_data = np.array(data_bulk)

plt.plot(meas_data.flatten())
plt.ylabel('Simulated DAQmx')
plt.xlabel('Samples')
plt.show()

outfile = "meas_data"
np.save(outfile, meas_data)
