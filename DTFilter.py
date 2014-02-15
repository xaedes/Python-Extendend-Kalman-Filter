#!/usr/bin/env python
from time import time


class DTFilter(object):
    """docstring for DTFilter"""
    def __init__(self,dt_estimate=0.0):
        super(DTFilter, self).__init__()

        self.last_time = None
        self.dt = dt_estimate
        self.gain = 0.2

    def get_time(self):
        return time()
        
    def update(self):
        if(self.last_time == None):
            self.last_time = self.get_time()
            return self.dt
        
        time_now = self.get_time()
        dt_now = time_now - self.last_time
        self.last_time=time_now

        # wenn neuer wert ganz stark abweicht vom alten, schwaeche den gain factor ab
        confidence = pow(1./2000.,abs(dt_now - self.dt))
        self.dt = confidence * sel
        return self.dt