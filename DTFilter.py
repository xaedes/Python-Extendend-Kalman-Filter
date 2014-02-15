#!/usr/bin/env python
# The MIT License (MIT)

# Copyright (c) 2014 xaedes

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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