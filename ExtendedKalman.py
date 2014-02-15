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

import numpy as np
import numdifftools as nd   # sudo pip install numdifftools


class ExtendedKalman(object):
    """ExtendedKalman
    # http://www.cbcity.de/das-kalman-filter-einfach-erklaert-teil-2
    # http://de.wikipedia.org/wiki/Kalman-Filter
    # http://services.eng.uts.edu.au/~sdhuang/1D%20Kalman%20Filter_Shoudong.pdf
    # http://services.eng.uts.edu.au/~sdhuang/Kalman%20Filter_Shoudong.pdf
    # http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf
    #
    # self.x: Systemzustand
    #
    # self.P: Unsicherheit ueber Systemzustand
    #
    # self.F: Dynamik
    #         x,u,and return type are np.matrix
    #
    # self.J_f_fun: Jacobi Matrix fuer f als Funktion zum auswerten
    #               conversions between np.array and np.matrix are necessary because nd.Jacobian needs np.array, but we use np.matrix everywhere
    #
    # self.Q: Dynamik Unsicherheit
    #
    # self.u: externe Beeinflussung des Systems
    #
    # self.B: Dynamik der externen Einfluesse
    #
    # self.h: Messfunktion
    #         x,and return type are np.matrix
    #         function h can be used to compute the predicted measurement from the predicted state
    #         (http://www.lr.tudelft.nl/fileadmin/Faculteit/LR/Organisatie/Afdelingen_en_Leerstoelen/Afdeling_AEWE/Applied_Sustainable_Science_Engineering_and_Technology/Education/AE4-T40_Kites,_Smart_kites,_Control_and_Energy_Production/doc/Lecture5.ppt)
    #
    # self.J_h_fun: Jacobi Matrix fuer h als Funktion zum auswerten
    #               conversions between np.array and np.matrix are necessary because nd.Jacobian needs np.array, but we use np.matrix everywhere
    #
    # self.R: Messunsicherheit
    """

    def __init__(self, n_states, n_sensors):
        super(ExtendedKalman, self).__init__()
        self.n_states = n_states
        self.n_sensors = n_sensors

        # x: Systemzustand
        self.x = np.matrix(np.zeros(shape=(n_states,1)))

        # P: Unsicherheit ueber Systemzustand
        self.P = np.matrix(np.identity(n_states)) 

        # F: Dynamik
        # x,u,and return type are np.matrix
        self.f = lambda x,u: np.matrix(np.identity(n_states)) * x

        # J_f_fun: Jacobi Matrix fuer f als Funktion zum auswerten
        # conversions between np.array and np.matrix are necessary because nd.Jacobian needs np.array, but we use np.matrix everywhere
        self.J_f_fun = lambda x: np.matrix(nd.Jacobian(lambda x: self.f(x, self.u))(np.array(x)))

        # Q: Dynamik Unsicherheit
        # self.Q = np.matrix(np.zeros(shape=(n_states,n_states)))
        self.Q = np.matrix(np.identity(n_states)) 

        # u: externe Beeinflussung des Systems
        self.u = np.matrix(np.zeros(shape=(n_states,1)))

        # B: Dynamik der externen Einfluesse
        self.B = np.matrix(np.identity(n_states))

        # h: Messfunktion
        # x,and return type are np.matrix
        # function h can be used to compute the predicted measurement from the predicted state
        #  (http://www.lr.tudelft.nl/fileadmin/Faculteit/LR/Organisatie/Afdelingen_en_Leerstoelen/Afdeling_AEWE/Applied_Sustainable_Science_Engineering_and_Technology/Education/AE4-T40_Kites,_Smart_kites,_Control_and_Energy_Production/doc/Lecture5.ppt)
        self.h = lambda x: np.matrix(np.zeros(shape=(n_sensors, 1)))

        # J_h_fun: Jacobi Matrix fuer h als Funktion zum auswerten
        # conversions between np.array and np.matrix are necessary because nd.Jacobian needs np.array, but we use np.matrix everywhere
        self.J_h_fun = lambda x: np.matrix(nd.Jacobian(self.h)(np.array(x)))


        # R: Messunsicherheit
        self.R = np.matrix(np.identity(n_sensors))

        self.first = True

    # jacobi, differentiation:
    # https://code.google.com/p/numdifftools/   (numerical) easiest to use, so just use this
    # http://sympy.org/en/index.html            (symbolic)
    # http://openopt.org/FuncDesigner           (automatic, better than numerical)

    def reset(self):
        self.x *= 0

    def update(self, Z):
        '''Z: new sensor values as numpy matrix'''

        # print Z

        # w: Innovation (http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf Eq. 7)
        w = Z - np.matrix(self.h(np.array(self.x)))
        # print w
# 
        # J_h:Jacobian of function h evaluated at current x
        # conversions between np.array and np.matrix are necessary because nd.Jacobian needs np.array, but we use np.matrix everywhere
        J_h = self.J_h_fun(self.x)

        # S: Residualkovarianz (http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf Eq. 9)
        S = J_h  * self.P * J_h.getT() + self.R

        # K: Kalman-Gain (http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf Eq. 10)
        K = self.P * J_h.getT() * S.getI()


        # x: Systemzustand
        self.x = self.x + K * w

        # P: Unsicherheit der Dynamik (http://services.eng.uts.edu.au/~sdhuang/1D%20Kalman%20Filter_Shoudong.pdf Eq. 8)
        self.P = self.P - K * S * K.getT()

    def predict(self):
        # x: Systemzustand (http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf Eq. 5)
        # print self.x
        self.x = np.matrix(self.f(np.array(self.x), np.array(self.u)))
        # print self.x

        # J_f:Jacobian of function f with respect to x evaluated at current x.
        J_f = self.J_f_fun(self.x)

        # print J_f
        # print self.Q
        # sys.exit()

        # P: Unsicherheit der Dynamik (http://services.eng.uts.edu.au/~sdhuang/Extended%20Kalman%20Filter_Shoudong.pdf Eq. 6)
        self.P = J_f * self.P * J_f.getT() + self.Q
