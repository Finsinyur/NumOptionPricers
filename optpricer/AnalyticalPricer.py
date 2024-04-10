import numpy as np
import math

from optpricer.utils import OptPayoff
from .utils import *

   

class EuropeanOption:

    def __init__(self, vol: float, K: float, r: float, 
                 exp_date: str, optpayoff: OptPayoff):
        
        exp_date = np.datetime64(exp_date)
        today = np.datetime64('today')
        self.T = np.busday_count(today, exp_date) / 252
        T_cal = (exp_date - today).astype(np.float64) / 365

        assert vol >= 0,\
            f"Negative values not allowed for F and vol! Provided vol: {vol}."
        
        self.r = r
        self.vol = vol
        self.K = K
        self.D = np.exp(-r * T_cal)
        self.optpayoff = optpayoff

    def cnorm(self,x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    

class Black76(EuropeanOption):

    def __init__(self, F: float, vol: float, K: float, r: float, 
                 exp_date: str, optpayoff: OptPayoff = OptPayoff.call):
        super().__init__(vol, K, r, exp_date, optpayoff)
        self.F = F

    def __get_d__(self):
        d2 =\
        (
            (np.log(self.F / self.K) - 0.5 * self.vol * self.vol *self.T)
            / (self.vol * np.sqrt(self.T))
        )

        d1 = d2 + self.vol * np.sqrt(self.T)

        return (d1, d2) if self.optpayoff == OptPayoff.call else (-d1, -d2)
    
    @property
    def price(self):
        d1, d2 = self.__get_d__()
        P = self.D * (self.F * self.cnorm(d1) - self.K * self.cnorm(d2))

        return P if self.optpayoff == OptPayoff.call else -P
    
    @property
    def delta(self):
        d1, _ = self.__get_d__()

        return self.cnorm(np.abs(d1)) if self.optpayoff == OptPayoff.call else self.cnorm(np.abs(d1)) - 1
    
class BlackScholes(EuropeanOption):
    def __init__(self, S: float, vol: float, K: float, r: float, d:float, exp_date: str, optpayoff: OptPayoff):
        super().__init__(vol, K, r, exp_date, optpayoff)
        self.S = S
        self.d = d

    def __get_d__(self):
        d2 =\
        (
            (np.log(self.S/self.K) + (self.r - self.d - 0.5 * self.vol * self.vol)*self.T)
            / (self.vol * np.sqrt(self.T))
        )
        d1 = d2 + self.vol * np.sqrt(self.T)

        return (d1, d2) if self.optpayoff == OptPayoff.call else (-d1, -d2)
    
    @property
    def price(self):
        d1, d2 = self.__get_d__()
        P = self.S * self.cnorm(d1) - self.D * self.K * self.cnorm(d2)

        return P if self.optpayoff == OptPayoff.call else -P
    
    @property
    def delta(self):
        d1, _ = self.__get_d__()

        return self.cnorm(np.abs(d1)) if self.optpayoff == OptPayoff.call else self.cnorm(np.abs(d1)) - 1

