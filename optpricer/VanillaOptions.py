import numpy as np
from .utils import *
from .MonteCarloModels import *


class __BaseVanillaOption__(MonteCarloBase):
    """
    Base class for pricing vanilla options using Monte Carlo simulation.

    Args:
        vol (float): Volatility of the underlying asset.
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        exp_date (str): Expiry date of the option in 'YYYY-MM-DD' format.
        optpayoff (OptPayoff): Payoff function of the option.
        opttype (OptType): Type of the option (0 for European, 1 for American, 2 for Asian, 3 for Dummy).
        N_sims (int): Number of simulations.
        antithetic (bool): Flag indicating whether to use antithetic variance reduction technique.

    Attributes:
        vol (float): Volatility of the underlying asset.
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        exp_date (numpy.datetime64): Expiry date of the option.
        today (numpy.datetime64): Current date.
        opttype (OptType): Type of the option.
        M_steps (int): Number of time steps until expiry.

    Raises:
        ValueError: If the expiry date is earlier than the current date.

    """
    def __init__(self, vol: float, K: float, r: float, 
                 exp_date: str, optpayoff: OptPayoff, opttype: OptType,
                 N_sims: int, antithetic: bool):
        
        # Initialize parameters        
        self.vol = vol
        self.K = K
        self.r = r

        # Convert expiry date to numpy.datetime64
        self.exp_date = np.datetime64(exp_date)
        self.today = np.datetime64('today')
        self.opttype = opttype
        
        # Calculate number of time steps until expiry
        if self.exp_date > self.today:
            M_steps =\
            (
                np.busday_count(self.today, self.exp_date)
                if self.opttype != 0 else 1
            )
        else:
            raise ValueError('Invalid expiry date of option (exp_date < today)!')
        
        # Initialize payoff type
        self.optpayoff = optpayoff

        # Call parent class (MonteCarloBase) constructor
        super().__init__(N_sims, antithetic, M_steps)

    def generate_asset_price_paths(self, P, fixed_seed = 20240229):
        """
        Generate asset price paths using Brownian motion.

        Args:
            P (str): Name of the underlying asset price variable.
            fixed_seed (int): Seed for random number generation.

        Returns:
            numpy.ndarray: Matrix of asset price paths.
        """

        # Calculate time step
        dt = 1/252 if self.opttype != 0 else np.busday_count(self.today, self.exp_date)/252
        # Generate Brownian motion
        simulated_brownians = self.__univariate_random_gen__(self.vol, dt, fixed_seed)

        if self.opttype == 0:
            # Non-path dependent
            X_T = self.increment_scheme(np.log(self.__dict__[P]), dt, simulated_brownians)
            return np.exp(X_T)
        else:
            # Path dependent
            paths = np.zeros([self.N_sims, self.M_steps + 1])
            paths[:,0] = np.log(self.__dict__[P])

            for j in range(1, self.M_steps + 1):
                paths[:, j]= self.increment_scheme(paths[:, j-1], dt, simulated_brownians[:, j-1])

            return np.exp(paths)
        
    def generate_payoffs(self, P, fixed_seed = 20240229):
        """Generate option payoffs based on asset price paths.

        Args:
            P (str): Name of the underlying asset price variable.
            fixed_seed (int): Seed for random number generation.

        Returns:
            numpy.ndarray: Array of option payoffs.

        Raises:
            ValueError: If the option type is not supported.

        """
        
        # Generate Monte Carlo price paths
        MC_price_paths = self.generate_asset_price_paths(P, fixed_seed=fixed_seed)

        # Compute option payoffs based on option type
        if self.opttype == 0 or self.opttype == 3:
            # European-styled options
            MC_payoffs = np.maximum(self.optpayoff * (MC_price_paths[:,-1] - self.K), 0)
            return MC_payoffs
        elif self.opttype == 1:
            # American-styled options
            return LSMC(self, MC_price_paths)
        else:
            raise ValueError(f"Option Type {self.opttype} currently not supported.")
        
    def __price__(self, P):
        """
        Compute the price of the option.

        Args:
            P (str): Name of the underlying asset price variable.

        Returns:
            float: Price of the option.

        """
        if self.opttype == 1:
            value = np.exp(-self.r/252) * np.mean(self.generate_payoffs(P))
        else:
            T = (self.exp_date - self.today).astype(np.float64)/365
            value = np.exp(-self.r * T) * np.mean(self.generate_payoffs(P))
        
        return value
    
    def __delta__(self, P):
        """
        Compute the delta of the option.

        Args:
            P (str): Name of the underlying asset price variable.

        Returns:
            float: Delta of the option.

        """
        # Define small change in asset price
        diff = 0.01

        # Remove unnecessary keys from the dictionary
        dict_shift = deepcopy(self.__dict__)
        ref_price = dict_shift[P]

        for key in ['today', 'M_steps']:
            dict_shift.pop(key, None)

        # Compute option price with slightly increased and decreased asset price
        dict_shift[P] = (1+diff)*ref_price
        optPlus = self.__class__(**dict_shift).price
        dict_shift[P] = (1-diff)*ref_price
        optMinus = self.__class__(**dict_shift).price

        # Compute delta using finite differencing
        return (optPlus - optMinus) / (2 * diff * ref_price)

    def __gamma__(self, P):

        # Define small change in asset price
        diff = 0.01

        # Remove unnecessary keys from the dictionary
        dict_shift = deepcopy(self.__dict__)
        ref_price = dict_shift[P]

        for key in ['today', 'M_steps']:
            dict_shift.pop(key, None)

        # Compute option price with slightly increased and decreased asset price
        dict_shift[P] = (1+diff)*ref_price
        optPlus = self.__class__(**dict_shift).delta
        dict_shift[P] = (1-diff)*ref_price
        optMinus = self.__class__(**dict_shift).delta

        # Compute delta using finite differencing
        return (optPlus - optMinus) / (2 * diff * ref_price)
    
class SpotOption(__BaseVanillaOption__):
    """
    Class for pricing spot options using Monte Carlo simulation.

    Args:
        S (float): Current price of the underlying asset.
        vol (float): Volatility of the underlying asset.
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        d (float): Dividend rate of the underlying asset.
        exp_date (str): Expiry date of the option in 'YYYY-MM-DD' format.
        optpayoff (OptPayoff): Payoff function of the option.
        opttype (OptType): Type of the option (0 for European, 1 for American, 3 for Bermudan).
        N_sims (int, optional): Number of simulations. Defaults to 100.
        antithetic (bool, optional): Flag indicating whether to use antithetic variance reduction technique. Defaults to True.

    Attributes:
        S (float): Current price of the underlying asset.
        d (float): Dividend rate of the underlying asset.

    Raises:
        AssertionError: If S or vol is negative.

    """
    def __init__(self, S: float, vol: float, K: float, r: float, d: float, exp_date: str, 
                 optpayoff: OptPayoff, opttype: OptType, N_sims: int = 100, antithetic: bool = True):
        """
        Initializes SpotOption with option parameters.

        Args:
            S (float): Current price of the underlying asset.
            vol (float): Volatility of the underlying asset.
            K (float): Strike price of the option.
            r (float): Risk-free interest rate.
            d (float): Dividend rate of the underlying asset.
            exp_date (str): Expiry date of the option in 'YYYY-MM-DD' format.
            optpayoff (OptPayoff): Payoff function of the option.
            opttype (OptType): Type of the option (0 for European, 1 for American, 3 for Bermudan).
            N_sims (int, optional): Number of simulations. Defaults to 100.
            antithetic (bool, optional): Flag indicating whether to use antithetic variance reduction technique. Defaults to True.

        """
        
        assert S >= 0 and vol >= 0,\
            f"Negative values not allowed for S and vol! Provided S: {S}, vol: {vol}."
        
        self.S = S
        self.d = d
        super().__init__(vol, K, r, exp_date, optpayoff, opttype, N_sims, antithetic)

    def increment_scheme(self, S_minus1, dt, simulated_brownians):
        """
        Increment scheme for asset price path generation.

        Args:
            S_minus1: Previous asset price.
            dt: Time step.
            simulated_brownians: Simulated Brownian motion.

        Returns:
            numpy.ndarray: Updated asset price.

        """
        S =\
        (
            S_minus1
            + (self.r - self.d - 0.5 * self.vol * self.vol) * dt
            + simulated_brownians
        )

        return S
       
    @property
    def price(self):
        """
        Price of the spot option.

        Returns:
            float: Option price.

        """    
        return self.__price__('S')
    
    @property 
    def delta(self):
        """
        Delta of the spot option.

        Returns:
            float: Option delta.

        """
        return self.__delta__('S')
    
    @property 
    def gamma(self):
        return self.__gamma__('S')
    
class FuturesOption(__BaseVanillaOption__):
    def __init__(self, F: float, vol: float, K: float, r: float, d: float, exp_date: str, 
                 optpayoff: OptPayoff, opttype: OptType, N_sims: int = 100, antithetic: bool = True):
        
        assert F >= 0 and vol >= 0,\
            f"Negative values not allowed for F and vol! Provided F: {F}, vol: {vol}."
        
        self.F = F
        self.d = d
        super().__init__(vol, K, r, exp_date, optpayoff, opttype, N_sims, antithetic)

    def increment_scheme(self, F_minus1, dt, simulated_brownians):
        F =\
        (
            F_minus1
            - 0.5 * self.vol * self.vol * dt
            + simulated_brownians
        )

        return F
       
    @property
    def price(self):        
        return self.__price__('F')
    
    @property 
    def delta(self):
        return self.__delta__('F')
    
    @property 
    def gamma(self):
        return self.__gamma__('F')

if __name__ == '__main__':
    S = 100
    K = 100
    vol = 0.25
    r = 0.05
    d = 0.0
    exp_date = '2024-05-10'
    optpayoff = OptPayoff.put
    opttype = 0
    test1 = SpotOption(S, vol, K, r, d, exp_date, optpayoff, opttype, N_sims=100000)
    print(test1.price)
    print(test1.delta)
    print(test1.gamma)

    test2 = FuturesOption(S, vol, K, r, d, exp_date, optpayoff, opttype, N_sims=100000)
    print(test2.price)
    print(test2.delta)
    print(test2.gamma)

    #python3 -m optpricer.VanillaOptions
