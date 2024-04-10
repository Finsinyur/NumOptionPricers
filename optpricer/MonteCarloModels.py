import numpy as np
from copy import deepcopy

class MonteCarloBase:
    """
    Base class for Monte Carlo simulations.

    Attributes:
        N_sims (int): Number of simulations.
        antithetic (bool): Flag indicating whether to use antithetic variance reduction technique.
        M_steps (int): Number of time steps.

    """
    def __init__(self, N_sims: int, antithetic: bool, M_steps: int):
        """
        Initializes MonteCarloBase with simulation parameters.

        Args:
            N_sims (int): Number of simulations.
            antithetic (bool): Flag indicating whether to use antithetic variance reduction technique.
            M_steps (int): Number of time steps.

        """
        self.N_sims = N_sims
        self.antithetic = antithetic
        self.M_steps = M_steps

    def __univariate_random_gen__(self, vol, dt, fixed_seed = 20240229):
        """
        Generates univariate random numbers.

        Args:
            vol: Volatility.
            dt: Time step.
            fixed_seed (int, optional): Seed for random number generation. Defaults to 20240229.

        Returns:
            numpy.ndarray: Array of simulated Brownian motion.

        """
        if fixed_seed:
            np.random.seed(fixed_seed)

        N_sims = int(self.N_sims/2) if self.antithetic else self.N_sims

        simulated_array = np.random.normal(scale = vol, 
                                           size = (N_sims, self.M_steps))
        
        if self.antithetic:
            simulated_brownians =\
            (
                np.sqrt(dt)
                * np.concatenate([simulated_array, -1*simulated_array])
            )

        else:
            simulated_brownians = np.sqrt(dt) * simulated_array

        return simulated_brownians
    
    def __multivariate_random_gen__(self, cov_matrix, dt, fixed_seed = 20240229):
        """
        Generates multivariate random numbers.

        Args:
            cov_matrix: Covariance matrix.
            dt: Time step.
            fixed_seed (int, optional): Seed for random number generation. Defaults to 20240229.

        Returns:
            numpy.ndarray: Array of simulated Brownian motion.

        """
        if fixed_seed:
            np.random.seed(fixed_seed)

        N_sims = int(self.N_sims/2) if self.antithetic else self.N_sims

        assert cov_matrix.shape[0] == cov_matrix.shape[1],\
            f"cov_matrix must be a square matrix! Current dimension: {cov_matrix.shape}."
        
        dimension = cov_matrix.shape[0]

        means = np.zeros(dimension)
        simulated_tensor = np.random.multivariate_normal(means,
                                                         cov_matrix,
                                                         (N_sims, self.M_steps))
        simulated_brownians = []

        for i in range(dimension):
            if self.antithetic:
                sim_tensor = np.concatenate(simulated_tensor[:,:,i],
                                            -1 * simulated_tensor[:,:,i])
                simulated_brownians.append(np.sqrt(dt) * sim_tensor)
            else:
                simulated_brownians.append(np.sqrt(dt) * simulated_tensor[:,:,i])

        return np.array(simulated_brownians)
    

def LSMC(option, simulated_price_paths):
    """
    Performs Least Squares Monte Carlo simulation.

    Args:
        option: Option object.
        simulated_price_paths: Array of simulated asset price paths.

    Returns:
        numpy.ndarray: Matrix of cashflows.

    """

    MC_payoffs =\
        np.round(
                np.maximum(option.optpayoff * (simulated_price_paths - option.K), 0),
                4
            )
    
    itm_price_paths = deepcopy(simulated_price_paths)
    itm_price_paths[MC_payoffs == 0] = 0

    cashflow_matrix = np.zeros((MC_payoffs.shape[0],1))
    cashflow_matrix = MC_payoffs[:, -1]
    cashflow_matrix[np.where(MC_payoffs[:, -1] < 0)] = 0

    for i in range(MC_payoffs.shape[1] - 1, 0, -1):
        Y = cashflow_matrix * np.exp(-option.r / 252)
        X = itm_price_paths[:, i-1]
        itm = np.where(X!=0)
        Y_ = Y[itm]
        X_ = X[itm]

        if X_.size > 0:
            model = np.polyfit(X_, Y_, 2)
            pf = np.poly1d(model)

            continuation_value = pf(X)
            mask = np.ones(continuation_value.shape[0], dtype=bool)
            mask[itm] = False
            continuation_value[mask] = 0

            exercise = np.where(MC_payoffs[:,i-1] > continuation_value)

            cashflow_matrix[exercise] = MC_payoffs[exercise, i-1]

    return cashflow_matrix