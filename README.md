# Numerical Option Pricers
Caden Lee

This is a general collection of basic numerical option pricers.
This repository largely focuses on Monte Carlo methods to price basic options. 
You may also find in here a module which contains analytical pricers for European-styled options, created to validate the Monte Carlo pricing models.

The program is structured as such:
- A `MonteCarloBase` class under the `MonteCarloModels` module which contains methods to generate univariate and multivariate random numbers
- A `__BaseVanillaOption__` class that inherits the `MonteCarloBase` parent class for Vanilla option pricing
- `SpotOption` and `FuturesOption` classes that inherit from `__BaseVanillaOption__` class, for pricing of European- and American-styled vanilla options
    - `SpotOption`: based on the Black-Scholes model
    - `FuturesOption`: based on the Black76 model

This structure allows expansion into other exotic options (for example, Spread Options), by creating an exotic option class that inherits from the `MonteCarloBase` parent class.

The Least-Squared Monte Carlo method is applied for pricing American-Styled options.

This repository is currently a work-in-progress. The author shall not be made liable for any error found in the program should this be used for production.