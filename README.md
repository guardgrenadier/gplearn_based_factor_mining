# gplearn_based_factor_mining
This project uses genetic programming (via gplearn) to discover technical factors from stock price and volume data. Several changes are made to the gplearn files in order to pass some tests for customized functions and to allow a new parameter in the SymbolicRegressor, which is crucial for calculating cross-section RankIC and using it as fitness in genetic programming.

Therefore, the demo can only work properly after you **overwrite your gplearn with the files uploaded in './overwrite_your_gplearn'**!

I also developed my own factor testing tool and uploaded it, with a purpose to learn and practice during working out the code. You may ignore them if you wish to use your own factor testing tool or existing packages like alphalens.

## About the demo
The demo demonstrates the project's ability to discover technical factors using CSI 1000 Index components.

To begin with, you should unzip the 'data.7z', which contains the price and volume data of CSI 1000 Index AND its components from 2014 to 2024. Then run 'generate_features.py' to generate necessary features for factor mining and neutralization.

Next, you may run the 'factor_mining.py'. The default parameters in 'factor_mining.py' are designed as a demo, since the key parameters(population_size, generations) are small and only 2 years of data are used in training, in order to reduce the time required. However, it still takes roughly 10 minutes to produce a factor.

The 'factor_mining.py' should print 5 factors with highest fitnesses, you may pass one or several factor expression to the list 'factor_expression_set' in 'factor_production.py' to create factor values of stocks.

After the productions of the factor data, you may again pass factor expressions to the list 'factor_test_set' in 'factor_test.py' to test the factor. The 'factor_test.py' will produce 2 graphs: 1 graph illustrating the results of the stratified backtest of the factor, and 1 graph evaluating the performance of each quantile portfolio in the stratified backtest. Note that the average RankIC, IC_IR are printed in your terminal. 

## Things to know before putting into practice
You may be surprised by how long the demo takes, and therefore wondering how much time will it take to actually use it. In my notebook, it takes about 3 hours to run the 'factor_mining .py', with the following parameters: population_size=1000, tournament_size=50, generations=5; and using data from 2014-2021.

There's an **important** parameter to notice, that is, the **'n_jobs'** in SymbolicRegressor. In my case with 16G RAM, the 'n_jobs' MUST NOT be greater than 4, or a memory error will happen. Currently the RAM bottleneck happened during calculating fitness of programs, or to be more exact, the **neutralization process** since a large amount of matrixs used in neutralization are created, which is also the process which takes most of the time. Unfortunately I have been failed to work out a solution. 

Another improvement you may try is writing more time-series customized functions. I only produced basic ones so that much information are still left undiscovered.
