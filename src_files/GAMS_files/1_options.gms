options

// Solution gap: the first one reached will end iteration
optca = 0       // Absolute gap between the found solution and the best possible solution
optcr = 0.0004  // Relative gap between the found solution and the best possible solution

solvelink = %Solvelink.Loadlibrary%  // Solvelink controls how the problem is passed from GAMS to the solver. Loadlibrary constant means that the model is passed in core without the use of temporary files.

*    profile = 8       // Profile will show the execution speed of statements at the defined depth within loops.

*    bratio = 0.25     // How large share of the candidate elements need to be found for advanced basis in LP problems. Default 0.25.
*    solveopt = merge  // How solution values are stored after multiple solves. Default merge.
*    savepoint = 1     // NOTE! Savepoint is controlled by Backbone model options.

threads = 1           // How many cores the solver can use: 0 = all cores; negative values = all cores - n. One wins in schedule runs only: dual simplex, which uses one, finishes every daily solve, and parallel runs need the other cores. An investment run takes more with run_model.py --threads. docs/running-the-model.md

solprint = Silent  // silent = no equations to backbone.lst, off = first rows of equations, on = full equations


