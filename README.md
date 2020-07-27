# SEIHRD
Numerical code for optimal control of SEIHRD epidemic model 

This code is provided as a supplement to the paper titled "Optimal Control of COVID-19 as a Compartmental Disease Model" by Aaron Z. Palmer Zelda B. Zabinsky and Shan Liu.

All the code was written by Aaron Z. Palmer. Any questions on the content of the paper may be addressed to the corresponding author, Aaron Z. Palmer, at azp@math.ubc.ca. Please do not contact with requests to extend this code or any solicitations.

This program is free software: you can redistribute it and/or modify it under the terms of the BSD-3-Clause license.

# Abstract
The COVID-19 pandemic has posed a policy making crisis where efforts to slow down or end the pandemic conflict with economic priorities.  This paper provides mathematical analysis of optimal disease control policies with idealized compartmental models for disease propagation and simplistic models of social and economic costs.  The optimal control strategies are categorized as 'suppression' and 'mitigation' strategies and are analyzed in both deterministic and stochastic models.  In the stochastic model, vaccination at an uncertain time is taken into account.

# To Run
The only code dependencies are NumPy, MatPlotLib, and JSON.

The following scripts can be run in the command line with % python script_name.py

SEIHRD_control.py runs the optimization of the control problem with input from a specified .json file and output to a similar file.   Console output details the convergence of the algorithm, and a plot is generated of the results (Figures 5 and 6 using deterministic_suppession.json and deterministic_mitigation.json respectively).

SEIHRD_value.py runs the dynamic programming algorithm for the reduced SIR problem.  Output is made to a .json file.

value_plot.py runs a stochastic simulation run using the output of SEIHRD_value.py and plots the results (Figures 8 and 9 with stochastic_suppression.json and stochastic_mitigation.json respectively).

SEIHRD_plot.py simply runs a deterministic simulation of the SEIHRD model (Figures 2 and 3).

SEIHRD_R_plot.py generates a plot of the final value for a range of R values (Figure 4).

R_e_plots.py takes the output of SEIHRD_control.py and compares the R_e values (Figure 7).

value_beta_plot.py plots the beta values of the outputs from SEIHRD_value.py (Figures 10 and 12)