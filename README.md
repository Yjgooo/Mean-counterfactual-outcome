# Mean-counterfactual-outcome
578B HW1 Q4

By checking the plot, one can see that estimator 1 has the best performance in terms of MSE. This is unsurprising, as OLS is the BLUE (and the same holds for any linear combination of the coefficients). 

The other two estimates didn't utilize the linear model assumption of the conditional Y| A=a, W=w. (though this doesn't always imply worse performance, as utilizing known model information doesn't necessarily give better estimators). 

Surprisingly, estimator 3 outperforms estimator 2 in both variance and MSE. In other words, using the estimated g_n(W) rather than the true g(W) actually improves estimation. The only cost is that it induces more bias. 