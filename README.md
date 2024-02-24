# Mean-counterfactual-outcome
578B HW1 Q4

By checking the plot, one can see that the estimator 1 has the best performance in terms of MSE. This is unsurprising, as OLS is the BLUE (and the same holds for any linear combination of the coefficients). 
The other two estimates also didn't utilize the linear model assumption of the conditional Y| A=a, W=w (though we know that utilizing known information doesn't neccesarily lead to smaller variance). 

Estimator 2 and 3 perform very similarly. Meaning that in this specific setting the estimation of g(W) = P(A = 1|W) costs us very little. 