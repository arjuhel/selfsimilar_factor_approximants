# Self-similar factor approximants

This repo is my rendition of the following paper:
> Gluzman S, Yukalov VI, Sornette D, Self-similar factor approximants, Phys. Rev. E 2003; 67: 026109.

#### Background & Intent:
Out of personal curiosity I did some work on various techniques of self-similar approximants and self-similar renormalization proposed by Yukalov, Gluzman, Sornette and Yukalova in various papers. For an overview of this topic, see [3] and [4].<br>
I decided to publish some of my python exploration work for the benefit of others who might be interested in this subject.

#### Proposed code:
This Jupyter notebook replicates the results presented in example A entitled "A. Convergence to exact result".

#### Brief technical summary
The paper details how a function can be approximated from the knowledge of its asymptotic expansion only.<br>
Given a function f(x), of which all we know is its expansion:<br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=f_0\sum_{n=0}^{k}{a_n%20x^n}"/>(2)<br> 
which is only valid for x->0, how can it be approximated for finite values of x?<br>
The paper proposes to rewrite this expansion as:<br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space;f_k^*(x)\approx%20f_0\prod_{p=1}^k{(1%20+A_{kp}x)^{n_{kp}}}" />(9)<br><br>

> "The controls A_kp and n_kp are determined by expanding the approximant (9) in powers of x and comparing this expansion with the series (2). For short, this can be called a reexpansion procedure, which sometimes is also named the accuracy-through-order relationship."

In [2], it is suggested that the problem can be rewritten into:<br> 

    (B_0 = n_1 + n_2 + n_3 + n_4)
    B_1 = n_1 * A_1 + n_2 * A_2 + n_3 * A_3 + n_4 * A_4
    B_2 = n_1 * A_1^2 + n_2 * A_2^2 + n_3 * A_3^2 + n_4 * A_4^2
    B_3 = n_1 * A_1^3 + n_2 * A_2^3 + n_3 * A_3^3 + n_4 * A_4^3 
    B_4 = n_1 * A_1^4 + n_2 * A_2^4 + n_3 * A_3^4 + n_4 * A_4^4
     
Where we know the moments B_k from equation 11 (using Faa Di Bruno's equations of the nth logarithmic derivative.) and we want to solve for the values of A and n.<br>
All the authors say is that "the solutions to these equations involve Vandermonde determinants" [2]<br><br>

After some research, I implemented two methods that can be used to solve this type of system:<br><br>
1. Solving the linear equations gives a polynomial P_k(u) of which the roots are the nodes of the quadrature.<br> 
Knowing the As, then solve for the ns, which is solving a linear Vandermonde system.<br> 
My implementation of this method is<br> 
```python
def Gaussian_quadrature_nodes_weights([B_0, B_1, ..., B_k]):
```

2. Find the recurrsion coefficients of the three-term recurrence of orthogonal polynomials (Chebyshev's algorithm).<br> 
Then use the Golub-Welsch algorithm to extract the nodes and weights from the recursion coefficients.<br> 
Implementation of this method is:<br> 
```python
def GaussianQuadrature([B_0, B_1, ..., B_k]):
```

These functions are included in the numerical_analysis file.


##### Results

The authors' results presented in example A can be replicated:<br>

![alt text](https://github.com/arjuhel/selfsimilar_factor_approximants/blob/main/ssfa_exampleA.jpeg?raw=true)

![alt text](https://github.com/arjuhel/selfsimilar_factor_approximants/blob/main/ssfa_replication.jpeg?raw=true)

The green line (8 terms) exactly approximates the function at finite values of x for the sole knowledge of the expansion coefficients.

#### Requirements
Tested on Python 3.7.6<br>
Requires: quadpy, scipy, numpy, sympy, matplotlib, math, re

#### Disclaimer
I had no communication with the authors of the paper and did this work independently, outside of any organization. 

#### References
[1] Self-similar factor approximants, Gluzman S, Yukalov VI, Sornette D, Phys. Rev. E; 67: 026109. (2003)<br>
[2] Optimization of self-similar factor approximants, V.I. Yukalov and S. Gluzman, Mol. Phys. 107, 2237–2244 (2009)<br>
[3] Interplay between Approximation theory and Renormalization group, Phys. of Particles and Nuclei, 50, 141-209 (2019)<br>
[4] Introduction to the method of selfsimilar approximants, Gluzman (2018) in Computational Analysis of Structured Media<br>
