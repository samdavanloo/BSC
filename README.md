
This package contains the code for the paper titled "Stochastic Composition Optimization of Functions without Lipschitz Continuous Gradient" available [HERE](https://arxiv.org/abs/2207.09364).

The package contains three algorithms within the paper for the Smooth of Relelative Smooth(SoR), Reletive Smooth of Smooth(RoS), and Relative Smooth of Relative Smooth (RoR) compositions.

For any question or comment, please contact Yin Liu at liu.6630@osu.edu.


## Content
```
.
├── method_Bregman.py	        // the Bregman gradient methods of SoR and RoS problem
├── SoR_gridsearch.py           // grid search for the hyper-parameters
├── SoR_experiment.py           // code for solving SoR  
├── SoR_result_process.py       // code for figures 
├── RoS_gridsearch.py           // grid search for the hyper-parameters
├── RoS_experiment              // code for solving RoS by SGD
├── RoS_VR_experiment.py    // code for solving RoS by SPIDER
└── RoS_result_process.py       // code for figures 

```

## Requirement

The package `CVXPY` is required to solve the subproblem of the RoS composition. 

