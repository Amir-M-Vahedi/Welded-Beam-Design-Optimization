# Welded Beam Design Optimization
This repository contains the Python implementation for the optimization of a welded beam design. The project focuses on minimizing the fabrication cost of a welded beam while satisfying various structural constraints. The project is based on the principles of optimum design and utilizes advanced numerical optimization techniques to find the best beam dimensions.

<p align="center">
  <img width="500" src="https://github.com/Amir-M-Vahedi/Welded-Beam-Design-Optimization/assets/115154998/16b3a757-bc15-4971-91b2-0f1259ba9c07">
</p>
## Project Overview

The welded beam design optimization problem is a classic problem in engineering design, where the goal is to minimize the cost of production while adhering to mechanical safety and performance constraints. This optimization involves adjusting the dimensions of the welded beam—specifically the width, length, height, and thickness—to achieve the lowest cost while meeting the required mechanical constraints.

### Design Variables

- **Width of the weld (h = x1)**
- **Length of the attached part of the beam (l = x2)**
- **Height of the beam (t = x3)**
- **Thickness of the beam (b = x4)**

These dimensions affect the structural integrity and cost efficiency of the beam, influencing factors like stress, deflection, and overall stability.

### Objective Function

The objective function is formulated to minimize the total cost of the beam, incorporating material and labor costs. The cost function is mathematically represented as:

Cost = (C1 + C3) * h * l^2 + C2 * t * b * (L + t)

Here, `C1`, `C2`, and `C3` represent cost coefficients associated with materials and labor, reflecting the direct costs impacted by the beam dimensions.

### Constraints
Ensuring structural integrity and safety, the design adheres to:
1. **Shear Stress:** Must not surpass the weld material's allowable stress.
2. **Bending Stress:** Should remain within the beam material's safe limits.
3. **Buckling Load:** The buckling load should exceed the applied load conditions.
4. **Deflection:** Maximum allowable deflection should not be exceeded.
5. **Manufacturability:** Constraints on dimensions ensuring practical manufacturability.

## Optimization Approach
Utilizing `scipy.optimize.minimize` with the SLSQP (Sequential Least Squares Quadratic Programming) algorithm, this project handles nonlinear constraints effectively. For more complex constraints, `differential_evolution` is employed to ensure robustness against potential local minima.


## Results and Analysis

Optimization trials with varying initial conditions have demonstrated consistent convergence towards an optimal design, underscoring the algorithm's effectiveness. Detailed results, including convergence plots and constraint evaluations, validate the structural integrity and cost efficiency of the optimized beam designs.

![Convergencegraph](https://github.com/Amir-M-Vahedi/Welded-Beam-Design-Optimization/assets/115154998/a26d0f08-018f-49fe-834f-53a30f0aae6c)

## References

- Kenneth M. Ragsdell and Don T. Phillips. "Optimal Design of a Class of Welded Structures Using Geometric Programming." Journal of Engineering for Industry.
- Stephen P. Timoshenko and James M. Gere. Theory of Elastic Stability.
