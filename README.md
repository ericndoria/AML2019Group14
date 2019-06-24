# AML2019Group14
This repository includes code, text and graphical output to demonstrate and showcase your understanding of gradient descent.  The calculations are implemented in a Python class.

**Why is gradient descent important in machine learning?**  
Gradient descent (GD) is important in machine learning as it enables optimisation of functions that describe various phenomena (either theoretical or real world). 

**How does plain vanilla gradient descent work?**  
Plain vanilla GD works by minimising the function based on a learning rate hyperparameter and the gradient of the function. On computing the gradient, the parameter to be updated is adjusted in the opposite direction as the gradient (hence gradient 'descent'). The adjustment is computed as the learning rate multiplied by the gradient. This adjustment is then subtracted from the latest parameter value. It is therefore important to select an appropriate learning rate, otherwise the minimum obtained may not be close to the true minimum of the function. This is described in the formulae below:  

update = learning_rate * gradient_of_parameters  
parameters = parameters - update

**Two modifications to plain vanilla gradient descent.**   
