This repository contains a module that is intended for using a physics informed neural network (or PINN) to solve ordinary differential equations and systems of ordinary differential equations. It also contains a streamlit app with some choices
for the user to select a differential equation and initial condition(s), along with in some cases parameters for the equation. The app will then generate the solution the equation and produce plotly graph that includes some intermediate solutions
along with the final solution to the equation. The ones on the app all have analytic solutions which can be seen along with what the neural network generated.

### Approach

The idea here is that we create a collection of points in the interval that we want the solution generated for. We then use the differential equation to define a loss function.  

More specifically, suppose we have the initial value problem:

$$
F(x, y, y', \ldots, y^{(k)}) = 0
$$

$$
y(x_0) = y_0, \quad y'(x_0) = y_1, \ldots, \quad y^{(k-1)}(x_0) = y_{k-1}
$$

We then create a neural network that we will refer to as $NN$ and assume our solution has the form:

$$
y = \sum_{n=0}^{k-1} \frac{y^{(n)}(x_0)(x - x_0)^n}{n!} + (x - x_0)^k  NN(x - x_0)
$$

This is modeling the idea that if the solution is analytic at $x_0$, we can write:

$$
y = \sum_{n=0}^{\infty} \frac{y^{(n)}(x_0)}{n!}(x - x_0)^n
$$

From the initial conditions, we can separate out the terms for $y, y', \ldots, y^{(k-1)}$:

$$
y = \sum_{n=0}^{k-1} \frac{y^{(n)}(x_0)(x - x_0)^n}{n!} + \sum_{n=k}^{\infty} \frac{y^{(n)}(x_0)(x - x_0)^n}{n!}
$$

For all the terms in the second summand we can factor out $(x - x_0)^k$ to obtain:

$$
y = \sum_{n=0}^{k-1} \frac{y^{(n)}(x_0)(x - x_0)^n}{n!} + (x - x_0)^k \sum_{n=k}^{\infty} \frac{y^{(n)}(x_0)(x - x_0)^{(n-k)}}{n!}
$$

The first half of that sum is the summation in our trial solution. The second half can be thought of as $(x - x_0)^k  F(x)$, where $F$ is some function analytic at $x_0$.  
So the neural network $NN(x)$ is trying to learn to approximate $F(x)$ as a function "centered" at $x_0$.

---

#### Why not just let $y = NN(x)$ and include the initial conditions as part of the loss function?

In theory, this can work. The issue is that this focuses a large part of the loss function on the initial conditions, and it makes learning to stay along with the initial conditions while also using the differential equation difficult.  

One can attempt to balance this by weighting loss from the initial conditions compared to loss from the differential equation, but that's another hyperparameter that would have to be tuned.  

With the setup we use, we can define the loss function as:

$$
L(x) = \frac{1}{|X|} \sum_{x \in X} \big(F(x, y(x), y'(x), \ldots, y^{(k)}(x))\big)^2
$$

which is the mean square residual comparing $F$ to $0$.

### Contents of the repository
`pinn_utils/pinn.py` : contains the PINN class used and the solve function used to solve a given DE

`PINN` class creates a feed forward Neural Network with an input layers, several hidden layers and an output layer. There is an activation function specified for the input layer,
either a single activation for all the hidden layers or a list of activations, one for each layer, and an output activation layer (strongly recommended to be the identity function).

example: 
```python
NN = pinn.PINN(
  num_hidden_layers=3,
  layer_width=64,
  input_activation=nn.Tanh(),
  hidden_activation=nn.Tanh(),
  output_activation=nn.Identity(),
  num_inputs=1,
  num_outputs=1
)
```

creates a PINN intended to solve a single ODE, will create an input layer, 3 hidden layers, and an output layer. Each layer will have 64 neurons, and the activation for all layers other than 
the output layer will be tanh.

The `solve` function takes in a PINN, a differential equation F that is expected to be zero, an x that contains the values over which the equation is to be solved and initial conditions $x_0$ and $y(x_0), y'(x_0),\ldots$

For example using the `NN` defined above, suppose we wanted to solve $y' = y$, $y(0)=1$ on the interval $\[-1,1\]$:
To get the differential equation we could define 
```python
F = lambda x, y, dy : y - dy
a=0
ics=[1]
x = torch.linspace(-1,1,200).reshape(-1,1)
```
Then to get the solution we could call
```python
solution = pinn.solve(
                      F=F,
                      a=a,
                      ics=ics,
                      NN=NN,
                      x=x,
                      epochs=1000,
                      lr=1e-3
                    )
```

`pinn_utils/pinn.py` also contains some functions used internally, notably `get_y_trial` which generated a trial function as described at the top given $x_0$, the initial conditions and $NN$,
and `get_loss` which takes the initial conditions, the neural network, and the $F$ defined to create a loss function used during training

`pinn_utils/de_sols.py` : Contains analytic solutions for the example DEs in the streamlit app
`pinn_utils/ode_meta.py` : Contains a dictionary of meta data used for solving the equation in the streamlit app
Typical data includes: the order of the DE, what parameters will need to be supplied, what the $F$ used to define the DE will be, what the analytic solution (if any) is, and information about text display in the graph

`stpinn.py` is the streamlit app that demonstrates the solver and shows for comparison the analytic solution
It can be run by doing 
```bash
streamlit run stpinn.py
```

