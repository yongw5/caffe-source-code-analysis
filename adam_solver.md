# adam_solver
- Adam Algorithm
- SGDSolver()定义
- ComputeUpdateValue
<img src="http://latex.codecogs.com/svg.latex?" border="0"/>
## Adam Algorithm
  Require: <img src="http://latex.codecogs.com/svg.latex?\alpha" border="0"/>: learning rate  
  Require: <img src="http://latex.codecogs.com/svg.latex?\beta_1,\beta_2\in[0,1]" border="0"/>Exponential decay rates for the moment estimates  
  Require: <img src="http://latex.codecogs.com/svg.latex?f(\theta)" border="0"/>: Stochastic objective function with paramters <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/>    
  Require: $\theta_0$: Initial parameter vector  
    　$m_0\leftarrow0$(Initalized 1<sup>st</sup>)  
    　$v_0\leftarrow0$(Initalized 2<sup>nd</sup>)  
    　$t\leftarrow0$(Initalized timestep)  
    　**while** $\theta_t$ not converged **do**  
      　　$t\leftarrow{t+1}$  
      　　$g_t\leftarrow\nabla_{\theta}f_t(\theta_{t-1})$ (Get gradients)  
      　　$m_t\leftarrow\beta_1m_{t-1}+(1-\beta_1)g_t$ (Update biased first moment estimate)  
      　　$v_t\leftarrow\beta_2v_{t-1}+(1-\beta_2)g_t^2$ (Update biased second raw moment estimate)  
      　　$\hat{m}_t\leftarrow\frac{m_t}{1-\beta_1^t}$ (Compute bias-corrected first moment estimate)  
      　　$\hat{v}_t\leftarrow\frac{v_t}{1-\beta_2^t}$ (Compute bias-corrected second raw moment estimate)  
      　　$\theta_t\leftarrow\theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t + \varepsilon}}$ (Update parameters)  
    　**end while**  
    　**return** $\theta_t$ (Resulting parameters)  
