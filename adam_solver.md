# adam_solver
- Adam Algorithm
- SGDSolver()定义
- ComputeUpdateValue
- <img src="http://latex.codecogs.com/svg.latex?" border="0"/>
## Adam Algorithm  
  Require: <img src="http://latex.codecogs.com/svg.latex?\alpha" border="0"/> learning rate  
  Require: <img src="http://latex.codecogs.com/svg.latex?\beta_1,\beta_2\in[0,1]" border="0"/> Exponential decay rates for the moment estimates  
  Require: <img src="http://latex.codecogs.com/svg.latex?f(\theta)" border="0"/> Stochastic objective function with paramters <img src="http://latex.codecogs.com/svg.latex?\theta" border="0"/>      
  Require: <img src="http://latex.codecogs.com/svg.latex?\theta_0" border="0"/>  Initial parameter vector    
    　<img src="http://latex.codecogs.com/svg.latex?m_0\leftarrow0" border="0"/> (Initalized 1<sup>st</sup>)  
    　<img src="http://latex.codecogs.com/svg.latex?v_0\leftarrow0" border="0"/> (Initalized 2<sup>nd</sup>)  
    　<img src="http://latex.codecogs.com/svg.latex?t\leftarrow0" border="0"/> (Initalized timestep)  
    　**while** <img src="http://latex.codecogs.com/svg.latex?\theta_t" border="0"/> not converged **do**  
      　　<img src="http://latex.codecogs.com/svg.latex?t\leftarrow{t+1}" border="0"/>  
      　　<img src="http://latex.codecogs.com/svg.latex?g_t\leftarrow\nabla_{\theta}f_t(\theta_{t-1})" border="0"/> (Get gradients)  
      　　<img src="http://latex.codecogs.com/svg.latex?m_t\leftarrow\beta_1m_{t-1}+(1-\beta_1)g_t" border="0"/> (Update biased first moment estimate)  
      　　<img src="http://latex.codecogs.com/svg.latex?v_t\leftarrow\beta_2v_{t-1}+(1-\beta_2)g_t^2" border="0"/> (Update biased second raw moment estimate)  
      　　<img src="http://latex.codecogs.com/svg.latex?\hat{m}_t\leftarrow\frac{m_t}{1-\beta_1^t}" border="0"/> (Compute bias-corrected first moment estimate)  
      　　<img src="http://latex.codecogs.com/svg.latex?\hat{v}_t\leftarrow\frac{v_t}{1-\beta_2^t}" border="0"/> (Compute bias-corrected second raw moment estimate)  
      　　<img src="http://latex.codecogs.com/svg.latex?\theta_t\leftarrow\theta_{t-1}-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t+\varepsilon}}" border="0"/> (Update parameters)  
    　**end while**  
    　**return** <img src="http://latex.codecogs.com/svg.latex?\theta_t" border="0"/> (Resulting parameters)  
