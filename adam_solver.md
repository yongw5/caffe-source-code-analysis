# adam_solver
- [Adam Algorithm](#Adam-Algorithm)
- [AdamSolver定义](#AdamSolver定义)
- [ComputeUpdateValue](#ComputeUpdateValue)
## Adam Algorithm  
- Algorithm  
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
- Caffe  
  <img src="http://latex.codecogs.com/svg.latex?(m_t)_i=\beta_1(m_{t-1})_i+(1-\beta_1)(\nabla{L(W_t))_i}" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?(v_t)_i=\beta_2(v_{t-1})_i+(1-\beta_2)(\nabla{L(W_t)})_i^2" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?(W_{t+1})_i=(W_t)_i-\eta\frac{\sqrt{1-(\beta_2)_i^t}}{1-(\beta_1)_i^t}\frac{(m_t)_i}{\sqrt{(v_t)_i}+\varepsilon}" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?correction=\frac{\sqrt{1-(\beta_2)_i^t}}{1-(\beta_1)_i^t}" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?(W_{t+1})_i=(W_t)_i-\eta*correction\frac{(m_t)_i}{\sqrt{(v_t)_i}+\varepsilon}" border="0"/>  
## `AdamSolver`定义
`AdamSolver`继承`SGDSolver`，重写了`ComputeUpdateVaule`函数
```
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};
```
## `ComputeUpdateValue`
```
template <typename Dtype>
void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum(); // beta_1
  const Dtype beta2 = this->param_.momentum2();  // beta_2

  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();  // m , first moment vector
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get(); // v, second moment vector
  Blob<Dtype>* val_t = this->temp_[param_id].get();

  const int t = this->iter_ + 1;
  const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
      (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();  // epsilon

  switch (Caffe::mode()) {
    case Caffe::CPU: {
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
        val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());

    // \sqrt(v_t)
    caffe_powx(N,
        val_v->cpu_data(), Dtype(0.5),
        val_t->mutable_cpu_data());
    
    // \sqrt(v_t) + epsilon
    caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
    
    // \frac{m_t}{\sqrt(v_t)+epsilon}
    caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data());
    
    // \eta *correction\frac{(m_t)}{\sqrt{(v_t}+epsilon}
    caffe_cpu_scale(N, local_rate*correction,
        val_t->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    adam_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), beta1, beta2,
        eps_hat, local_rate*correction);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}
```
