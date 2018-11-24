# sgd_solver
- [Stochastic Gradient Descent](#Stochastic-Gradient-Descent)
- [SGDSolver()定义](#SGDSolver()定义)
- [GetLearningRate](#GetLearningRate)
- [ClipGradients](#ClipGradients)
- [ApplyUpdate](#ApplyUpdate)
- [Normalize](#Normalize)
- [Regularize](#Regularize)
- [ComputeUpdateValue](#ComputeUpdateValue)
## Stochastic Gradient Descent
- Basic
  - Without Regularization  
    <img src="http://latex.codecogs.com/svg.latex?L(W)=\frac{1}{N}\sum_{i=1}^{N}{L_i(f(x_i,W),y_i)}" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?w_{k,l}^{t+1}=w_{k,l}^{t}+\Delta{w_{k,l}^{t+1}}" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?\Delta{w_{k,l}^{t+1}}=-\eta\frac{\partial{L}}{\partial{w_{k,l}}}" border="0"/>  
  - With Regularization  
    <img src="http://latex.codecogs.com/svg.latex?L(W)=\frac{1}{N}\sum_{i=1}^{N}{L_i(f(x_i,W),y_i)}+\lambda{R(W)}" border="0"/>  
  - L1 Regularization  
    <img src="http://latex.codecogs.com/svg.latex?R(W)=\Sigma_k\Sigma_l|w_{k,l}|" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?\frac{\partial{R(W)}}{\partial{w_{k,l}}}=sign(w_{k,l})" border="0"/>  
  - L2 Regualrization  
    <img src="http://latex.codecogs.com/svg.latex?R(W)=\Sigma_k\Sigma_l{w_{k,l}^2}" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?\frac{\partial{R(W)}}{\partial{w_{k,l}}}=2w_{k,l}" border="0"/>  
- SGD with momentum  
  <img src="http://latex.codecogs.com/svg.latex?\Delta{w_{k,l}^{t+1}}=-\eta\frac{\partial{L}}{\partial{w_{k,l}}}-\alpha\Delta{w_{k,l}^{t}}" border="0"/>  
- SGD with weight decay  
  <img src="http://latex.codecogs.com/svg.latex?\Delta{w_{k,l}^{t+1}}=-\eta\frac{\partial{L}}{\partial{w_{k,l}}}-\lambda\eta{w_{k,l}^{t}}" border="0"/>  
- Combine momentum and weight decay  
  <img src="http://latex.codecogs.com/svg.latex?\Delta{w_{k,l}^{t+1}}=-\eta\frac{\partial{L}}{\partial{w_{k,l}}}-\alpha\Delta{w_{k,l}^{t}}-\lambda\eta{w_{k,l}^{t}}" border="0"/>  
where, <img src="http://latex.codecogs.com/svg.latex?\eta" border="0"/> is learning rate; <img src="http://latex.codecogs.com/svg.latex?\alpha" border="0"/> is momentum and usually set to 0.9; <img src="http://latex.codecogs.com/svg.latex?\lambda" border="0"/> is weight decay
- SGD Solver Function  
  <img src="http://latex.codecogs.com/svg.latex?w_{k,l}^{t+1}=w_{k,l}^{t}-\eta\frac{\partial{L}}{\partial{w_{k,l}}}-\alpha\Delta{w_{k,l}^{t}}-\lambda\eta{w_{k,l}^{t}}" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?w_{k,l}^{t+1}=w_{k,l}^{t}-ComputeUpdateValue(param_{id},rate)" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?Regularize(param_{id})=\frac{\partial{L}}{\partial{w_i}}+\lambda{w_{k,l}^{t}}" border="0"/>  
  <img src="http://latex.codecogs.com/svg.latex?ComputeUpdateValue(param_{id},rate)=\eta{Regularize(param_{id})}+\alpha\Delta{w_{k,l}^{t}}" border="0"/>  
## SGDSolver()定义
```
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }
  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();  // 初始化history_，update_，temp_
  Dtype GetLearningRate();  // 获取当前迭代学习率
  virtual void ApplyUpdate();  // 更新权重
  virtual void Normalize(int param_id);  // 归一化
  virtual void Regularize(int param_id);  // 正则化
  virtual void ComputeUpdateValue(int param_id, Dtype rate);  // 计算权重更新值
  virtual void ClipGradients();  // 防止梯度太大，梯度削减
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history_保留了历史momentum数据。 update_维护更新相关数据。 temp维护计算渐变/更新时可能需要的其他信息.
  // 在PreSolve中初始化
  // shape net_params[i]->shape()
  // history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  // update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  // temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};
```
## GetLearningRate
返回当前迭代次数的学习率  
|lr policies|return value|  
|:---|:---|  
|fixed|base_lr|  
|step|base_lr * gamma ^ (floor(iter / step))|  
|exp|base_lr * gamma ^ iter|  
|inv|base_lr * (1 + gamma * iter) ^ (- power)|  
|multistep| similar to step |  
|poly|base_lr * (1 - iter/max_iter) ^ (power)|  
|sigmoid|base_lr * ( 1 / (1 + exp(-gamma * (iter - stepsize))))|
```
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}
```
## ClipGradients
```
template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }  // clip_gradients<0，不进行
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {  // 所有网络的梯度平方和
    sumsq_diff += net_params[i]->sumsq_diff();  // sumsq_diff() 梯度的平方
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {  // 梯度爆炸
    Dtype scale_factor = clip_gradients / l2norm_diff;  // 削减系数
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);  // 全部参数削减
    }
  }
}
```
## ApplyUpdate
```
template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << this->iter_
        << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);  // 归一化
    Regularize(param_id);  // 正则化
    ComputeUpdateValue(param_id, rate);  // 计算更新值
  }
  this->net_->Update();  // 更新权重
}
```
## Normalize
当`iter_size `大于1时，对weight进行平均处理
```
template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {  // Delta(W) / iter_size
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
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
## Regularize
- L1 Regularization  
    <img src="http://latex.codecogs.com/svg.latex?R(W)=\Sigma_k\Sigma_l|w_{k,l}|" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?\frac{\partial{R(W)}}{\partial{w_{k,l}}}=sign(w_{k,l})" border="0"/>  
- L2 Regualrization  
    <img src="http://latex.codecogs.com/svg.latex?R(W)=\Sigma_k\Sigma_l{w_{k,l}^2}" border="0"/>  
    <img src="http://latex.codecogs.com/svg.latex?\frac{\partial{R(W)}}{\partial{w_{k,l}}}=2w_{k,l}" border="0"/> 
```
template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];  // 每层的weight_decay
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        // 先求L1导数。caffe_cpu_sign，输出-1,0,1
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
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
## ComputeUpdateValue
```
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // ComputeUpdateValue = Regularize + momentum*last_update
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    // 保留本次更新值，下次使用
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
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
