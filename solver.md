# solver.hpp & solver.cpp
- Solver Class
- Solve()
- Step()
- Test()
  
## Solver Class
类的定义
```
class Solver {
 public:
  explicit Solver(const SolverParameter& param);  //构造函数
  explicit Solver(const string& param_file);
  void Init(const SolverParameter& param);  //初始化函数
  void InitTrainNet();
  void InitTestNets();

  void SetActionFunction(ActionCallback func);//设置Action函数（该函数返回NONE,STOP和SNAPSHOT）
  SolverAction::Enum GetRequestedAction();//获取Action（NONE,STOP和SNAPSHOT）
  
  virtual void Solve(const char* resume_file = NULL);//Solver的主要接口，接受solver.prototxt开始训练
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);//默认情况下，iter将为零。 传递一个非零的数字，以恢复预训练网的训练。
  void Update();//调用ApplyUpdate()更新网络权重
  
  void Restore(const char* resume_file);//恢复训练
  void Snapshot();//保存Snapshot
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return iter_; }
  
  class Callback {  // 仅仅在多卡训练的时候被使用，将多个GPU之间的Solver进行同步
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }
  void CheckSnapshotWritePermissions();
  virtual inline const char* type() const { return ""; }

 protected:
  virtual void ApplyUpdate() = 0;//更新权重，SGD、Adam等具体实现
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  void TestAll();//测试
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;//Solver参数
  int iter_;//迭代次数
  int current_step_;//用于学习率的调整，比如lr_policy == "step"等
  shared_ptr<Net<Dtype> > net_;//训练网络，只能一个
  vector<shared_ptr<Net<Dtype> > > test_nets_;//测试网络，可以有多个
  vector<Callback*> callbacks_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;//将最近几次（average_loss参数决定，默认是1）的loss平均后的值
  ActionCallback action_request_function_;//Action函数
  bool requested_early_exit_;//是否提前退出
  Timer iteration_timer_;//时间统计的一个类的实例
  float iterations_last_;//上一次迭代，和iteration_timer_配合

  DISABLE_COPY_AND_ASSIGN(Solver);
}; 
```
## Solve实现
`Solve`的代码如下：
```
template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);  // 从shapshot读取保存的训练状态以及权重，从此处开始训练
  }

  int start_iter = iter_;
  Step(param_.max_iter() - iter_);  // 调用Step进行迭代优化，迭代次数（param_.max_iter() - iter_）

  if (param_.snapshot_after_train()  // 保存snapshot
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {  // 提前退出
    LOG(INFO) << "Optimization stopped early.";
    return;
  }

  if (param_.display() && iter_ % param_.display() == 0) {  // 输出loss信息
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {  // 测试验证集
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}
```
## Step()实现
```
template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();  // 用最近几次（average_loss）的loss求平均值（平滑窗口大小）
  losses_.clear();
  smoothed_loss_ = 0;  // 平滑后的loss
  iteration_timer_.Start();

  while (iter_ < stop_iter) {
    net_->ClearParamDiffs(); // 清空梯度信息
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {  // 本次迭代需要进行测试
      if (Caffe::root_solver()) {
        TestAll();
      }
      if (requested_early_exit_) {
        break;  // 需要提前退出，跳出循环
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0; // 本次迭代需要输出
    net_->set_debug_info(display && param_.debug_info());  // 设置debug_info

    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
    UpdateSmoothedLoss(loss, start_iter, average_loss);  // 更新loss平均值
    if (display) {  // 输出信息
      float lapse = iteration_timer_.Seconds();  // 上一次输出到本次总共时间
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);  // 一次迭代的时间
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();  // 从新开始计时
      iterations_last_ = iter_;  // 更新上次输出是的迭代次数
      const vector<Blob<Dtype>*>& result = net_->output_blobs();  // 输出网络loss
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();  // loss的值
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];  // loss的名字
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];  // loss的权重
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    ApplyUpdate();  // 更新权重，需要子类实现（SGD，Adam等）

    ++iter_;
    SolverAction::Enum request = GetRequestedAction();

    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {  // 保存shapshot
      Snapshot();
    }
    if (SolverAction::STOP == request) {  // 提前退出
      requested_early_exit_ = true;
      break;
    }
  }
}
```
## Test()实现
```
template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    while (request != SolverAction::NONE) {  // 可以相应多次（比如两次Ctrl+C）
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      break;  // 退出for循环
    }
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) { // 第一次测试迭代，test_score、test_score_output_id是空的
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {  // 提前退出
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}
```