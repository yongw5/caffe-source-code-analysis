# caffe.cpp
- [Google Gflags](#Google-Gflags的使用)
- [RegisterBrewFunction](#RegisterBrewFunction)
- [train()](#train())

## Google Gflags的使用
- 定义变量  
`DEFINE_xxx(name, val, txt)`
其中，`xxx` 是定义的变量的类型，Gflags支持的类型如下

  |Gflags|C++|  
  |:---|:---|  
  |DEFINE_bool| boolean|  
  |DEFINE_int32|32-bit integer|  
  |DEFINE_int64|64-bit integer|  
  |DEFINE_uint32|unsigned 64-bit integer|
  |DEFINE_double|double|
  |DEFINE_string|string|  
  
  `name`是变量名，`val`是变量默认值，`txt`是变量说明。例如：  
  `DEFINE_bool(big_menu, true, "Include 'advanced' options in the menu listing");`  
- 访问变量  
  通过`FLAGS_name`像正常变量一样访问标志参数，例如：  
  `FLAGS_big_menu;`  
  若在不同文件（不是定义该变量的文件）中，用`DECLARE_xxx(name)`来声明引入这个参数，例如：  
  `DECLARE_bool(big_menu)`  
- 整合、初始化所有参数  
  通常在`main()`中调用：  
  `google::ParseCommandLineFlags(&argc, &argv, true)`  
  最后一个参数称为`remove_flags`。 如果为`true`，则`ParseCommandLineFlags`从`argv`中删除标志及其参数，并相应地修改`argc`。 在这种情况下，在函数调用之后，`argv`将只保存命令行参数，而不保存命令行标志。
- 在命令行设置参数  
  除`boolean`类型的参数，都可以用如下方式设置：  
  `app_containing_foo --int_val=20`  
  `app_containing_foo -double_val=10`  
  `app_containing_foo --string_val "example string"`  
  `app_containing_foo -double 0.33`  
  `boolean`类型的参数设置：  
  `app_containing_foo --big_menu`  
  `app_containing_foo --nobig_menu`  
  `app_containing_foo --big_menu=true`  
  `app_containing_foo --big_menu=false`  
- 改变参数默认值  
  有时有一个参数是在`lib`库中定义，您希望在某个程序中更改其默认值，而不是在其他程序中更改，可以在`main`中`ParseCommandLineFlags`前设置新值，比如：  
  ```
  DECLARE_bool(lib_verbose);   // mylib has a lib_verbose flag, default is false
  int main(int argc, char** argv) {
    FLAGS_lib_verbose = true;  // in my app, I want a verbose lib by default
    ParseCommandLineFlags(...);
  }
  ```
## RegisterBrewFunction
Caffe在Command Line Interfaces中一共提供了4种功能:train/test/time/device_query，分别对应着四个函数。  
```
usage: caffe <command> <args>  
commands:  
  train           train or finetune a model
  test            score a model
  device_query    show GPU diagnostic information
  time            benchmark model execution time
```
`main`函数通过命令行参数调用`GetBrewFunction`获取相应的函数(train/test/time/device_query)执行：  `GetBrewFunction(caffe::string(argv[1]))()`  
`GetBrewFunction`定义如下：  
```
static BrewFunction GetBrewFunction(const caffe::string& name) {
  if (g_brew_map.count(name)) {
    return g_brew_map[name];
  } else {
    LOG(ERROR) << "Available caffe actions:";
    for (BrewMap::iterator it = g_brew_map.begin());
         it ! = g_brew_map.end(); ++it) {
      LOG(ERROR) << "\t" << it->first;
    }
    LOG(FATAL) << "Unknown action: " << name;
    return NULL;
  }
}
```
其中`g_brew_map`是一个`std::map`类型的全局变量，定义如下：  
```
typedef int (*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;
```
其元素是函数名和指向该函数的函数指针， 由宏定义`RegisterBrewFunction`添加元素（train/test/time/device_query函数名和函数指针），定义如下：  
```
#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: \
  __Registerer_##func() { \
    g_brew_map[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}
```
比如：  
```
int train() {...}
RegisterBrewFunction(train);
```
宏展开为：  
```
int train() {...}
namespace {
class __Registerer_train {
 public:
  __Registerer_train() {
    g_brew_map["train"] = &train;
  }    
};
__Register_train g_registerer_train;
}
```
定义了一个类`__Registerer_train`，其构造函数`__Register_train()`是向`g_brew_map`添加元素`BrewMap::value_type ("train", &train)`  
若`argv[1]`的参数是`train`则：  
`return GetBrewFunction(caffe::string(argv[1]))();`  
等价于执行：  
`return train();`
## train()
调用`solver`的`Solve()`进行训练过程：  
`solver->Solve();`
## test()
进行前传统计网络输出层结果  
```
for (int i = 0; i < FLAGS_iterations; ++i) {
  float iter_loss;
  const vector<Blob<float>*>& result = caffe_net.Forward(&iter_loss);
  ...
```
