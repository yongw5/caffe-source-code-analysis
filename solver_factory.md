# Solver Factory
- [简单工厂模式](#简单工厂模式)
- [`REGISTER_SOLVER_CLASS`宏定义](#`REGISTER_SOLVER_CLASS`宏定义)
- [`SolverRegistry`类](#`SolverRegistry`类)
  
## [简单工厂模式][1]
- 定义  
  定义一个工厂类，根据传入的参数不同返回不同的实例，被创建的实例具有共同的父类或接口。
- 实例  
  创建一个`Shape`的抽象类
  ```
  class Shape {
   public:
    virtual void draw() =0;
  };
  ``` 
  现在实现具体的类
  ```
  class CircleShape: public Shape {
   public:
    CircleShape() {
      std::cout << "Construct a CircleShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a circle." << std::endl;
    }
  };
  class RectShape: public Shape {
   public:
    RectShape() {
      std::cout << "Construct a RectShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a rectangle." << std::endl;
    }
  };
  class TriangleShape: public Shape {
  public:
    TriangleShape() {
      std::cout << "Construct a TriangleShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a triangle." << std::endl;
    }
  };
  ```
  普通地，在`main`函数中根据传入的shape类型实例化对应的类  
  ```
  int main(int argc, char *argv[]) {
    std::string shape_name(argv[1]);
    Shape* shape = nullptr;
    if (shape_name == "Circle")
      shape = new CircleShape();
    else if (shape_name == "Rect")
      shape = new RectShape();
    else if (shape_name == "Triangle")
      shape = new TriangleShape();
    else {
      std::cout << "Invalidate Shape Input" << std::endl;
      return -1;
    }
    shape->draw();
    delete(shape);
    return 0;
  }
  ```
  这是有效的，但每次添加新的`Shape`，都必须更新此代码并重新构建整个应用程序。 工厂设计模式或工厂方法模式（或动态构造函数）是一种用于创建对象的机制，而无需确切地知道需要创建哪个对象或如何实际创建对象。 类工厂提供了一个接口，其中子类可以实现创建特定对象所需的功能。  
  类工厂是用于创建其他对象的对象。 当类被添加到应用程序时，它们使用类工厂来修复它们的创建例程，然后可以根据请求实例化它们。 一个类工厂可以将`main`简化为这样的：  
  ```
  int main(int argc, char *argv[]) {
    std::string shape_name(argv[1]);
    Shape* shape = nullptr;
    Factory factory;
    shape = facotry.CreateShape(shape_name);
    if(!shape){
      std::cout << "Invalidate Shape Input" << std::endl;
      return -1;
    }
    shape->draw();
    delete(shape);
    return 0;
  }
  ```
  这里先跳过向工厂类注册实例化派生类方法的细节，首先需要定义一个能够向`ShapeFactory`进行注册的实例化派生类的方法。  
  ```
  Shape* CreatorCircleShape() { return new CircleShape(); }
  Shape* CreatorRectShape() { return new RectShape(); }
  Shape* CreatorTriangleShape() { return new TriangleShape(); }
  ```
  现在可以定义工厂类，根据参数，可以调用`CreateShape`方法实例化不用的派生类：  
  ```
  class ShapeFactory {
   public:
    typedef Shape* (*Creator)();
    Shape* CreateShape(const std::string& shape_name) {
      if(creator_map_.count(shape_name)) {
        return creator_map_[shape_name]();
      } else {
        return (Shape*)NULL;
      }
    }
    void Register(const std::string& shape_name, Creator creator) {
      creator_map_[shape_name] = creator;
    }
   private:
    std::map<std::string, Creator> creator_map_;
  };
  ```
  现在`main`函数可以简化为：  
  ```
  int main(int argc, char* argv[]) {
    ShapeFactory factory;
    factory.Register("Circle", CreatorCircleShape);
    factory.Register("Rect", CreatorRectShape);
    factory.Register("Triangle", CreatorTriangleShape);

    std::string shape_name(argv[1]);
    Shape* shape = nullptr;
    shape = facotry.CreateShape(shape_name);
    if(!shape){
      std::cout << "Invalidate Shape Input" << std::endl;
      return -1;
    }
    shape->draw();
    delete(shape);
    return 0;
  }
  ```
  定义实例化类的函数和注册可以用宏定义简化：  
  ```
  #define REGISTE_SHAPE_CREATOR(name) \
    Shape* Creator##name##Shape() { return new name##Shape(); } \
    factory.Register(#name, Creator##name##Shape)
  ``` 
  这时，需要定义一个全局工厂类实例：  
  ```
  ShapeFactory factory;
  REGISTE_SHAPE_CREATOR(Circle);
  REGISTE_SHAPE_CREATOR(Rect);
  REGISTE_SHAPE_CREATOR(Triangle);
  ```
  `main`函数：  
  ```
  extern Factory factory;
  int main(int argc, char* argv[]) {
    std::string shape_name(argv[1]);
    Shape* shape = nullptr;
    shape = facotry.CreateShape(shape_name);
    if(!shape){
      std::cout << "Invalidate Shape Input" << std::endl;
      return -1;
    }
    shape->draw();
    delete(shape);
    return 0;
  }
  ```
  进一步地，可以将工厂类的接口定义为`static`类型的函数，直接通过`ShapeFactory`就可以访问。  
  ```
  class ShapeFactory {
   public:
    typedef Shape* (*Creator)();
    typedef std::map<std::string, Creator> CreatorMap;
    ShapeFactory(std::string shape_name, Creator creator) { Register(shape_name, creator); }
    static Shape* CreateShape(const std::string& shape_name) {
      if(get_map().count(shape_name)) {
        return get_map()[shape_name]();
      } else {
        return (Shape*)NULL;
      }
    }
    static void Register(const std::string& shape_name, Creator creator) {
      get_map()[shape_name] = creator;
    }
   private:
    static CreatorMap& get_map() {
      static CreatorMap creator_map_;
      return creator_map_;
    }
  };
  ```
  将`creator_map_`作为静态局部变量包装在名为`get_map()`的成员函数中。 `Register()`方法将通过此静态函数访问`creator_map_`。 这可以保证在访问之前创建它。  
  现在修改`REGISTE_SHAPE_CREATOR`宏定义：  
  ```
  #define REGISTER_SHAPE_CLASS(type) \
  namespace {\
  Shape* Creator_##type##Shape() { \
    return new type##Shape(); \
  }  \
  static ShapeFactory g_register(#type, Creator_##type##Shape); \
  }
  ```
  每个类实例化`Creator`：  
  ```
  class CircleShape: public Shape {
   public:
    CircleShape() {
      std::cout << "Construct a CircleShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a circle." << std::endl;
    }
  };
  REGISTE_SHAPE_CREATOR(Circle);

  class RectShape: public Shape {
   public:
    RectShape() {
      std::cout << "Construct a RectShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a rectangle." << std::endl;
    }
  };
  REGISTE_SHAPE_CREATOR(Rect);
  
  class TriangleShape: public Shape {
  public:
    TriangleShape() {
      std::cout << "Construct a TriangleShape class." << std::endl;
    }
    void draw() {
      std::cout << "Draw a triangle." << std::endl;
    }
  };
  REGISTE_SHAPE_CREATOR(Triangle);
  ```
  现在，`main`函数可以为：  
  ```
  int main(int argc, char* argv[]) {
    std::string shape_name(argv[1]);
    Shape* shape = nullptr;
    shape = ShapeFactory::CreateShape(shape_name);
    if(!shape){
      std::cout << "Invalidate Shape Input" << std::endl;
      return -1;
    }
    shape->draw();
    delete(shape);
    return 0;
  }
  ```
## `REGISTER_SOLVER_CLASS`宏定义
```
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
```
根据传入的`type`定义一个实例化`type`对应类函数。比如`type`为`SGD`，宏展开为：  
```
template <typename Dtype>
Solver<Dtype>* Creator_SGDSolver(const SolverParameter& param) {
  return new SGDSolver<Dtype>(param)
}
REGISTER_SOLVER_CREATOR(SGD, Creator_SGDSolver)
```
`REGISTER_SOLVER_CREATOR`定义为下：
```
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \
```
定义两个静态的全局变量，只在定义该变量的源文件内有效，在同一源程序的其它源文件(即声明了该变量的CPP文件,或包含该变量声明头文件的CPP文件)中不能使用它。则`REGISTER_SOLVER_CREATOR(SGD, Creator_SGDSolver)`宏展开为：  
```
static SolverRegisterer<float> g_creator_f_SGD("SGD", Creator_SGDSolver<float>);
static SolverRegisterer<double> g_creator_d_SGD("SGD", Creator_SGDSolver<double>)
```
`SolverRegisterer`的定义是：  
```
template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};
```
在构造函数中`type`对应的`creator`向`SolverRegistry`添加。

## [SolverRegistry类][2]
首先，定义如下：
```
template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();
  static void AddCreator(const string& type, Creator creator);
  static Solver<Dtype>* CreateSolver(const SolverParameter& param);
  static vector<string> SolverTypeList();

 private:
  SolverRegistry() {}
  static string SolverTypeListString();
};
```
这个类的构造函数是private的，无法构造一个这个类型的变量，这个类也没有数据成员，所有的成员函数也都是static的，可以直接调用  
其中`AddCreator`定义如下： 
```
static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }
```
将`Solver<Dtype>* (*creator)`指针添加到一个`std::map<string, Creator>`类型的`registry`变量中。  
`registry`由`Registry()`得到，其定义如下：
```
static CreatorRegistry& Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry();
  return *g_registry_;
}
```
这个函数中定义了一个指向`std::map<string, Creator>`类型的`static`指针变量`g_registry`。因为这个变量是static的，所以即使多次调用这个函数，也只会定义一个`g_registry`，各个`Solver`的注册的过程正是往`g_registry`指向的那个`map`里添加以`Solver`的`type`为`key`，对应的`Creator`函数指针为`value`的内容。  
`CreateSolver`就是根据参数用注册的`Creator`实例化不用的`Solver`，其定义如下：
```
static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
  const string& type = param.type();
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
      << " (known types: " << SolverTypeListString() << ")";
  return registry[type](param);
}
```
因此，不同`Solver`注册的过程为：  
```
Register_Solver_Class(SGD)
  1)定义Creator_SGDSolver函数
  2)Register_Solver_Creator
    1)定义SolverRegister<float>类型的static变量
    2)定义SolverRegister<double>类型的static变量
      SolverRegistry::AddCreator
      将上面定义的Creator_SGDSolver的指针添加到g_registry_中
```
实例化过程：
```
SolverRegistry::CreateSolver(solver_param)
  在g_registry_中找到type对应的Creator函数指针
    调用Creator指向的函数
      new SGDSolver<Dtype>(solver_param)
```
[1]:http://alanse7en.github.io/caffedai-ma-jie-xi-4/
[2]:http://blog.fourthwoods.com/2011/06/04/factory-design-pattern-in-c/ 
