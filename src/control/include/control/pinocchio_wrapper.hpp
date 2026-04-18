#pragma once

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace pinocchio {

// 在头文件中声明 extern template，避免每个编译单元重复实例化这些模板类，
// 由 pinocchio_wrapper.cpp 统一完成显式实例化，减少编译时间和目标文件体积。
extern template class ModelTpl<double>;
extern template class DataTpl<double>;

// 该函数仅用于在单个编译单元中集中“触发”控制器所需算法模板实例化。
// 业务代码无需调用它；它的意义是把模板代码生成位置固定到 wrapper 层。
void instantiate_used_algorithms_for_control();

} // namespace pinocchio
