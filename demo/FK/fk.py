"""FK 练习：只修改 fk()，直接运行 python3 demo/FK/fk.py 核对。"""

import numpy as np

# 供你在 fk() 中使用。
from lie import exp_se3, make_transform, rotation_axis,world_to_twist





def main():
    # 两连杆平面机械臂：两杆均长 1 m，零位沿世界 +x 方向伸直。
    # 两个关节均绕 +z 轴转动，第二关节在零位位于 [1, 0, 0]。
    # 转动关节：w 是单位轴方向，v = -cross(w, 轴上一点)。
    vs1=world_to_twist([0,0,1],np.zeros(3),[0,0,0])
    vs2=world_to_twist([0,1,0],np.zeros(3),[0,0,0.2])
    vs3=world_to_twist([0,1,0],np.zeros(3),[1,0,0.2])
    #创建初始T
    M=make_transform(t=[1.5,0,0.2])
    #vs1 vs2 vs3全部旋转40度
    q=np.array([np.pi/4,-np.pi/4,np.pi/4])
    #计算指数积分
    T1=exp_se3(vs1*q[0])
    T2=exp_se3(vs2*q[1])
    T3=exp_se3(vs3*q[2])
    #计算最终T,保留3位小数
    T=np.round(T1@T2@T3@M,3)
    print("最终位姿T:\n",T)

    #用自身旋量计算
    vb1=world_to_twist([0,0,1],np.zeros(3),[-1.5,0.0,-0.2])
    vb2=world_to_twist([0,1,0],np.zeros(3),[-1.5,0.0,0.0])
    vb3=world_to_twist([0,1,0],np.zeros(3),[-0.5,0,0])
    M=make_transform(t=[1.5,0,0.2])
    q=np.array([np.pi/4,-np.pi/4,np.pi/4])
    T1=exp_se3(vb1*q[0])
    T2=exp_se3(vb2*q[1])
    T3=exp_se3(vb3*q[2])
    T=np.round(M@T1@T2@T3,3)
    print("最终位姿T:\n",T)

if __name__ == "__main__":
    main()
