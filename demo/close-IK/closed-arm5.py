from spatialmath import SE2, Twist2,SO2
from spatialmath.base import tr2adjoint2
import numpy as np
from typing import cast
def main():
    l1,l2,l3,l4=1.0,1.0,1.0,1.0
    q1,q2,q3,q4,q5=0.0,0.0,0.0,0.0,0.0
    #定义右边轴的旋量
    S1=Twist2.UnitRevolute([0.2,0])
    S2=Twist2.UnitRevolute([0.2+l1,0])
    M1=SE2(0.2+l1+l2,0,0)
    print(f"当前末端位姿: {M1}")
    #定义左边轴的旋量
    S3=Twist2.UnitRevolute([-0.2,0])
    S4=Twist2.UnitRevolute([-0.2-l3,0])
    M2=SE2(-0.2-l3-l4,0,0)
    
    T1F=M1
    T2F=M2
    #算误差
    for i in range(500):
        #在这里*表示矩阵乘法，@表示乘法并归一化
        #在这里*表示矩阵乘法，@表示乘法并归一化
        #在这里*表示矩阵乘法，@表示乘法并归一化
        #左边的位姿
        T1=cast(SE2,S1.exp(q1)*S2.exp(q2)*M1);E1=cast(SE2,S1.exp(q1));E12=cast(SE2,S1.exp(q1)*S2.exp(q2))  #type: ignore
        T2=cast(SE2,S3.exp(q3)*S4.exp(q4)*M2);E3=cast(SE2,S3.exp(q3));E34=cast(SE2,S3.exp(q3)*S4.exp(q4))  #type: ignore
        T1_error=T2*T1.inv()
        T2_error=T1*T2.inv()
        assert isinstance(T1_error,SE2)
        assert isinstance(T2_error,SE2)
        v1_error=T1_error.log(twist=True)
        # print(v1_error)
        v2_error=T2_error.log(twist=True)
        #计算雅克比
        J1=np.column_stack([S1.A,tr2adjoint2(E1.A)@S2.A])
        J2=np.column_stack([S3.A,tr2adjoint2(E3.A)@S4.A])
        #用雅克比的逆算q1,q2,q3,q4
        dq12=np.linalg.pinv(J1)@v1_error
        dq34=np.linalg.pinv(J2)@v2_error
        q1+=dq12[0];q2+=dq12[1];q3+=dq34[0];q4+=dq34[1]
        T1F=cast(SE2,S1.exp(q1)*S2.exp(q2)*M1);T2F=cast(SE2,S3.exp(q3)*S4.exp(q4)*M2)  #type: ignore
    print(f"当前位姿{T1F}\n 当前位姿{T2F}")
    print(f"当前关节角: {[q1,q2,q3,q4]}")

if __name__=='__main__':
    main()