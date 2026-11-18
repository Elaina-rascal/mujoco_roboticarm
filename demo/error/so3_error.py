import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
import time
def time_wrapper(func,):
    def inner():
        start_time=time.time_ns()
        res=0
        for i in range (10):
            res=func()
        print(res)
        final_time=time.time_ns()
        dt=(final_time-start_time)/(10*1000)
        print(f"平均执行{dt}us")
        return res
    return inner
#min(Log(R(yaw,pitch)R-1))
def so3_error(x, R_target):
    yaw, pitch = x

    R = np.asarray(Rot.from_euler("ZY", [yaw, pitch]).as_matrix())
    R_err = R.T @ R_target

    return np.linalg.norm(Rot.from_matrix(R_err).as_rotvec())**2
import pinocchio as pin
def se3_error(v,T,H,M):

    return 
@time_wrapper
def test():
    R_target=Rot.from_euler('zyx', [90, 45, 30], degrees=True).as_matrix()
    res = minimize(so3_error, x0=[0.0, 0.0], args=(R_target,))
    yaw, pitch = res.x
    return yaw,pitch
@time_wrapper
def se3_test():
    T=pin.SE3(np.eye(3),np.asarray([2,2,3]))
    H=np.diag([1,1,1,1,1,1])
if __name__=="__main__":
    test()