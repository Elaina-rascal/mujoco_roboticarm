import pinocchio as pin
from pathlib import Path
def main():
    print(pin.__version__)
    print(pin.__file__)
    xml = str(Path(__file__).resolve().parent.parent/'mujoco_arm_publisher' / 'models' / 'universal_robots_ur5e' / 'ur5e.xml')
    model = pin.buildModelsFromMJCF(xml) #type: ignore
    # pin.forwardKinematics(model, data)
    
    test=pin.SE3.Identity()
    print(test)
    # print(data)

# def invkinetic(model, data, q_init, q_goal, max_iterations=1000, tolerance=1e-4):
    # pin.forwardKinematics(model, data)
if __name__ == "__main__":
    main()