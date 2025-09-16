Simulator demo set:

A. https://github.com/spencerfolk/rotorpy
use rotorpy packege

1. #install rotorpy：
   
   git clone https://github.com/spencerfolk/rotorpy.git 
   
   cd rotorpy
   
   pip install -e .

2. #Basic usage (recommended to run this first): 
   
   python examples/basic_usage.py
   
   #Parallel batch simulation (CPU/GPU compatible):
   
   python examples/batched_simulation.py
   
   python examples/benchmark_batched_simulation.py # (Check for parallel speedup)

3. #Necessary imports and how to create and execute an instance of the simulator.
minimum working example：demo_min.py

   python demo_min.py
   #save the animation as my_demo.mp4


B. use https://github.com/utiasDSL/gym-pybullet-drones
1. install:
   
   git clone https://github.com/utiasDSL/gym-pybullet-drones.git
   
   cd gym-pybullet-drones/
   
   pip3 install -e .

2. run the example:

    cd gym_pybullet_drones/examples/
    
    python3 pid.py 

    #The purpose of pid.py is to demonstrate a simple PID controller for quadrotor drones, which tracks a desired trajectory in the PyBullet simulation. It shows how to integrate a controller with the environment and observe drone behavior. (read)

3.