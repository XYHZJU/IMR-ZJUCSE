from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
import rrt
import time

if __name__ == '__main__':
    vision = Vision()
    action = Action()
    debugger = Debugger()
    planner = PRM()
    while True:
        # 1. path planning & velocity planning
        start_x, start_y = vision.my_robot.x, vision.my_robot.y
        goal_x, goal_y = -2400, -1500
        #path_x, path_y, road_map, sample_x, sample_y = planner.plan(vision=vision, 
            #start_x=start_x, start_y=start_y, goal_x=goal_x, goal_y=goal_y)
        
        planner2 = rrt.RRT(begin=[start_x, start_y], goal=[goal_x, goal_y])
        path_x,path_y, road_map, nodelist = planner2.generate_RRTree(vision)

        #fine = False
        action.controlObs(vision)
        debugger.draw_all2(nodelist, road_map, path_x, path_y)

        # 2. send command
        action.sendCommand(vx=0, vy=0, vw=0)
        

        # 3. draw debug msg
        #debugger.show_path(path_x,path_y)
        
        #debugger.show_path(path2[0],path2[1])
        #debugger.draw_finalpath(package,path_x, path_y)
        #debugger.draw_all(sample_x, sample_y, road_map, path_x, path_y)
        
        time.sleep(0.05)
