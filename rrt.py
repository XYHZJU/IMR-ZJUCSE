#from vision import Vision
from scipy.spatial import KDTree
import numpy as np
import random
import math
import copy
import time
from debug import Debugger

better_planning =True
debugger = Debugger()
def length(path):
    lx = []
    ly = []
    dis = 0
    for x,y in  path:
        lx.append(x)
        ly.append(y)
    for i in range(len(lx)-1):
        dis = dis + math.sqrt(pow((lx[i+1]-lx[i]),2)+pow((ly[i+1]-ly[i]),2))
        #print(dis)
def cal_ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
    A=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    C=math.degrees(math.acos((c*c-a*a-b*b)/(-2*a*b)))
    return B
 
#cal_ang((0, 0), (1, 1), (0, 1))



class Node(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #self.cost = cost
        self.parent = None


class RRT(object):
    def __init__(self, N_SAMPLE=100, KNN=15, MAX_EDGE_LEN=5000,begin=[0,0], goal=[100,100]):
        self.N_SAMPLE = N_SAMPLE
        self.KNN = KNN
        self.MAX_EDGE_LEN = MAX_EDGE_LEN
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200

        self.begin = Node(begin[0],begin[1])
        self.end = Node(goal[0],goal[1])
        self.expandDis = 100
        self.goalSampleRate = 0.05  # 选择终点的概率
        self.maxIter = 500
        self.nodeList = [self.begin]
        
        

    def plan(self, vision, start_x, start_y, goal_x, goal_y):
        # Obstacles
        obstacle_x = [-999999]
        obstacle_y = [-999999]
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        # Obstacle KD Tree
        # print(np.vstack((obstacle_x, obstacle_y)).T)
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        # Sampling
        sample_x, sample_y = self.sampling(start_x, start_y, goal_x, goal_y, obstree)
        # Generate Roadmap
        road_map = self.generate_roadmap(sample_x, sample_y, obstree)
        # Search Path
        path_x, path_y = self.dijkstra_search(start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y)

        return path_x, path_y, road_map, sample_x, sample_y

    def obstacle(self, vision):
        # Obstacles
        obstacle_x = [-999999]
        obstacle_y = [-999999]
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        # Obstacle KD Tree
        # print(np.vstack((obstacle_x, obstacle_y)).T)
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        
        return obstree

    def random_node(self):
        """
        产生随机节点
        :return:
        """
        node_x = random.uniform(self.minx, self.maxx)
        node_y = random.uniform(self.miny, self.maxy)
        node = [node_x, node_y]
        #node2 = Node(node_x,node_y)
 
        return node

    @staticmethod
    def collision_check(new_node, obstacle_list):
        a = 1
        for (ox, oy, size) in obstacle_list:
            dx = ox - new_node.x
            dy = oy - new_node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                a = 0  # collision
 
        return a  # safe

    @staticmethod
    def get_nearest_list_index(node_list, rnd):
        """
        :param node_list:
        :param rnd:
        :return:
        """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        min_index = d_list.index(min(d_list))
        return min_index
    
    def get_KNN_list_index(self,node_list, rnd):
        """
        :param node_list:
        :param rnd:
        :return:
        """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        nlist = np.array(d_list)
        nlist = nlist.argsort()
        #d_list.sort()
        nn_list = nlist[:self.KNN]
        
        #min_index = d_list.index(min(d_list))
        return nn_list

    def sampling(self, start_x, start_y, goal_x, goal_y, obstree):
        sample_x, sample_y = [], []

        while len(sample_x) < self.N_SAMPLE:
            tx = (random.random() * (self.maxx - self.minx)) + self.minx
            ty = (random.random() * (self.maxy - self.miny)) + self.miny

            distance, index = obstree.query(np.array([tx, ty]))

            if distance >= self.robot_size + self.avoid_dist:
                sample_x.append(tx)
                sample_y.append(ty)

        sample_x.append(start_x)
        sample_y.append(start_y)
        sample_x.append(goal_x)
        sample_y.append(goal_y)

        return sample_x, sample_y
    
    def sample2(self):
        
        rnd = []
        if random.random() > self.goalSampleRate:
                rnd = self.random_node()
        else:
                rnd = [self.end.x, self.end.y]
        return rnd

    def generate_roadmap(self, sample_x, sample_y, obstree):
        road_map = []
        nsample = len(sample_x)
        sampletree = KDTree(np.vstack((sample_x, sample_y)).T)

        for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):
            distance, index = sampletree.query(np.array([ix, iy]), k=nsample)
            edges = []
            # print(len(index))

            for ii in range(1, len(index)):
                nx = sample_x[index[ii]]
                ny = sample_y[index[ii]]

                # check collision
                if not self.check_obs(ix, iy, nx, ny, obstree):
                    edges.append(index[ii])

                if len(edges) >= self.KNN:
                    break

            road_map.append(edges)

        return road_map

    def generate_RRTree(self,vision):
        
        "return path_x, path_y, road_map"
        iter = 0
        #obstree = self.obstacle(self,vision)
        road_map = [[0]]
        obstacle_x = [-999999]
        obstacle_y = [-999999]
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        # Obstacle KD Tree
        # print(np.vstack((obstacle_x, obstacle_y)).T)
        obslist = [obstacle_x, obstacle_y]
        
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        
        #print("obx:",obstacle_x)
        #print("oby:",obstacle_y)

        #print(obslist[0][index], obslist[1][index])

        while True:
            
            edge = []
            if(iter<self.maxIter):
                iter = iter +1
            else:
                break

            rnd = self.sample2()
            min_index = self.get_nearest_list_index(self.nodeList, rnd)
            
            #print(nn_list)
            # print(min_index)
 
            # expand tree
            nearest_node = self.nodeList[min_index]
 
            # 返回弧度制
            theta = math.atan2(rnd[1] - nearest_node.y, rnd[0] - nearest_node.x)
 
            new_node = copy.deepcopy(nearest_node)
            new_node.x += self.expandDis * math.cos(theta)
            new_node.y += self.expandDis * math.sin(theta)

            bestnode = min_index
            #ans = self.show_obs(new_node.x, new_node.y, obstacle_x, obstacle_y, obstree)
            compare = []
            if(len(self.nodeList)>self.KNN):
                nn_list =self.get_KNN_list_index(self.nodeList,[new_node.x,new_node.y])
                #print(nn_list,min_index)
                for l in range(self.KNN):
                    dx = self.nodeList[nn_list[l]].x - new_node.x
                    dy = self.nodeList[nn_list[l]].y - new_node.y
                    long = math.hypot(dx,dy)
                    if long<=self.expandDis:
                        new_node.parent = nn_list[l]
                        #debugger.show_point(new_node.x,new_node.y)
                        
                        tarnode = self.nodeList[nn_list[l]]
                        #debugger.show_point(tarnode.x,tarnode.y)
                        if not self.show_obs(tarnode.x, tarnode.y, new_node.x, new_node.y, obstacle_x, obstacle_y, obstree):
                        #if not self.check_obsnew(new_node.x, new_node.y, tarnode.x, tarnode.y, obstree, obslist, obstacle_x, obstacle_y):
                        #if not self.collision_check(new_node, self.obstacleList):
                            temp_nodeList = self.nodeList.copy()
                            #print(len(self.nodeList),len(temp_nodeList))
                            temp_nodeList.append(new_node)

                            cpath = [[new_node.x, new_node.y]]
                            last_index = len(temp_nodeList) - 1
                            #print(temp_nodeList)
                            while temp_nodeList[last_index].parent is not None:
                                node = temp_nodeList[last_index]
                                cpath.append([node.x, node.y])
                                last_index = node.parent
                            cpath.append([self.begin.x, self.begin.y])
                            
                            compare.append([length(cpath),nn_list[l]])
                compare.sort()
                if len(compare)>0 and better_planning:
                    #print("previous:",min_index," now:",compare[0][1])
                    bestnode = compare[0][1]
            if  self.show_obs(self.nodeList[bestnode].x, self.nodeList[bestnode].y, new_node.x, new_node.y, obstacle_x, obstacle_y, obstree):
                continue
            else:
                new_node.parent = bestnode
            


            #if not self.check_obs(new_node.x, new_node.y, tarnode.x, tarnode.y, obstree):
                #continue
 
                self.nodeList.append(new_node)
            road_map.append([len(road_map)])
            road_map[bestnode].append(self.nodeList.index(self.nodeList[-1]))

            # check goal
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis/2:
                print("Goal!!")
                break

        path = [[self.end.x, self.end.y]]
        path_x = [self.end.x]
        path_y = [self.end.y]
        last_index = len(self.nodeList) - 1
        last_index = self.get_nearest_list_index(self.nodeList, path[0])
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            
            path.append([node.x, node.y])
            path_x.append(node.x)
            path_y.append(node.y)
            last_index = node.parent
            
        path.append([self.begin.x, self.begin.y])
        path_x.append(self.begin.x)
        path_y.append(self.begin.y)
        #print("path:",path)
        #print("x:",path_x)
        #print("y:",path_y)
        #t1 = [1,0]
        #t2 = [100,100]
        #Debugger.show_path(t1, t2)
 
        return path_x,path_y,road_map,self.nodeList

            
        road_map = []
        nsample = len(sample_x)
        sampletree = KDTree(np.vstack((sample_x, sample_y)).T)

        for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):
            distance, index = sampletree.query(np.array([ix, iy]), k=nsample)
            edges = []
            # print(len(index))

            for ii in range(1, len(index)):
                nx = sample_x[index[ii]]
                ny = sample_y[index[ii]]

                # check collision
                if not self.check_obs(ix, iy, nx, ny, obstree):
                    edges.append(index[ii])

                if len(edges) >= self.KNN:
                    break

            road_map.append(edges)

        return road_map

    def check_obs(self, ix, iy, nx, ny, obstree):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)

        if dis > self.MAX_EDGE_LEN:
            return True

        step_size = self.robot_size + self.avoid_dist
        steps = round(dis/step_size)
        for i in range(steps):
            distance, index = obstree.query(np.array([x, y]))
            if distance <= self.robot_size*2 + self.avoid_dist*2:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

        # check for goal point
        distance, index = obstree.query(np.array([nx, ny]))
        if distance <= self.robot_size*2 + self.avoid_dist*2:
            return True

        return False
    
    def check_obsnew(self, ix, iy, nx, ny, obstree, obslist,obstacle_x, obstacle_y):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)

        if dis > self.MAX_EDGE_LEN:
            return True

        step_size = self.robot_size + self.avoid_dist
        steps = round(dis/step_size)

        #distance, index = obstree.query(np.array([x, y]))
        #print("obsx:",obstacle_x)

        for i in range(steps):
            distance, index = obstree.query(np.array([x, y]))
            #print(obslist[0][index], obslist[1][index])
            if distance <= self.robot_size*2 + self.avoid_dist*2:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

        # check for goal point
        distance, index = obstree.query(np.array([nx, ny]))
        if distance <= self.robot_size*2 + self.avoid_dist*2:
            return True

        return False
    def show_obs(self,ix, iy, nx, ny, obstacle_x, obstacle_y, obstree):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)
        mindis = 1000
        if dis > self.MAX_EDGE_LEN:
            return True
        distance, index = obstree.query(np.array([x, y]),k=self.KNN)
        #print("index:",len(obstacle_x))
        #print(len(obstacle_x),max(index))
        if (max(index)>=self.KNN and max(index)<=len(obstacle_x)):
            
            for s_index in index:
                #print("obsx:",obstacle_x[s_index])
                obsx = obstacle_x[s_index]
                obsy = obstacle_y[s_index]

                odx = obsx - ix
                ody = obsy - iy

                odx2 = obsx - nx
                ody2 = obsy - ny
                obdis = math.hypot(odx,ody)
                obdis2 = math.hypot(odx2,ody2)
                if obdis < mindis:
                    mindis = obdis

                theta = cal_ang((nx,ny), (ix,iy), (obsx,obsy))
                if  obdis < dis and obdis2<2*self.avoid_dist:
                    #print(obdis,dis)
                    #debugger.show_circle1(ix,iy,dis)
                    return True
                #print(mindis)
        
        return False

    def show_obs2(self,ix, iy, nx, ny, obstacle_x, obstacle_y, obstree):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)
        mindis = 1000
        if dis > self.MAX_EDGE_LEN:
            return True
        distance, index = obstree.query(np.array([x, y]),k=self.KNN)
        #print("index:",len(obstacle_x))
        #print(len(obstacle_x),max(index))
        if (max(index)>=self.KNN and max(index)<=len(obstacle_x)):
            
            for s_index in index:
                #print("obsx:",obstacle_x[s_index])
                obsx = obstacle_x[s_index]
                obsy = obstacle_y[s_index]

                odx = obsx - ix
                ody = obsy - iy

                odx2 = obsx - nx
                ody2 = obsy - ny
                obdis = math.hypot(odx,ody)
                obdis2 = math.hypot(odx2,ody2)
                if obdis < mindis:
                    mindis = obdis

                theta = cal_ang((nx,ny), (ix,iy), (obsx,obsy))
                if  theta<=30:
                    #print(obdis,dis)
                    #debugger.show_circle1(ix,iy,dis)
                    return True
                #print(mindis)
        
        return False

    def dijkstra_search(self, start_x, start_y, goal_x, goal_y, road_map,
        sample_x, sample_y):
        path_x, path_y = [], []
        start = Node(start_x, start_y, 0.0, -1)
        goal = Node(goal_x, goal_y, 0.0, -1)

        openset, closeset = dict(), dict()
        openset[len(road_map)-2] = start

        path_found = True
        while True:
            if not openset:
                print("Cannot find path")
                path_found = False
                break

            c_id = min(openset, key=lambda o: openset[o].cost)
            current = openset[c_id]

            if c_id == (len(road_map) - 1):
                print("Goal is found!")
                goal.cost = current.cost
                goal.parent = current.parent
                break

            del openset[c_id]
            closeset[c_id] = current

            # expand
            for i in range(len(road_map[c_id])):
                n_id = road_map[c_id][i]
                dx = sample_x[n_id] - current.x
                dy = sample_y[n_id] - current.y
                d = math.hypot(dx, dy)
                node = Node(sample_x[n_id], sample_y[n_id],
                    current.cost + d, c_id)
                if n_id in closeset:
                    continue
                if n_id in openset:
                    if openset[n_id].cost > node.cost:
                        openset[n_id].cost = node.cost
                        openset[n_id].parent = c_id
                else:
                    openset[n_id] = node

        if path_found:
            path_x.append(goal.x)
            path_y.append(goal.y)
            parent = goal.parent
            while parent != -1:
                path_x.append(closeset[parent].x)
                path_y.append(closeset[parent].y)
                parent = closeset[parent].parent

        return path_x, path_y
