
 
import matplotlib.pyplot as plt
import random
import math
import copy
import numpy as np
import scipy
 #test1
show_animation = True
KNN = 5
better_planning =True

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
 
class Node(object):
    """
    RRT Node
    """
 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
 
 
class RRT(object):
    """
    Class for RRT Planning
    """
 
    def __init__(self, start, goal, obstacle_list, rand_area):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:random sampling Area [min,max]
        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expandDis = 0.6
        self.goalSampleRate = 0.05  # 选择终点的概率是0.05
        self.maxIter = 500
        self.obstacleList = obstacle_list
        self.nodeList = [self.start]
 
    def random_node(self):
        """
        产生随机节点
        :return:
        """
        node_x = random.uniform(self.min_rand, self.max_rand)
        node_y = random.uniform(self.min_rand, self.max_rand)
        node = [node_x, node_y]
        #node2 = Node(node_x,node_y)
 
        return node
 
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

    @staticmethod
    def get_KNN_list_index(node_list, rnd):
        """
        :param node_list:
        :param rnd:
        :return:
        """
        d_list = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 for node in node_list]
        nlist = np.array(d_list)
        nlist = nlist.argsort()
        #d_list.sort()
        nn_list = nlist[:KNN]
        
        #min_index = d_list.index(min(d_list))
        return nn_list
 
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
 
    def planning(self):
        """
        Path planning
        animation: flag for animation on or off
        """
 
        while True:
            # Random Sampling
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = [self.end.x, self.end.y]
 
            # Find nearest node
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
            compare = []
            if(len(self.nodeList)>KNN):
                nn_list =self.get_KNN_list_index(self.nodeList,[new_node.x,new_node.y])
                print(nn_list,min_index)
                for l in range(KNN):
                    new_node.parent = nn_list[l]
                    if not self.collision_check(new_node, self.obstacleList):
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
                        cpath.append([self.start.x, self.start.y])
                        
                        compare.append([length(cpath),nn_list[l]])
                compare.sort()
                if len(compare)>0 and better_planning:
                    print("previous:",min_index," now:",compare[0][1])
                    bestnode = compare[0][1]
                #print(compare)

                    #break #

            new_node.parent = bestnode
 
            if not self.collision_check(new_node, self.obstacleList):
                continue
 
            self.nodeList.append(new_node)
 
            # check goal
            dx = new_node.x - self.end.x
            dy = new_node.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                print("Goal!!")
                break
 
            if True:
                self.draw_graph(rnd)
 
        path = [[self.end.x, self.end.y]]
        last_index = len(self.nodeList) - 1
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append([node.x, node.y])
            last_index = node.parent
        path.append([self.start.x, self.start.y])
 
        return path
 
    def draw_graph(self, rnd=None):
        """
        Draw Graph
        """
        print('aaa')
        plt.clf()  # 清除上次画的图
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^g")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")
 
        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "sk", ms=10*size)
 
        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.end.x, self.end.y, "^b")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)
 
    def draw_static(self, path):
        """
        画出静态图像
        :return:
        """
        plt.clf()  # 清除上次画的图
 
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                    node.y, self.nodeList[node.parent].y], "-g")
 
        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "sk", ms=10*size)
 
        plt.plot(self.start.x, self.start.y, "^r")
        plt.plot(self.end.x, self.end.y, "^b")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
 
        plt.plot([data[0] for data in path], [data[1] for data in path], '-r')
        plt.grid(True)
        plt.show()


 
def main():
    print("start RRT path planning")
 
    obstacle_list = [
        (5, 1, 1),
        (3, 6, 2),
        (3, 8, 2),
        (1, 1, 2),
        (3, 5, 2),
        (9, 5, 2)]
 
    # Set Initial parameters
    rrt = RRT(start=[0, 0], goal=[8, 9], rand_area=[-2, 10], obstacle_list=obstacle_list)
    path = rrt.planning()
    #print(path)
    length(path)
 
    # Draw final path
    if show_animation:
        plt.close()
        rrt.draw_static(path)
 
 
if __name__ == '__main__':
    main()