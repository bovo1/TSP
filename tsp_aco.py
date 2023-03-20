#개미 군체가 있다.
#개미들은 각 노드(가야할 도시)들을 한번 씩 다 방문한다. 모두 다 다른 경로로.
#도시에 도착했을 때의 거리 비용을 각 개미에게 저장함. -> 딕셔너리 형태
#다 돌고나서 원래 있었던 시작 노드에 돌아가야 한다.
#모든 개미가 다 돌아올 때 까지 기다리게 된다면 그만큼의 시간 소모가 있다.
#그러므로 먼저 도착한 개미는 곧바로 경로를 수정하여 도시를 경유할 수 있어야 한다.
#경로를 수정하기 위해 개미들이 수집한 경로 비용 정보가 필요하다.
#그 딕셔너리 정보를 공유한다고 생각하기 보다는 페로몬을 이용해야 한다.
#모든 개미들은 일정량의 페로몬을 분비한다.
#시간이 지날 수록 페로몬은 옅어지게 된다. 하지만 개미들이 많이 들락날락 거린 경로일수록
#그 경로에 뿌려진 페로몬의 양? 농도?는 다른 경로에 비해 많다.
#개미는 시각에 의존하기 보다는 그 페로몬을 의존하기 때문에 페로몬이 많은 경로를 선택한다.
#그렇게 점점 경로 비용이 좋지 않은 경로에는 페로몬이 사라지게 되고
#좋은 비용을 가진 경로엔 페로몬이 많아지게 되므로 그 경로가 선택된다.
#이를 구현하기 위해서 개미들은 경로를 지날 때마다 일정량의 페로몬을 분비하도록 해야하며
#그 페로몬의 수치는 시간이 지남에 따라 옅어지게 만들어야 한다.
#그렇다면 거리 비용이 저장된 딕셔너리를 쓰는 게 의미가 있는 것인가?
#개미들이 이 경로가 더 좋다고 선택할 기준: 페로몬과 자신이 가지고 있는 경로비용
#개미는 자신이 한번 간 경로는 절대 다시 가지 않음.
#한번 간 경로임을 판단하기 위해 경로 리스트가 이용된다.
#
import random, math, copy, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from haversine import haversine


class ACO:  #ACO는 자연현상 그 자체인...? 클래스로 보면 이해하기 좋을 것 같다.

    class Node: #지점
        def __init__(self, lat_long, num):
            self.position = lat_long #튜플 형태
            self.pos_number = num   #지점 번호


    class Path:     #Path는 경로? 지점과 지점 사이의 경로라고 보면 된다.
        def __init__(self, path_distance, initial_pheromone):
            self.path_distance = path_distance
            self.pheromone = initial_pheromone
            self.max_pheromone = 0
            self.min_pheromone = 0
    
    class Ant:  #경로와 지점을 오가는 개미 !
        def __init__(self, num_nodes, paths, ID):
            self.num_nodes = num_nodes
            self.paths = copy.deepcopy(paths) #전체 거점들
            #self.paths = copy.copy(paths)
            self.visited_node = []    #이미 갔던 지점
            self.unvisited_node = []    #안 간 지점
            self.visited_dist = 0
            self.id = ID
            self.to_add_pheromone = 0
    
    def __init__(self, colony_size = 10, pheromone_deposit_weight=10.0, initial_pheromone=1.0, step=100, nodes=None, start=None, end=None,evaporation_pheromone=5.0):
        self.colony_size = colony_size
        self.steps = step
        self.initial_pheromone = initial_pheromone
        self.nodes = [self.Node(nodes[i], i) for i in range(len(nodes))]

        self.num_nodes = len(nodes)
        self.start = start
        self.end = end
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.evaporation_pheromone = evaporation_pheromone

        self.paths = [[None] * self.num_nodes for _ in range(self.num_nodes)]   #각 지점간의 경로

        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                self.paths[i][j] = self.paths[j][i] = self.Path(haversine(self.nodes[i].position, self.nodes[j].position), initial_pheromone)
        
        print(self.paths)
        
        self.ants = [self.Ant(self.num_nodes, self.paths, i) for i in range(self.colony_size)]
        #print("개미의 속성인 path 확인:",self.ants[0].id,self.ants[0].paths, len(self.ants[0].paths))
        self.global_best_tour = None
        self.global_best_distance = 0.0
        self.global_best_ant = None
        self.global_best_tour_pheromone = 0.0

        self.ras_tour_dist = []
    
    #지점 탐색
    def go_to_node(self, start_node, step):
        for n in range(self.num_nodes-1):
            if len(self.ants[0].visited_node) == 0:
                for i in range(len(self.ants)):
                    self.ants[i].visited_node.append(self.nodes[start_node])
                    #self.ants[i].paths[start_node] = None

            for i in range(len(self.ants)):
                self.ants[i].unvisited_node = [un for un in self.nodes if un not in self.ants[i].visited_node]
                #print(self.ants[i].id,"방문한 노드의 수:", len(self.ants[i].visited_node))
                #print("방문하지 않은 노드의 수: ", len(self.ants[i].unvisited_node))

                pheromone_list = [k.pheromone for k in self.ants[i].paths[self.ants[i].visited_node[-1].pos_number] if k != None]
                #ph_list = [k.pheromone for k in self.paths[self.ants[i].visited_node[-1].pos_number] if k != None]
                #print("pheromone_list: ",pheromone_list)
                #print("PH_LIST: ", ph_list)
                prob = [i / sum(pheromone_list) for i in pheromone_list]
                #print(prob)
                next_node = np.random.choice(self.ants[i].unvisited_node, size = 1, p = prob)[0]
                next_node_idx = self.ants[i].unvisited_node.index(next_node)
                self.ants[i].paths[self.ants[i].visited_node[-1].pos_number] = None
                for j in range(len(self.ants[i].paths)):
                    if self.ants[i].paths[j] != None:
                        self.ants[i].paths[j][self.ants[i].visited_node[-1].pos_number] = None
                self.ants[i].visited_node.append(next_node)
        for i in range(len(self.ants)): 
            self.ants[i].visited_node.append(self.nodes[start_node])    #시작한 지점으로 돌아가기 위해서 다시 마지막에 추가해주는 것이다.

    #각 개미가 간 거리의 합
    def ant_total_distance(self, step, RAS=False):
        for i in range(len(self.ants)):
            for j in range(len(self.ants[i].visited_node) - 1):
                self.ants[i].visited_dist += self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].path_distance
                if RAS==True:
                    self.ras_tour_dist.append((self.ants[i], self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number]))
            if step == 0:
                self.global_best_distance = self.ants[0].visited_dist
                self.global_best_tour = self.ants[0].visited_node
                self.global_best_ant = self.ants[0]

            if self.ants[i].visited_dist < self.global_best_distance:
                self.global_best_distance = self.ants[i].visited_dist
                self.global_best_tour = self.ants[i].visited_node
                self.global_best_ant = self.ants[i]
                self.global_best_tour_pheromone = 0
                #print(self.ants[i].visited_dist)
                for j in range(len(self.ants[i].visited_node)-1):
                    self.global_best_tour_pheromone += self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone
        #print(type(tour_dist))
        if RAS==True:
            self.ras_tour_dist = sorted(self.ras_tour_dist, key=lambda x: x[1].pheromone) 
            self.ras_tour_dist = self.ras_tour_dist[:self.num_nodes]       
        #print(len(self.ras_tour_dist))
            #print(self.ants[i].id, self.ants[i].visited_dist)
        
    
    #페로몬 분비
    def add_pheromone(self, EAS=False, RAS=False):
        for i in range(len(self.ants)):
            self.ants[i].to_add_pheromone = (self.pheromone_deposit_weight / self.ants[i].visited_dist) * 1000
            for j in range(len(self.ants[i].visited_node) - 1):
                self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone
                if RAS == True:
                    if (self.ants[i], self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number]) in self.ras_tour_dist:
                        self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone +=  (self.ras_tour_dist.index((self.ants[i], self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number])) * 0.5)
        if EAS == True:
            for i in range(len(self.global_best_tour)- 1):
                self.paths[self.global_best_tour[i].pos_number][self.global_best_tour[i+1].pos_number].pheromone += self.global_best_ant.to_add_pheromone

    def mmas_ant_total_distance_and_add_pheromone(self, step):
        max_pheromone = 5
        min_pheromone = 1
        iter_tour = None
        iter_dist = 0.0     #얘네들은 개미 마다 갱신하는 반복최선해이다.
        for i in range(len(self.ants)):
            for j in range(len(self.ants[i].visited_node) - 1):
                self.ants[i].visited_dist += self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].path_distance

        if step == 0:
            self.global_best_distance = self.ants[0].visited_dist
            iter_dist = self.ants[0].visited_dist
            self.global_best_tour = self.ants[0].visited_node
            iter_tour = self.ants[0].visited_node
            self.global_best_ant = self.ants[0]

        for i in range(len(self.ants)):
            #for j in range(len(self.ants[i].visited_node) - 1):             
            #    max_pheromone = (1 * self.pheromone_deposit_weight)
            #    min_pheromone = 5

            self.ants[i].to_add_pheromone = (self.pheromone_deposit_weight / self.ants[i].visited_dist) * 1000
            max_pheromone = self.ants[i].to_add_pheromone * 100
            min_pheromone = max_pheromone * 0.1
            #반복 최선해 갱신
            if self.ants[i].visited_dist < iter_dist:
                iter_dist = self.ants[i].visited_dist
                iter_tour = self.ants[i].visited_node
                #반복 최선해의 경우 전역 최선해보다 더 적은 페로몬 양을 추가함.
                for j in range(len(self.ants[i].visited_node) - 1):                  
                    self.paths[iter_tour[j].pos_number][iter_tour[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone / 100
            

            #전역 최선해 갱신
            if self.ants[i].visited_dist < self.global_best_distance:
                self.global_best_distance = self.ants[i].visited_dist
                self.global_best_tour = self.ants[i].visited_node
                for j in range(len(self.ants[i].visited_node) - 1):
                    self.paths[self.global_best_tour[j].pos_number][self.global_best_tour[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone

            for j in range(len(self.ants[i].visited_node) - 1):
                if self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone >= max_pheromone:
                    self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone = max_pheromone
                elif self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone <= min_pheromone:
                    self.paths[self.ants[i].visited_node[j].pos_number][self.ants[i].visited_node[j+1].pos_number].pheromone = min_pheromone


    def aco(self):
        print(f"<- ACO 시작 -> {self.steps}회 반복")
        for step in range(self.steps):
            for i in range(len(self.ants)):
                self.ants[i].paths = copy.deepcopy(self.paths)
                self.ants[i].visited_node = []
                self.ants[i].visited_dist = 0
            self.go_to_node(self.start, step)
            self.ant_total_distance(step)
            self.add_pheromone()

            if step == 0:
                first_tour = self.global_best_tour
                first_distance = self.global_best_distance

            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.paths[i][j] != None:
                        self.paths[i][j].pheromone -= (0.0005 * self.evaporation_pheromone)
        print("--------------------------------------------------------------------------------\n")
        print("<결과>\n")
        print(f"시작 지점: {self.start}")
        print("순서 : {0}".format(' -> '.join(str(i.pos_number) for i in self.global_best_tour)))
        print("초기 순서 : {0}".format(' -> '.join(str(i.pos_number)for i in first_tour)))
        print(f"거리 총합 : {round(self.global_best_distance, 3)} km")
        print(f"초기 거리 총합 : {round(first_distance, 3)} km")
        print("초기 거리 대비 효율: {0}% 감소".format(round(abs((self.global_best_distance - first_distance)/first_distance)*100), 3))
        print(f"페로몬 총합 : {round(self.global_best_tour_pheromone, 3)}")
        print("\n--------------------------------------------------------------------------------")
        return self.global_best_distance, self.global_best_tour
    
    def aco_elitist(self):
        print(f"<- ACO ELITIST 시작 -> {self.steps}회 반복")
        for step in range(self.steps):
            #print("STEP: ", step+1)
            for i in range(len(self.ants)):
                self.ants[i].paths = copy.deepcopy(self.paths)
                self.ants[i].visited_node = []
                #self.ants[i].unvisited_node =  unvisited는 어차피 따로 초기화를 시켜주기 때문에 여기서 안해도됨.
                self.ants[i].visited_dist = 0
            self.go_to_node(self.start, step)
            self.ant_total_distance(step)
            self.add_pheromone(EAS=True)

            if step == 0:
                first_tour = self.global_best_tour
                first_distance = self.global_best_distance
            
            #self.add_pheromone(E=True)
            #자연에서 페로몬은 증발한다.
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.paths[i][j] != None:
                        #print("증발될 페로몬 양: ",(0.005 * self.evaporation_pheromone))
                        self.paths[i][j].pheromone -= (0.0005 * self.evaporation_pheromone)
            
        print("--------------------------------------------------------------------------------\n")
        print("<결과>\n")
        print(f"시작 지점: {self.start}")
        print("순서 : {0}".format(' -> '.join(str(i.pos_number) for i in self.global_best_tour)))
        print("초기 순서 : {0}".format(' -> '.join(str(i.pos_number)for i in first_tour)))
        print(f"거리 총합 : {round(self.global_best_distance, 3)} km")
        print(f"초기 거리 총합 : {round(first_distance, 3)} km")
        print("초기 거리 대비 효율: {0}% 감소".format(round(abs((self.global_best_distance - first_distance)/first_distance)*100), 3))
        print(f"페로몬 총합 : {round(self.global_best_tour_pheromone, 3)}")
        print("\n--------------------------------------------------------------------------------")
        return self.global_best_distance, self.global_best_tour

    def ras(self):
        print(f"<- RANK BASED ASO 시작 -> {self.steps}회 반복")
        for step in range(self.steps):
            #print("STEP: ", step+1)
            self.ras_tour_dist = []
            for i in range(len(self.ants)):
                self.ants[i].paths = copy.deepcopy(self.paths)
                self.ants[i].visited_node = []
                #self.ants[i].unvisited_node =  unvisited는 어차피 따로 초기화를 시켜주기 때문에 여기서 안해도됨.
                self.ants[i].visited_dist = 0
            self.go_to_node(self.start, step)
            self.ant_total_distance(step, RAS=True)
            self.add_pheromone(RAS=True)

            if step == 0:
                first_tour = self.global_best_tour
                first_distance = self.global_best_distance
            
            #self.add_pheromone(E=True)
            #자연에서 페로몬은 증발한다.
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.paths[i][j] != None:
                        #print("증발될 페로몬 양: ",(0.005 * self.evaporation_pheromone))
                        self.paths[i][j].pheromone -= (0.0005 * self.evaporation_pheromone)
            
        print("--------------------------------------------------------------------------------\n")
        print("<결과>\n")
        print(f"시작 지점: {self.start}")
        print("순서 : {0}".format(' -> '.join(str(i.pos_number) for i in self.global_best_tour)))
        print("초기 순서 : {0}".format(' -> '.join(str(i.pos_number)for i in first_tour)))
        print(f"거리 총합 : {round(self.global_best_distance, 3)} km")
        print(f"초기 거리 총합 : {round(first_distance, 3)} km")
        print("초기 거리 대비 효율: {0}% 감소".format(round(abs((self.global_best_distance - first_distance)/first_distance)*100), 3))
        print(f"페로몬 총합 : {round(self.global_best_tour_pheromone, 3)}")
        print("\n--------------------------------------------------------------------------------")      
        return self.global_best_distance, self.global_best_tour

    def mmas(self):
        print(f"<- MAX MIN ASO 시작 -> {self.steps}회 반복")
        for step in range(self.steps):
            #print("STEP: ", step+1)
            for i in range(len(self.ants)):
                self.ants[i].paths = copy.deepcopy(self.paths)
                self.ants[i].visited_node = []
                #self.ants[i].unvisited_node =  unvisited는 어차피 따로 초기화를 시켜주기 때문에 여기서 안해도됨.
                self.ants[i].visited_dist = 0
            self.go_to_node(self.start, step)
            #self.ant_total_distance(step, MMAS=True)
            #self.add_pheromone(MMAS=True)
            self.mmas_ant_total_distance_and_add_pheromone(step)

            if step == 0:
                first_tour = self.global_best_tour
                first_distance = self.global_best_distance

            #자연에서 페로몬은 증발한다.
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.paths[i][j] != None:
                        #print("증발될 페로몬 양: ",(0.005 * self.evaporation_pheromone))
                        self.paths[i][j].pheromone -= (0.0005 * self.evaporation_pheromone)
        print("--------------------------------------------------------------------------------\n")
        print("<결과>\n")
        print(f"시작 지점: {self.start}")
        print("순서 : {0}".format(' -> '.join(str(i.pos_number) for i in self.global_best_tour)))
        print("초기 순서 : {0}".format(' -> '.join(str(i.pos_number)for i in first_tour)))
        print(f"거리 총합 : {round(self.global_best_distance, 3)} km")
        print(f"초기 거리 총합 : {round(first_distance, 3)} km")
        print("초기 거리 대비 효율: {0}% 감소".format(round(abs((self.global_best_distance - first_distance)/first_distance)*100), 3))
        print(f"페로몬 총합 : {round(self.global_best_tour_pheromone, 3)}")
        print("\n--------------------------------------------------------------------------------")

        return self.global_best_distance, self.global_best_tour

    def draw_graph(self, aco_tour, aco_e_tour, ras_tour, mmas_tour):
        aco_x = []
        aco_y = []
        aco_e_x = []
        aco_e_y = []
        ras_x = []
        ras_y = []
        mmas_x = []
        mmas_y = []

        font = {'color' : 'blue',
                'weight': 'bold'}

        for i in range(len(aco_tour)):
            aco_x.append(aco_tour[i].position[0])
            aco_y.append(aco_tour[i].position[1])
            aco_e_x.append(aco_e_tour[i].position[0])
            aco_e_y.append(aco_e_tour[i].position[1])
            ras_x.append(ras_tour[i].position[0])
            ras_y.append(ras_tour[i].position[1])
            mmas_x.append(mmas_tour[i].position[0])
            mmas_y.append(mmas_tour[i].position[1])

        plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.plot(aco_x, aco_y, color = 'k', alpha=0.5)
        plt.scatter(aco_x, aco_y ,color='k')
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(aco_x)):
            plt.text(aco_x[i], aco_y[i], aco_tour[i].pos_number, fontdict=font)
        plt.title("ACO")

        plt.subplot(2,2,2)
        plt.scatter(aco_e_x, aco_e_y, color='k')
        plt.plot(aco_e_x, aco_e_y, color = 'k', alpha=0.5)
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(aco_e_x)):
            plt.text(aco_e_x[i], aco_e_y[i], aco_e_tour[i].pos_number, fontdict=font)
        plt.title("ACO ELITIST")


        plt.subplot(2,2,3)
        plt.scatter(ras_x, ras_y, color='k')
        plt.plot(ras_x, ras_y, color = 'k', alpha=0.5)
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(ras_x)):
            plt.text(ras_x[i], ras_y[i], ras_tour[i].pos_number, fontdict=font)
        plt.title("RAS")

        plt.subplot(2,2,4)
        plt.scatter(mmas_x, mmas_y, color='k')
        plt.plot(mmas_x, mmas_y, color = 'k', alpha=0.5)
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(mmas_x)):
            plt.text(mmas_x[i], mmas_y[i], mmas_tour[i].pos_number, fontdict=font)
        plt.title("MMAS")

        #plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------
#   애니메이션 구현

        fig = plt.figure(figsize=(10,8))
        plt.subplot(2,2,1)
        plt.xlim(-90, 90)
        plt.ylim(-180, 180)
        plt.scatter(aco_x, aco_y ,color='k')
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(aco_x)):
            plt.text(aco_x[i], aco_y[i], aco_tour[i].pos_number, fontdict=font)
        f_1 = plt.plot([], [], color = 'k')[0]
        
        
        
        list_x1 = []
        list_y1 = []
        list_x2 = []
        list_y2 = []
        list_x3 = []
        list_y3 = []
        list_x4 = []
        list_y4 = []
        def animate_aco(i):
            list_x1.append(i)
            list_y1.append(aco_y[aco_x.index(i)])
            f_1.set_data(list_x1, list_y1)

        def animate_eas(i):
            list_x2.append(i)
            list_y2.append(aco_e_y[aco_e_x.index(i)])
            f_2.set_data(list_x2, list_y2)

        def animate_ras(i):
            list_x3.append(i)
            list_y3.append(ras_y[ras_x.index(i)])
            f_3.set_data(list_x3, list_y3)

        def animate_mmas(i):
            list_x4.append(i)
            list_y4.append(mmas_y[mmas_x.index(i)])
            f_4.set_data(list_x4, list_y4)
        plt.title("ACO")
        graph_ani_aco = FuncAnimation(fig=fig, func=animate_aco, frames=aco_x, interval=300)
        plt.subplot(2,2,2)
        f_2 = plt.plot([], [], color = 'k')[0]
        plt.scatter(aco_e_x, aco_e_y, color='k')
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(aco_e_x)):
            plt.text(aco_e_x[i], aco_e_y[i], aco_e_tour[i].pos_number, fontdict=font)
        plt.title("EAS")

        graph_ani_eas = FuncAnimation(fig=fig, func=animate_eas, frames=aco_e_x, interval=300)

        plt.subplot(2,2,3)
        f_3 = plt.plot([], [], color = 'k')[0]
        plt.scatter(ras_x, ras_y, color='k')
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(ras_x)):
            plt.text(ras_x[i], ras_y[i], ras_tour[i].pos_number, fontdict=font)
        plt.title("RAS")
        graph_ani_ras = FuncAnimation(fig=fig, func=animate_ras, frames=ras_x, interval=300)


        plt.subplot(2,2,4)
        f_4 = plt.plot([], [], color = 'k')[0]
        plt.scatter(mmas_x, mmas_y, color='k')
        plt.scatter(aco_x[0], aco_y[0], color = 'r')
        for i in range(len(mmas_x)):
            plt.text(mmas_x[i], mmas_y[i], mmas_tour[i].pos_number, fontdict=font)
        graph_ani_mmas = FuncAnimation(fig=fig, func=animate_mmas, frames=mmas_x, interval=300)
        plt.title("MMAS")

        #graph_ani_aco.save('animaition_aco.gif', writer='imagemagick', dpi=100)
        #graph_ani_eas.save('animaition_eas.gif', writer='imagemagick', dpi=100)
        #graph_ani_ras.save('animaition_ras.gif', writer='imagemagick', dpi=100)
        #graph_ani_mmas.save('animaition_mmas.gif', writer='imagemagick', dpi=100)
        plt.show()

    def all_run(self):
        whole_time = time.time()
        aco_start_time = time.time()
        aco_dist, aco_tour = self.aco()
        aco_elitist_start_time = time.time()
        aco_e_dist, aco_e_tour = self.aco_elitist()
        ras_start_time = time.time()
        ras_dist, ras_tour = self.ras()
        mmas_start_time = time.time()
        mmas_dist, mmas_tour = self.mmas()
        mmas_end_time = time.time()
        whole_time2 = time.time()

        aco_run_time = aco_elitist_start_time - aco_start_time
        aco_elitist_run_time = ras_start_time - aco_elitist_start_time
        ras_run_time = mmas_start_time - ras_start_time
        mmas_run_time = mmas_end_time - mmas_start_time
        whole_run_time = whole_time2 - whole_time

        run_time_list = [('ACO', aco_run_time), ('ACO_ELITIST', aco_elitist_run_time), ('RAS', ras_run_time), ('MMAS', mmas_run_time)]
        dist_list = [('ACO', aco_dist), ('ACO_ELITIST', aco_e_dist), ('RAS', ras_dist), ('MMAS', mmas_dist)]
        run_time_list = sorted(run_time_list, key=lambda x: x[1])
        dist_list = sorted(dist_list, key=lambda x: x[1])

        print("<비교>\n")
        print("거리가 짧은 순서: {0}".format(' -> '.join(i[0] for i in dist_list)))
        print("수행 시간 순서: {0}".format(' -> '.join(i[0] for i in run_time_list)))
        for i in range(len(run_time_list)):
            print(run_time_list[i])
        print("전체 수행 시간: ", whole_run_time)

        self.draw_graph(aco_tour, aco_e_tour, ras_tour, mmas_tour)

colony_size = 20   #군체 사이즈 -> 개미의 수
step = 100
nodes_1 = [(random.uniform(-90, 90), random.uniform(-180, 180)) for i in range(0, 5)] #5개의 거점 위도, 경도를 랜덤으로 생성
start = 0
end = 0
#Paths = [[None] * len(nodes) for _ in range(len(nodes))]
#print(nodes_1)
"""
dist_nodes_1 = []
for i in range(len(nodes_1)):
    for j in range(i, len(nodes_1)):
        dist_nodes_1.append((i,j,haversine(nodes_1[i], nodes_1[j])))
for i in range(len(dist_nodes_1)):
    print(dist_nodes_1[i])
"""
evap_phe = 5.0

#nodes_2 는 테스트 데이터로 고정하겠음.

aco_problem = ACO(colony_size=colony_size, step=step, start=start, end=end,nodes=nodes_1, evaporation_pheromone=evap_phe)

aco_problem.all_run()

#보완할 것:
#만약 다시 시작 노드로 돌아올 때 경로 탐색을 어떻게 하는 게 좋은지.