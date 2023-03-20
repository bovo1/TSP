import random, math, copy, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from haversine import haversine

class ACO:  #ACO는 자연현상 그 자체인...? 클래스로 보면 이해하기 좋을 것 같다.

    class Item: #아이템
        def __init__(self, num, items_list):
            self.position = items_list[3] #튜플 형태
            self.pos_number = num   #지점 번호
            self.item_cbm = items_list[2]    #아이템 무게
            self.loaded = False


    class Path:     #Path는 경로? 지점과 지점 사이의 경로라고 보면 된다.
        def __init__(self, path_distance, initial_pheromone, item1, item2):
            self.path_distance = path_distance
            self.pheromone = initial_pheromone
            self.max_pheromone = 0
            self.min_pheromone = 0
            self.item1 = item1
            self.item2 = item2

    class Ant:  #경로와 지점을 오가는 개미 !
        def __init__(self, num_nodes, paths, cars_list):
            self.num_items = num_nodes
            self.paths = copy.deepcopy(paths) #전체 거점들
            #self.paths = copy.copy(paths)
            self.visited_items = []    #이미 갔던 지점
            self.unvisited_items = []    #안 간 지점
            self.visited_dist = 0
            self.id = cars_list[1]
            self.to_add_pheromone = 0
            self.cbm = 0  #각 개미가 적재하고 있는 용량
            self.capacity = cars_list[5]
            self.hub = cars_list[0]
            self.travel_time = 0
            self.whole_travel_time = int(cars_list[4]) - int(cars_list[3])
            self.cross_count = 0
            self.item_cross_count = 0

    
    def __init__(self, hub, items_lists, cars_lists, args):
        self.hub = hub
        if args[14][1] == True:
            self.colony_size = len(cars_lists)
        else:
            self.colony_size = len([i for i in cars_lists if i[2] != 'X'])
        
        #print(items_lists)
        #print(cars_lists)
        #print(args)

        self.steps = args[15][1]
        self.initial_pheromone = args[16][1]
        self.items = [self.Item(i, items_lists[i]) for i in range(len(items_lists))]
        self.num_items = len(items_lists)
        self.start = 0 #hub -> 0
        self.end = 0  #hub    -> 0
        self.pheromone_deposit_weight = args[17][1]
        self.evaporation_pheromone = args[18][1]
        self.no_over_capacity = args[12][1]
        self.no_over_time = args[13][1]
        self.ant_speed = args[8][1]
        self.paths = [[None] * self.num_items for _ in range(self.num_items)]  #각 지점간의 경로

        for i in range(self.num_items):
            for j in range(i+1, self.num_items):
                self.paths[i][j] = self.paths[j][i] = self.Path(haversine(self.items[i].position, self.items[j].position), self.initial_pheromone, self.items[i], self.items[j])
        

        self.ants = [self.Ant(self.num_items, self.paths, cars_lists[i]) for i in range(self.colony_size)]
        self.global_best_tour = None
        self.global_best_distance = 0.0
        self.global_best_ant = None
        self.global_best_tour_pheromone = 0.0
        self.ras_tour_dist = []
        self.loading_unloading_time = args[9][1]
        self.max_capacity = args[3][1]
        self.limit_capacity = args[4][1]
        self.limit_time = args[5][1]
        self.qunatity_cost = args[0][1]
        self.hub_radius_cost = args[1][1]
        self.crossline_cost = args[2][1]
        self.car_customer_dist_cost = args[14][1]
        self.max_capacity_cost = args[6][1]
        self.timewindow_cost = args[7][1]
        self.loading_unloading_time = args[9][1]
        self.al_cross_var = 3
        

        """
        자 생각을 해보자. ACO는 마아아아악 돌다가 가장 좋다고 판단되는 루트
        근데 vrp 에서는 루트가 여러개인거지
        그러면 가장 좋다고 판단되는 루트를 어떻게 판단하는가?
        사전에 설정한 조건들
        그러니까 음... 거리가 제일 짧고, 시간도 짧고, 교차도 적고, 그런 거에 이제 맞으면 가장 좋은 루트지!
        """
        #물품을 차에 넣어주는 함수 (중복되는 구간이 많으므로 함수를 만들었다.)
    def assign(self, car, item, cost):
        if item.loaded == False:
            car.dist_cost += car.route[-2].distanceTo(item)
            car.travel_time += haversine(car.route[-2].position, item.position, unit='km') / self.ant_speed
            car.total_cost += cost
            item.loaded = True
            car.customers += [item]
            car.travel_time += (self.loading_unloading_time / 60) * len(item.customers_list)
            car.real_cbm += item.cbm
            #unloaded.remove(item)

    def find_path(self, i, j):
        for p1 in range(len(self.ants[i].paths)):
            for p2 in range(len(self.ants[i].paths[p1])):
                if self.ants[i].paths[p1][p2].item1 == self.ants[i].visited_items[-2] and self.ants[i].paths[p1][p2].item2 == self.ants[i].unvisited_items[j]:
                    return self.ants[i].paths[p1][p2]

        #시간 조건
    def time_condition(self, i,j):
        if self.ants[i].unvisited_items[j].loaded == False:
            #이 거리는 경우의 수에 따라 임시적이고 가변적인 값이 될 수 있으므로
            #차의 속성이 아닌 지역 변수와 리스트에 보관해주도록 한다.
            dist = haversine(self.ants[i].visited_items[-2].position, self.ants[i].unvisited_items[j].position)
            dist_time = dist / self.ant_speed   #km/h 기준
            item_loading_time = (self.loading_unloading_time /60)
            total_time = dist_time + item_loading_time
            car_times = [dist, total_time]
        else:
            car_times = None
        return car_times

    def ccw(self, a, b, c):
        #a -> (1, 2)
        #return 값이 양수면 반시계 방향, 음수면 시계방향, 0 이면 평행
        #return a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - (b[0]*a[1] + c[0]*b[1] + a[0]*c[1])
        result = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

        if result > 0:
            return 1
        elif result < 0:
            return -1
        else:
            return 0

    #선이 교차하는지 여부
    def intersect_line(self, a,b,c,d):
        if self.ccw(a,b,c) * self.ccw(a,b,d) < 0:
            #교차
            if self.ccw(c,d,a) * self.ccw(c,d,b) < 0:
                return True
        else:
            return False
    
    #지점 탐색
    #조건에 만족해야 다음 지점에 선택될 수 있다.
    def go_to_node(self, start_node, step):
        for n in range(self.num_items-1):
            if len(self.ants[0].visited_items) == 0:
                for i in range(len(self.ants)):
                    self.ants[i].visited_items.append(self.items[start_node])
                    print(self.ants[i].visited_items)

            for i in range(len(self.ants)):
                self.ants[i].unvisited_items = [un for un in self.items if un not in self.ants[i].visited_items]
                print(self.ants[i].unvisited_items)
                #조건 만족 검사
                for j in range(len(self.ants[i].unvisited_items)):
                    if self.ants[i].cbm + self.ants[i].unvisited_items[j].cbm <= self.ants[i].capacity:
                        #용량 체크
                        if self.no_over_capacity == True:
                            if self.ants[i].cbm + self.ants[i].unvisited_items[j].cbm > self.ants[i].capacity * self.max_capacity:
                                #만약 제한 용량을 넘는다면 해당 개미의 경로에 페로몬을 감소시킴.
                                over_capacity = (self.ants[i].cbm + self.ants[i].unvisited_items[j].cbm) - (self.ants[i].capacity * self.max_capacity)
                                for p1 in range(len(self.ants[i].paths)):
                                    for p2 in range(len(self.ants[i].paths[p1])):
                                        if self.ants[i].paths[p1][p2].item1 == self.ants[i].visited_items[-2] and self.ants[i].paths[p1][p2].item2 == self.ants[i].unvisited_items[j]:
                                            self.ants[i].paths[p1][p2].pheromone -= (over_capacity * self.max_capacity_cost)

                        else:
                            if self.ants[i].cbm + self.ants[i].unvisited_items[j].cbm > self.ants[i].capacity - self.limit_capacity:
                                for p1 in range(len(self.ants[i].paths)):
                                    for p2 in range(len(self.ants[i].paths[p1])):
                                        if self.ants[i].paths[p1][p2].item1 == self.ants[i].visited_items[-2] and self.ants[i].paths[p1][p2].item2 == self.ants[i].unvisited_items[j]:
                                            self.ants[i].paths[p1][p2].pheromone = -math.inf               
                        #시간 체크
                        if self.no_over_time == True:
                            ant_times = []
                            ant_time = self.time_condition(i,j)
                            #ant_time = [dist, total_time]
                            if ant_time != None:
                                if self.ants[i].travel_time + ant_time[1] > self.ants[i].whole_travel_time:
                                    this_path = self.find_path(i,j)
                                    this_path.pheromone = -math.inf
                            else:
                                if self.ants[i].travel_time + ant_time[1] > self.ants[i].whole_travel_time + self.limit_time:
                                    this_path = self.find_path(i,j)
                                    this_path.pheromone = -math.inf
                        
                        #교차 체크
                        self.ants[i].item_cross_count = 0
                        for a in range(len(self.ants)):
                            for b in range(len(self.ants[a].visited_items - 1)):
                                if self.intersect_line(self.ants[a].visited_items[b].position, self.ants[a].visited_items[b+1].position, self.ants[i].visited_items[-1].position, self.ants[i].unvisited_items[j].position) == True:
                                    #교차O
                                    self.ants[i].item_cross_count += 1
                                    this_path = self.find_path(i,j)
                                    this_path.pheromone -= (self.ants[i].item_cross_count * self.crossline_cost)


                pheromone_list = [k.pheromone for k in self.ants[i].paths[self.ants[i].visited_items[-1].pos_number] if k != None]

                prob = [i / sum(pheromone_list) for i in pheromone_list]
                next_node = np.random.choice(self.ants[i].unvisited_items, size = 1, p = prob)[0]
                next_node_idx = self.ants[i].unvisited_items.index(next_node)
                self.ants[i].paths[self.ants[i].visited_items[-1].pos_number] = None
                for j in range(len(self.ants[i].paths)):
                    if self.ants[i].paths[j] != None:
                        self.ants[i].paths[j][self.ants[i].visited_items[-1].pos_number] = None
                self.ants[i].visited_items.append(next_node)
        for k in range(len(self.ants)): 
            self.ants[k].visited_items.append(self.items[start_node])    #시작한 지점으로 돌아가기 위해서 다시 마지막에 추가해주는 것이다.
            


    #각 개미가 간 거리의 합
    def ant_total_distance(self, step, RAS=False):
        for i in range(len(self.ants)):
            for j in range(len(self.ants[i].visited_items) - 1):
                self.ants[i].visited_dist += self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].path_distance
                if RAS==True:
                    self.ras_tour_dist.append((self.ants[i], self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number]))
            if step == 0:
                self.global_best_distance = self.ants[0].visited_dist
                self.global_best_tour = self.ants[0].visited_items
                self.global_best_ant = self.ants[0]

            if self.ants[i].visited_dist < self.global_best_distance:
                self.global_best_distance = self.ants[i].visited_dist
                self.global_best_tour = self.ants[i].visited_items
                self.global_best_ant = self.ants[i]
                self.global_best_tour_pheromone = 0
                for j in range(len(self.ants[i].visited_items)-1):
                    self.global_best_tour_pheromone += self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone
        if RAS==True:
            self.ras_tour_dist = sorted(self.ras_tour_dist, key=lambda x: x[1].pheromone) 
            self.ras_tour_dist = self.ras_tour_dist[:self.num_items]       

        
    
    #페로몬 분비
    def add_pheromone(self, EAS=False, RAS=False):
        for i in range(len(self.ants)):
            self.ants[i].to_add_pheromone = (self.pheromone_deposit_weight / self.ants[i].visited_dist) * 1000
            for j in range(len(self.ants[i].visited_items) - 1):
                self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone
                if RAS == True:
                    if (self.ants[i], self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number]) in self.ras_tour_dist:
                        self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone +=  (self.ras_tour_dist.index((self.ants[i], self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number])) * 0.5)
        if EAS == True:
            for i in range(len(self.global_best_tour)- 1):
                self.paths[self.global_best_tour[i].pos_number][self.global_best_tour[i+1].pos_number].pheromone += self.global_best_ant.to_add_pheromone

    def mmas_ant_total_distance_and_add_pheromone(self, step):
        max_pheromone = 5
        min_pheromone = 1
        iter_tour = None
        iter_dist = 0.0     #얘네들은 개미 마다 갱신하는 반복최선해이다.
        for i in range(len(self.ants)):
            for j in range(len(self.ants[i].visited_items) - 1):
                self.ants[i].visited_dist += self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].path_distance

        if step == 0:
            self.global_best_distance = self.ants[0].visited_dist
            iter_dist = self.ants[0].visited_dist
            self.global_best_tour = self.ants[0].visited_items
            iter_tour = self.ants[0].visited_items
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
                iter_tour = self.ants[i].visited_items
                #반복 최선해의 경우 전역 최선해보다 더 적은 페로몬 양을 추가함.
                for j in range(len(self.ants[i].visited_items) - 1):                  
                    self.paths[iter_tour[j].pos_number][iter_tour[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone / 100
            

            #전역 최선해 갱신
            if self.ants[i].visited_dist < self.global_best_distance:
                self.global_best_distance = self.ants[i].visited_dist
                self.global_best_tour = self.ants[i].visited_items
                for j in range(len(self.ants[i].visited_items) - 1):
                    self.paths[self.global_best_tour[j].pos_number][self.global_best_tour[j+1].pos_number].pheromone += self.ants[i].to_add_pheromone

            for j in range(len(self.ants[i].visited_items) - 1):
                if self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone >= max_pheromone:
                    self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone = max_pheromone
                elif self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone <= min_pheromone:
                    self.paths[self.ants[i].visited_items[j].pos_number][self.ants[i].visited_items[j+1].pos_number].pheromone = min_pheromone


    def aco(self):
        print(f"<- ACO 시작 -> {self.steps}회 반복")
        for step in range(self.steps):
            for i in range(len(self.ants)):
                self.ants[i].paths = copy.deepcopy(self.paths)
                self.ants[i].visited_items = []
                self.ants[i].visited_dist = 0
            self.go_to_node(self.start, step)
            for k in range(len(self.ants)):
                print(self.ants[k].visited_items)
            self.ant_total_distance(step)
            self.add_pheromone()

            if step == 0:
                first_tour = self.global_best_tour
                first_distance = self.global_best_distance

            for i in range(self.num_items):
                for j in range(i+1, self.num_items):
                    if self.paths[i][j] != None:
                        self.paths[i][j].pheromone -= (0.0005 * self.evaporation_pheromone)
        print("--------------------------------------------------------------------------------\n")
        print("<결과>\n")
        print(f"시작 지점: {self.start}")
        print(self.ants)
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
                self.ants[i].visited_items = []
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
            for i in range(self.num_items):
                for j in range(i+1, self.num_items):
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
                self.ants[i].visited_items = []
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
            for i in range(self.num_items):
                for j in range(i+1, self.num_items):
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
                self.ants[i].visited_items = []
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
            for i in range(self.num_items):
                for j in range(i+1, self.num_items):
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

#colony_size = 50    #군체 사이즈 -> 개미의 수
#step = 200
#nodes_1 = [(random.uniform(-90, 90), random.uniform(-180, 180)) for i in range(0, 40)] #5개의 거점 위도, 경도를 랜덤으로 생성
#start = 0
#end = 0
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
#evap_phe = 5.0



#aco_problem = ACO(colony_size=colony_size, step=step, start=start, end=end,nodes=nodes_1, evaporation_pheromone=evap_phe)

#aco_problem.all_run()

#보완할 것:
#만약 다시 시작 노드로 돌아올 때 경로 탐색을 어떻게 하는 게 좋은지.