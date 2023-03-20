from aco import * 
import os
import pandas as pd
import csv

class Parser:
    def __init__(self, file):
        self.directory = 'aco_data/'
        self.file = self.directory + file + '/'     #aco_data/1160파주/
        self.hub = file[:4]

    def get_info(self):
        file_list = os.listdir(self.file) 

        for file in file_list:
            if 'hub' in file:
                self.hub_file = self.file + file 
            elif 'car' in file:
                self.car_file = self.file + file 
            elif 'item2' in file:
                self.item_file = self.file + file 

        print("<사용한 파일 경로>")
        print("HUB INFO: ", self.hub_file, '\n', "CAR INFO: ", self.car_file, '\n', "ITEM INFO: ", self.item_file)

        hub_info = pd.read_csv(self.hub_file, encoding='cp949', delimiter='\t')
        car_info = pd.read_csv(self.car_file, encoding='utf-8', delimiter='\t')
        item_info = pd.read_csv(self.item_file)
        cars_list = []
        items_list = []
        hub = hub_info['허브'][0]
        
        for i in range(len(item_info)):
            items_list.append([item_info.iloc[i][0], item_info.iloc[i][2], item_info.iloc[i][3], (item_info.iloc[i][8], item_info.iloc[i][9])])
        
        for i in range(len(car_info)):
            cars_list.append([car_info['허브'][i], car_info['차량ID'][i], car_info['용차'][i], car_info['운행시작'][i], car_info['운행종료'][i], car_info['CAP'][i]])

        return hub, items_list, cars_list