from Parser import Parser as ps
from vrp_aco import *
import time,sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

form_class = uic.loadUiType("tms_aco.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("TMS")
        self.Btn1.clicked.connect(self.btn_cliked)

    def confirm_condition(self):
        #조건 확인
        global args_list
        condition_count = 19
        args_list = [[None, None] for _ in range(condition_count)] * condition_count
        if self.quantity_var.value() != None:
            args_list[0][0] = 'quantity_cost'
            args_list[0][1] = float(self.quantity_var.value())

        if self.hub_radius_var.value() != None:
            args_list[1][0] = 'hub_radius_cost'
            args_list[1][1] = float(self.hub_radius_var.value())

        if self.crossline_var.value() != None:
            args_list[2][0] = 'crossline_cost'
            args_list[2][1] = float(self.crossline_var.value())
        
        if self.car_customer_var.value() != None:
            args_list[14][0] = 'car_customer_cost'
            args_list[14][1] = float(self.car_customer_var.value())

        args_list[3][0] = 'max_capacity_percent'
        args_list[3][1] = float(self.max_capacity_cost.value())

        args_list[4][0] = 'limit_capacity'
        args_list[4][1] = float(self.limit_capacity_var.value())

        args_list[5][0] = 'limit_time'
        args_list[5][1] = float(self.limit_time_var.value())
        
        if self.max_capacity_cost_var_2.value() != None:
            args_list[6][0] = 'max_capacity_cost'
            args_list[6][1] = float(self.max_capacity_cost_var_2.value())

        if self.timewindow_var.value() != None:
            args_list[7][0] = 'timewindow_cost'
            args_list[7][1] = float(self.timewindow_var.value())

        if self.car_speed_var.value() != None:
            args_list[8][0] = 'car_speed'
            args_list[8][1] = float(self.car_speed_var.value())

        args_list[9][0] = 'loading_unloading'
        args_list[9][1] = float(self.loading_unloading_var.value())


        args_list[12][0] = 'no_over_cap'
        if self.no_over_cap.isChecked():    
            args_list[12][1] = True
        else:
            args_list[12][1] = False
        
        args_list[13][0] = 'no_over_time'        
        if self.no_over_time.isChecked():
            args_list[13][1] = True
        else:
            args_list[13][1] = False

        args_list[14][0] = 'another_car'
        if self.available_yong.isChecked():
            args_list[14][1] = True
        else:   args_list[14][1] = False

        if self.count_var.value() != None:
            args_list[15][0] = 'step'
            args_list[15][1] = self.count_var.value()
        
        if self.initial_pheromone_var.value() != None:
            args_list[16][0] = 'initial_pheromone'
            args_list[16][1] = self.initial_pheromone_var.value()

        if self.pheromone_deposit_weight.value() != None:
            args_list[17][0] = 'pheromone_deposit_weight'
            args_list[17][1] = self.pheromone_deposit_weight.value()
        
        if self.evaporation_pheromone_var.value() != None:
            args_list[18][0] = 'evaporation_pheromone'
            args_list[18][1] = self.evaporation_pheromone_var.value() 

        

    def btn_cliked(self, file_nm):
        QMessageBox.about(self, "확인", self.localBox.currentText())
        self.confirm_condition()

        file_nm = self.localBox.currentText()
        Parser = ps(file_nm)
        hub, items_list, cars_list = Parser.get_info()
        aco_problem = ACO(hub, items_list, cars_list, args_list)
        aco_problem.aco()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()
    



