# -*- coding: utf-8 -*-
import time
import sys
from PyQt5.QtGui import QFont,QPixmap
from PyQt5.QtCore import QDir, Qt, QFile
from PyQt5.QtWidgets import QApplication, QGridLayout, QLineEdit,QPushButton, QWidget, QGraphicsScene,QTextBrowser,QVBoxLayout,QFileDialog,QGraphicsView,QLabel
from PyQt5.Qt import QHBoxLayout

from preprocessImg import ProcessImg
from predict import PredictImg
from pyasn1.compat.octets import null

class Window(QWidget):
    img_path = ''
    
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        
        self.setFixedSize(700,500)
        self.edit_path = QLineEdit()
        self.button_browse = QPushButton('选取图片')
        self.view_img = QGraphicsView()
        self.view_text = QTextBrowser()
        self.scene = QGraphicsScene()
        self.pixmap = QPixmap()
        self.label1 = QLabel('识别图片')
        self.label1.setAlignment(Qt.AlignCenter)
        self.label2 = QLabel('识别结果')
        self.label2.setAlignment(Qt.AlignCenter)
        self.label1.setFont(QFont("Roman times",12,QFont.Bold))
        self.label2.setFont(QFont("Roman times",12,QFont.Bold))
        mainLayout = QGridLayout()
        topLayout = QHBoxLayout()
        bottomLayout = QGridLayout()
        
        topLayout.addWidget(self.edit_path)
        topLayout.addWidget(self.button_browse)
        bottomLayout.addWidget(self.label1,0,0)
        bottomLayout.addWidget(self.label2,0,1)
        bottomLayout.addWidget(self.view_img,1,0)
        bottomLayout.addWidget(self.view_text,1,1)
        mainLayout.addLayout(topLayout, 0, 1)
        mainLayout.setSpacing(10)  # 设置各组件间的距离
        mainLayout.addLayout(bottomLayout, 1, 1)
        mainLayout.setRowStretch(0,1)  # 设置因子
        mainLayout.setRowStretch(1,4)  # 设置因子
        
        mainLayout.setContentsMargins(5, 15, 5, 15) # int left, int top, int right, int bottom
        self.setLayout(mainLayout)
        self.view_text.setText("Welcome,please choice a dog image")
        self.setWindowTitle("宠物狗识别")
        self.view_text.setFont(QFont("Roman times",11,QFont.Bold))

        self.button_browse.clicked.connect(self.browse)  # 使用信号与槽将按键事件与函数关联
        
    def browse(self):
        
        file = QFileDialog.getOpenFileName(self, "选取文件",  "E:/", "Images (*.jpg)")
        print(file)
        filepath = file[0]
        if filepath != '' :  # 文件不为空
            self.edit_path.setText(filepath)
            Window.img_path = filepath
            pixmap = QPixmap(Window.img_path)
            self.pixmap = pixmap.scaled(self.view_img.size(),Qt.KeepAspectRatio)
            self.scene.clear()
            self.scene.addPixmap(self.pixmap)
            
            
            self.view_img.setScene(self.scene)
            
            self.recognize()
        #------------------------------------ 

    
    def recognize(self):
        process = ProcessImg(Window.img_path)
        array = process.resize()
        print(array)
        predict  = PredictImg(array)
        index1,prob1,index2,prob2,index3,prob3 = predict.find_array_max()
        name1 = predict.show_label_name(index1)
        name2 = predict.show_label_name(index2)
        name3 = predict.show_label_name(index3)
        print(index1,prob1)
        print(index2,prob2)
        print(index3,prob3)
        #self.view_text.setText('识别结果：\n\n   种类: '+str(index)+':'+name+'\n\n'+'   概率: '+str(prob)) #
        self.view_text.setText('预测结果1:\n'+'宠物狗种类: '+name1+'\n预测概率: '+str(round(prob1,3))+'\n\n'
                               '预测结果2:\n'+'宠物狗种类: '+name2+'\n预测概率: '+str(round(prob2,3))+'\n\n'
                               '预测结果3:\n'+'宠物狗种类: '+name3+'\n预测概率: '+str(round(prob3,3))) #识别结果：\n\n
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
    
    
    
    
    
    
    
    