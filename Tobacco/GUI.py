class Ui_MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = []  # 图片文件列表
        self.current_image_index = 0  # 当前显示的图片索引
        self.show_current_image()  # 显示当前图片
        self.model = None  # 当前使用的模型
        self.setupUi(self)

        self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.is_camera_opened = False  # 摄像头有没有打开标记

        # 定时器：30ms捕获一帧
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(30)

        self.processing_dialog = None  # "处理中..." 对话框


    def setupUi(self, MyWindow):
        MyWindow.setObjectName("MyWindow")
        MyWindow.resize(1200, 1000)
        MyWindow.setFixedSize(1200, 1000)
        self.centralwidget = QtWidgets.QWidget(MyWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(10, 890, 1190, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(810, 80, 16, 810))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(830, 120, 350, 700))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.plainTextEdit.sizePolicy().hasHeightForWidth())
        self.plainTextEdit.setSizePolicy(sizePolicy)
        self.plainTextEdit.setMinimumSize(QtCore.QSize(300, 700))

        font = QtGui.QFont()
        font.setItalic(False)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(412, 102, 400, 350))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 504, 400, 350))
        self.label_5.setStyleSheet("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(412, 504, 400, 350))
        self.label_6.setObjectName("label_6")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        self.label_3.setGeometry(QtCore.QRect(10, 102, 400, 350))
        self.label_3.setObjectName("label_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(400, 840, 75, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(980, 80, 75, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(850, 840, 75, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_4.sizePolicy().hasHeightForWidth())
        self.pushButton_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(1090, 840, 75, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(600, 10, 150, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(700, 10, 300, 30))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(200, 840, 150, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 55, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setGeometry(QtCore.QRect(150, 55, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.previous_button.setFont(font)
        self.previous_button.setObjectName("pushButton_2")
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setGeometry(QtCore.QRect(290, 55, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.next_button.setFont(font)
        self.next_button.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 150, 30))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(180, 10, 150, 30))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        MyWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MyWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 908, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menuFeedback = QtWidgets.QMenu(self.menu_4)
        self.menuFeedback.setObjectName("menuFeedback")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        MyWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MyWindow)
        self.statusbar.setObjectName("statusbar")
        MyWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MyWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionOpen = QtWidgets.QAction(MyWindow)
        self.actionOpen.setObjectName("actionOpen")



        self.actionSave = QtWidgets.QAction(MyWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionTailor = QtWidgets.QAction(MyWindow)
        self.actionTailor.setObjectName("actionTailor")
        self.actionText = QtWidgets.QAction(MyWindow)
        self.actionText.setObjectName("actionText")
        self.actionHistogram = QtWidgets.QAction(MyWindow)
        self.actionHistogram.setObjectName("actionHistogram")
        self.actionEqualization = QtWidgets.QAction(MyWindow)
        self.actionEqualization.setObjectName("actionEqualization")
        self.actionGray = QtWidgets.QAction(MyWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionThreshold = QtWidgets.QAction(MyWindow)
        self.actionThreshold.setObjectName("actionEqualization_2")
        self.actionEntropy = QtWidgets.QAction(MyWindow)
        self.actionEntropy.setObjectName("actionEntropy")
        self.actionEmail = QtWidgets.QAction(MyWindow)
        self.actionEmail.setObjectName("actionEmail")
        self.actionWeb = QtWidgets.QAction(MyWindow)
        self.actionWeb.setObjectName("actionWab")
        self.actionSetting = QtWidgets.QAction(MyWindow)
        self.actionSetting.setObjectName("actionSetting")
        self.actionModel = QtWidgets.QAction(MyWindow)
        self.actionModel.setObjectName("actionModel")
        self.actionWeight = QtWidgets.QAction(MyWindow)
        self.actionWeight.setObjectName("actionWeight")
        self.menu.addAction(self.actionNew)
        self.menu.addAction(self.actionOpen)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionSetting)
        self.menu_2.addAction(self.actionTailor)
        self.menu_2.addAction(self.actionText)
        self.menu_3.addAction(self.actionHistogram)
        self.menu_3.addAction(self.actionGray)
        self.menu_3.addAction(self.actionThreshold)
        self.menu_3.addAction(self.actionEntropy)
        self.menuFeedback.addAction(self.actionEmail)
        self.menuFeedback.addAction(self.actionWeb)
        self.menu_4.addAction(self.menuFeedback.menuAction())
        self.menu_5.addAction(self.actionModel)
        self.menu_5.addAction(self.actionWeight)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())

# *************************************信号连接****************************************************
        # 菜单栏
        self.retranslateUi(MyWindow)
        self.actionOpen.triggered.connect(self.open_folder)
        self.actionGray.triggered.connect(self.btnGray_Clicked)
        self.actionThreshold.triggered.connect(self.btnThreshold_Clicked)
        self.actionHistogram.triggered.connect(self.btnrgbHistogram_Clicked)
        self.actionSave.triggered.connect(self.save_Alldata)
        self.actionWeight.triggered.connect(self.copyWeightFile)
        self.actionModel.triggered.connect(self.copyModelFile)
        self.actionEmail.triggered.connect(self.open_email_client)

        self.actionNew.triggered.connect(self.Newbuild)

        # 内部界面
        self.pushButton_6.clicked.connect(self.btnReadImage_Clicked)
        self.pushButton_3.clicked.connect(self.predictFlueTobacco)
        self.previous_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        # self.comboBox.activated['int'].connect(MyWindow.renwe)
        # self.comboBox.currentIndexChanged.connect(MyWindow.renwe)
        self.pushButton_7.clicked.connect(self.btnOpenCamera_Clicked)
        self.pushButton_8.clicked.connect(self.btnCapture_Clicked)
        self.pushButton_4.clicked.connect(self.saveTextFile)
        self.pushButton_5.clicked.connect(self.clearText)
        # self.plainTextEdit.copyAvailable['bool'].connect(self.pushButton_4.animateClick)
        QtCore.QMetaObject.connectSlotsByName(MyWindow)
# *************************************信号连接****************************************************

        self.show_current_image()  # 显示当前图片

    def retranslateUi(self, MyWindow):
        _translate = QtCore.QCoreApplication.translate
        MyWindow.setWindowTitle(_translate("MyWindow", "基于神经网络的烟叶等级识别系统V1.0"))
        self.plainTextEdit.setPlainText(_translate("MyWindow", "识别结果："))
        self.plainTextEdit.setReadOnly(True)

        self.label_4.setText(_translate("MyWindow", "<html><head/><body><p align=\"center\">图像处理结果</p></body></html>"))
        self.label_5.setText(_translate("MyWindow", "<html><head/><body><p align=\"center\">摄像头</p></body></html>"))
        self.label_6.setText(_translate("MyWindow", "<html><head/><body><p align=\"center\">抓取图</p></body></html>"))
        self.label_3.setText(_translate("MyWindow", "<html><head/><body><p align=\"center\">图像</p></body></html>"))
        self.pushButton_8.setText(_translate("MyWindow", "抓图"))
        self.pushButton_3.setText(_translate("MyWindow", "识别"))
        self.pushButton_4.setText(_translate("MyWindow", "导出"))
        self.pushButton_5.setText(_translate("MyWindow", "清除"))
        self.label_2.setText(_translate("MyWindow", "文件地址："))
        self.pushButton_7.setText(_translate("MyWindow", "打开摄像头"))
        self.pushButton_6.setText(_translate("MyWindow", "浏览"))
        self.previous_button.setText(_translate("MyWindow", "上一张"))
        self.next_button.setText(_translate("MyWindow", "下一张"))
        self.label.setText(_translate("MyWindow",
                                      "<html><head/><body><p><span style=\" font-weight:600; color:#000000;\">选择识别类型：</span></p></body></html>"))
        self.comboBox.setItemText(0, _translate("MyWindow", "烟叶成熟度"))
        self.comboBox.setItemText(1, _translate("MyWindow", "烘烤阶段"))
        self.comboBox.setItemText(2, _translate("MyWindow", "烤烟等级"))
        self.menu.setTitle(_translate("MyWindow", "文件"))
        self.menu_2.setTitle(_translate("MyWindow", "编辑"))
        self.menu_3.setTitle(_translate("MyWindow", "工具"))
        self.menu_4.setTitle(_translate("MyWindow", "帮助"))
        self.menuFeedback.setTitle(_translate("MyWindow", "反馈"))
        self.menu_5.setTitle(_translate("MyWindow", "加载"))
        self.actionNew.setText(_translate("MyWindow", "新建"))
        self.actionOpen.setText(_translate("MyWindow", "打开"))
        self.actionSave.setText(_translate("MyWindow", "保存"))
        self.actionTailor.setText(_translate("MyWindow", "裁剪"))
        self.actionText.setText(_translate("MyWindow", "文本标注"))
        self.actionHistogram.setText(_translate("MyWindow", "直方图"))
        self.actionEqualization.setText(_translate("MyWindow", "均衡化"))
        self.actionGray.setText(_translate("MyWindow", "灰度化"))
        self.actionThreshold.setText(_translate("MyWindow", "阈值分割"))
        self.actionEntropy.setText(_translate("MyWindow", "求熵"))
        self.actionEmail.setText(_translate("MyWindow", "Email"))
        self.actionWeb.setText(_translate("MyWindow", "Wab"))
        self.actionSetting.setText(_translate("MyWindow", "设置"))
        self.actionModel.setText(_translate("MyWindow", "模型"))
        self.actionWeight.setText(_translate("MyWindow", "权重"))
