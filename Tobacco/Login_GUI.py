class Ui_MyWindow1(QtWidgets.QMainWindow):

    def __init__(self):
        super(Ui_MyWindow1, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)

    def setupUi(self, MyWindow):
        MyWindow.setObjectName("MainWindow")
        MyWindow.resize(600, 300)
        # MyWindow.setWindowIcon(QIcon('swu.png'))
        MyWindow.setStyleSheet("background-image:url(backgrand.png)")
        self.centralwidget = QtWidgets.QWidget(MyWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 100, 50, 30))
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(240, 100, 200, 30))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 150, 40, 18))
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(240, 150, 200, 30))
        self.lineEdit_2.setText("")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButtonOK = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonOK.setGeometry(QtCore.QRect(220, 200, 75, 23))
        self.pushButtonOK.setObjectName("pushButtonOK")
        # self.pushButtonOK.setStyleSheet("QPushButton{font - family: '宋体';font - size: 30px;color: rgb(0, 0, 255, 255);} \
        # QPushButtonbackground - color: rgb(170, 200, 50)}\ QPushButton: hover{background - color: rgb(50, 170, 200)}")

        self.pushButtonCancel = QtWidgets.QPushButton(self.centralwidget)
        self.pushButtonCancel.setGeometry(QtCore.QRect(320, 200, 75, 23))
        self.pushButtonCancel.setObjectName("pushButtonCancel")
        MyWindow.setCentralWidget(self.centralwidget)

        self.pushButtonOK.clicked.connect(self.user_login)
        self.pushButtonCancel.clicked.connect(MyWindow.close)
        self.retranslateUi(MyWindow)
        QtCore.QMetaObject.connectSlotsByName(MyWindow)

    def retranslateUi(self, MyWindow):
        _translate = QtCore.QCoreApplication.translate
        MyWindow.setWindowTitle(_translate("MainWindow", "基于神经网络的烟叶识别系统V1.0"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "请输入帐号"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "请输入密码"))
        self.label.setText(_translate("MainWindow", "账号"))
        self.label_2.setText(_translate("MainWindow", "密码"))
        self.pushButtonOK.setText(_translate("MainWindow", "确定"))
        # self.pushButtonOK.setStyleSheet("QPushButton{font - family: '宋体';font - size: 30px;color: rgb(0, 0, 255, 255);} \
        #         QPushButtonbackground - color: rgb(170, 200, 50)}\ QPushButton: hover{background - color: rgb(50, 170, 200)}")
        self.pushButtonCancel.setText(_translate("MainWindow", "取消"))

    def user_login(self):
        usr_name = self.lineEdit.text()
        usr_pwd = self.lineEdit_2.text()
        if usr_name == 'XNDX' and usr_pwd == '123456':
            QMessageBox.information(self, '消息', '登录成功')

            ui_hello.show()
            myWindow.close()
        else:
            # QMessageBox.information(self, '消息', '账号或密码错误')
            QMessageBox.warning(self,
                                "警告",
                                "登录失败！",
                                QMessageBox.Yes)

            self.lineEdit.setFocus()
