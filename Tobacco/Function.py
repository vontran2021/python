    def btnOpenCamera_Clicked(self):
        '''
        打开和关闭摄像头
        '''
        self.is_camera_opened = ~self.is_camera_opened
        if self.is_camera_opened:
            self.pushButton_7.setText("关闭摄像头")
            self._timer.start()
        else:
            self.pushButton_7.setText("打开摄像头")
            self._timer.stop()

    def btnCapture_Clicked(self):
        '''
        捕获图片
        '''
        # 摄像头未打开，不执行任何操作
        if not self.is_camera_opened:
            return

        self.captured = self.frame
        # print(self.captured)
        # print(type(self.captured))


        # 后面这几行代码几乎都一样，可以尝试封装成一个函数
        rows, cols, channels = self.captured.shape
        bytesPerLine = channels * cols
        # Qt显示图片时，需要先转换成QImgage类型
        QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        self.label_6.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_6.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def btnReadImage_Clicked(self):
        '''
        从本地读取图片
        '''
        # 打开文件选取对话框
        global filename

        filename, _ = QFileDialog.getOpenFileName(self, '打开图片')
        if filename:
            self.captured = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            self.captured = cv.cvtColor(self.captured, cv.COLOR_BGR2RGB)

            rows, cols, channels = self.captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.label_3.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.label_3.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lineEdit.setText(str(filename))
        self.plainTextEdit.appendPlainText(str(filename))
        print(str(filename))

    def btnGray_Clicked(self):
        '''
        灰度化
        '''
        global GrayImage
        GrayImage = None  # 设置空变量用于储存灰度图
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            return
        self.show_processing_dialog()
        self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        GrayImage = Image.fromarray(self.cpatured.astype('uint8')).convert('RGB')

        rows, columns = self.cpatured.shape
        bytesPerLine = columns
        # 灰度图是单通道，所以需要用Format_Indexed8
        QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_4.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        QtCore.QTimer.singleShot(2000, self.close_processing_dialog)  # 延迟关闭对话框


    def btnThreshold_Clicked(self):
        '''
        Otsu自动阈值分割,需要进行灰度化处理后
        '''
        if not hasattr(self, "captured"):
            return
        self.show_processing_dialog()
        self.cpatured = cv.cvtColor(self.captured, cv.COLOR_RGB2GRAY)
        _, self.cpatured = cv.threshold(
            self.cpatured, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        rows, columns = self.cpatured.shape
        bytesPerLine = columns
        # 阈值分割图也是单通道，也需要用Format_Indexed8
        QImg = QImage(self.cpatured.data, columns, rows, bytesPerLine, QImage.Format_Indexed8)
        self.label_4.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_4.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        QtCore.QTimer.singleShot(5000, self.close_processing_dialog)  # 延迟关闭对话框

    def btnrgbHistogram_Clicked(self):
        '''
        画出RGB三通道曲线图
        '''
        if not hasattr(self, "captured"):
            return
        self.show_processing_dialog()
        # BGR转RGB
        self.captured = cv2.cvtColor(self.captured, cv2.COLOR_BGR2RGB)
        temp_img_path = rgbCurve(self.captured)
        # r, g, b = cv2.split(self.captured)
        # # 对通道值进行排序
        # r_sorted = np.sort(r.flatten())
        # g_sorted = np.sort(g.flatten())
        # b_sorted = np.sort(b.flatten())
        # # 绘制RGB三通道值曲线
        # plt.plot(r_sorted, color='red', label='Red')
        # plt.plot(g_sorted, color='green', label='Green')
        # plt.plot(b_sorted, color='blue', label='Blue')
        # plt.xlabel('Pixel')
        # plt.ylabel('Intensity')
        # plt.legend()
        # # 将绘制的曲线保存为临时图片
        # temp_img_path = './temp_plot.png'
        # plt.savefig(temp_img_path)
        # plt.close()
        # 创建标签控件并显示绘制的曲线图片
        pixmap = QPixmap(temp_img_path)
        pixmap = pixmap.scaled(400, 350)
        self.label_4.setPixmap(pixmap)
        # # 后面这几行代码几乎都一样，可以尝试封装成一个函数
        # rows, cols, channels = pixmap.shape
        # bytesPerLine = channels * cols
        # # Qt显示图片时，需要先转换成QImgage类型
        # QImg = QImage(self.captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
        # self.label_6.setPixmap(QPixmap.fromImage(QImg).scaled(
        #     self.label_6.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        QtCore.QTimer.singleShot(5000, self.close_processing_dialog)  # 延迟关闭对话框

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        '''
        循环捕获图片
        '''
        ret, self.frame = self.camera.read()

        img_rows, img_cols, channels = self.frame.shape
        bytesPerLine = channels * img_cols

        cv.cvtColor(self.frame, cv.COLOR_BGR2RGB, self.frame)
        QImg = QImage(self.frame.data, img_cols, img_rows, bytesPerLine, QImage.Format_RGB888)
        self.label_5.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.label_5.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def get_image_from_label(self, label):
        '''
        捕获label上所显示图片
        '''
        pixmap = label.pixmap()
        image = pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        image.save(buffer, "JPEG")
        buffer.seek(0)
        data = buffer.data()
        nparr = np.frombuffer(data, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        return image_pil

# 图像识别
    def predictFlueTobacco(self):
        '''
        识别烟草图像
        '''
        model_name = self.comboBox.currentText()  # 获取模型类别下拉菜单“类别字符串”
        print(model_name)

        self.predictImg = None  # 识别图像储存变量
        # predictImg = self.captured
        self.show_processing_dialog()  # 加载等待断点
        # 如果没有捕获图片，则不执行操作
        if not hasattr(self, "captured"):
            self.plainTextEdit.setPlainText("no image select, to return")  # 如果没有加载图片，则预警
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '出现错误')
            msg_box.exec_()
            return
        if not hasattr(self, "predictImg"):
            self.plainTextEdit.setPlainText("no image select, to return")
            msg_box = QMessageBox(QMessageBox.Critical, '错误', '出现错误')
            msg_box.exec_()
            return
        if self.predictImg:
            self.plainTextEdit.appendPlainText("captureImage")  # 以添加新行的方式将识别结果载入界面
            self.predictImg = self.get_image_from_label(self.label_6)  # 获取摄像图片截图
        else:
            self.predictImg = Image.fromarray(self.captured)  # 获取本地图像

        # if filename is not None:
        #     self.predictImg = Image.open(str(filename))
        # self.predictImg = cv.cvtColor(self.predictImg, cv.COLOR_BGR2RGB)
        # self.captured = self.predictImg
        # self.comboBox.setItemText(0, _translate("MyWindow", "烟叶成熟度"))
        # self.comboBox.setItemText(1, _translate("MyWindow", "烘烤阶段"))
        # self.comboBox.setItemText(2, _translate("MyWindow", "烤烟等级"))

        # 模型加载
        if model_name == "烟叶成熟度":
            predict_class = RecognitionFreshTobacco.image_predic(self.predictImg)
            self.plainTextEdit.appendPlainText(str(predict_class))  # 将结果打印在文本框中
        elif model_name == "烘烤阶段":
            predict_class = RecognitionCuringTobacco.image_predic(self.predictImg)
            self.plainTextEdit.appendPlainText(str(predict_class))  # 将结果打印在文本框中
        elif model_name == "烤烟等级":
            predict_class = RecognitionFluedTobacco.image_predic(self.predictImg)
            self.plainTextEdit.appendPlainText(str(predict_class))  # 将结果打印在文本框中
        # self.plainTextEdit.setPlainText(str(predict_class))

        # self.plainTextEdit.appendPlainText(str(predict_class))  # 将结果打印在文本框中

        QtCore.QTimer.singleShot(5000, self.close_processing_dialog)  # 延迟关闭对话框

# 打开文件夹
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            self.load_images_from_folder(folder_path)

    def load_images_from_folder(self, folder_path):
        self.images = []
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            image_extensions = (".jpg", ".jpeg", ".png", ".gif")  # 图片文件扩展名
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_name.lower().endswith(image_extensions):
                    self.images.append(file_path)
            self.current_image_index = 0
            self.show_current_image()

    def show_current_image(self):
        if self.images:
            filename = self.images[self.current_image_index]
            pixmap = QPixmap(filename)
            self.label_3.setPixmap(pixmap.scaled(400, 350, aspectRatioMode=QtCore.Qt.KeepAspectRatio))
            self.captured = cv.imread(str(filename))
            self.plainTextEdit.appendPlainText(filename)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_current_image()

# 保存图片到本地
    def saveImage(self):
        screen = QApplication.primaryScreen()
        pix = screen.grabWindow(self.label_4.winId())

        fd, type = QFileDialog.getSaveFileName(self.centralwidget, "保存图片", "", "*.jpg;;*.png;;All Files(*)")
        pix.save(fd)
        self.lineEdit.setText(str(fd))
# 全存
    def save_Alldata(self):
        # 获取label显示的图片
        pixmap_1 = self.label_4.pixmap()
        pixmap_2 = self.label_6.pixmap()

        # 获取lineEdit的文本
        text = self.plainTextEdit.toPlainText()

        # 弹出文件保存对话框，选择保存路径和文件名
        file_dialog = QFileDialog()
        file_path= file_dialog.getExistingDirectory(self, '选择文件夹')

        if file_path:
            # 保存图片
            if pixmap_1 is not None:
                pixmap_1.save(file_path + "/图像处理.jpg", "JPG")
            if pixmap_2 is not None:
                pixmap_2.save(file_path + "/抓取图.jpg", "JPG")

            # 保存文本
            if text:
                with open(file_path + "/text.txt", "w") as file:
                    file.write(text)

        msg_box = QMessageBox(QMessageBox.Information, '提示', '保存成功')
        msg_box.exec_()


# 保存文本文件
    def saveTextFile(self):
        fd, fp = QFileDialog.getSaveFileName(self.centralwidget, "保存文件", "", "*.txt;;All Files(*)")
        f = open(fd, 'w')
        f.write(self.plainTextEdit.toPlainText())
        f.close()
        self.lineEdit.setText(str(fd))
        msg_box = QMessageBox(QMessageBox.information, '提示', '保存成功')
        msg_box.exec_()

    def clearText(self):
        self.plainTextEdit.clear()

    def show_processing_dialog(self):
        self.processing_dialog = QMessageBox(self)
        self.processing_dialog.setWindowTitle("处理中")
        self.processing_dialog.setText("正在处理图像，请稍候...")
        # self.processing_dialog.setStandardButtons(QMessageBox.NoButton)
        self.processing_dialog.show()

    def close_processing_dialog(self):
        if self.processing_dialog is not None:
            self.processing_dialog.close()
            self.processing_dialog = None

    def copyWeightFile(self):
        # 获取当前程序所在的目录
        current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # 要加载的文件路径,[0]表示仅取路径
        source_file = QFileDialog.getOpenFileName(self, '加载权重', "", "*.pth")[0]
        source_file = str(source_file)
        # 获取源文件的文件名
        file_name = os.path.basename(source_file)

        # 将权重加载到当前目录
        destination_file = os.path.join(current_dir, file_name)
        self.lineEdit.setText(str(file_name))


        try:
            shutil.copy(source_file, destination_file)
            print("File copied successfully!")
        except IOError as e:
            print(f"Unable to copy file. Error: {e}")

    def copyModelFile(self):
        # 获取当前程序所在的目录
        current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

        # 要加载的文件路径
        source_file = QFileDialog.getOpenFileName(self, '加载模型', "", "*.py")[0]
        source_file = str(source_file)
        # 获取源文件的文件名
        main_name = os.path.basename(source_file)

        # 将权重加载到当前目录
        destination_file = os.path.join(current_dir, main_name)
        self.lineEdit.setText(str(main_name))
        # print(main_name)
        # print(source_file)

        try:
            shutil.copy(source_file, destination_file)
            print("File copied successfully!")
        except IOError as e:
            print(f"Unable to copy file. Error: {e}")
# **********************************************************************************************
#     def switch_model(self, index):
#         model_name = self.models_combobox.currentText()
#
#         if model_name == "ShuffleNetV2":
#             weight_file = "shufflenetv2.pth"  # 替换为ShuffleNetV2对应的权重文件
#             self.model = ShuffleNetV2()  # 替换为ShuffleNetV2的模型初始化
#         elif model_name == "MobileNetV3_small":
#             weight_file = "mobilenetv3_small.pth"  # 替换为MobileNetV3_small对应的权重文件
#             self.model = MobileNetV3_small()  # 替换为MobileNetV3_small的模型初始化
#
#         # 加载权重文件
#         self.model.load_weights(weight_file)
#
#         # 进行图像识别
#         self.recognize_image()
#
#     def recognize_image(self):
#         if self.images:
#             image_path = self.images[self.current_image_index]
#             image = QImage(image_path)
#             # 在此处执行图像识别的逻辑，使用 self.model 进行识别
#             # 将识别结果显示在相应的控件上
#
#     def start_recognition(self):
#         self.recognize_image()

# ************************************************************************************************
    # 清屏重新开始
    def Newbuild(self):
        self.plainTextEdit.clear()
        self.label_3.clear()
        self.label_4.clear()
        self.label_5.clear()
        self.label_6.clear()
        self.lineEdit.clear()
        msg_box = QMessageBox(QMessageBox.Information, '提示', '新建成功')
        msg_box.exec_()

# 帮助：跳转邮件界面
    def open_email_client(self):
        email_address = "fengchuan2021@126.com"
        url = QUrl("mailto:" + email_address)
        QDesktopServices.openUrl(url)
