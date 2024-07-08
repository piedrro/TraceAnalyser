# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\turnerp\PycharmProjects\TraceAnalyser\src\TraceAnalyser\GUI\GUI_windows\plotsettings_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(396, 575)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.gridLayout_16 = QtWidgets.QGridLayout()
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.plot_normalise = QtWidgets.QCheckBox(Form)
        self.plot_normalise.setObjectName("plot_normalise")
        self.gridLayout_16.addWidget(self.plot_normalise, 1, 1, 1, 1)
        self.plot_split_lines = QtWidgets.QCheckBox(Form)
        self.plot_split_lines.setObjectName("plot_split_lines")
        self.gridLayout_16.addWidget(self.plot_split_lines, 1, 0, 1, 1)
        self.plot_showx = QtWidgets.QCheckBox(Form)
        self.plot_showx.setChecked(True)
        self.plot_showx.setObjectName("plot_showx")
        self.gridLayout_16.addWidget(self.plot_showx, 0, 0, 1, 1)
        self.plot_showy = QtWidgets.QCheckBox(Form)
        self.plot_showy.setChecked(True)
        self.plot_showy.setObjectName("plot_showy")
        self.gridLayout_16.addWidget(self.plot_showy, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_16)
        self.show_detected_states = QtWidgets.QCheckBox(Form)
        self.show_detected_states.setChecked(True)
        self.show_detected_states.setObjectName("show_detected_states")
        self.verticalLayout.addWidget(self.show_detected_states)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.label_8 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.verticalLayout.addWidget(self.label_8)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.plot_show_correction_factors = QtWidgets.QCheckBox(Form)
        self.plot_show_correction_factors.setChecked(True)
        self.plot_show_correction_factors.setObjectName("plot_show_correction_factors")
        self.gridLayout_4.addWidget(self.plot_show_correction_factors, 0, 0, 1, 1)
        self.plot_show_ml_predictions = QtWidgets.QCheckBox(Form)
        self.plot_show_ml_predictions.setChecked(True)
        self.plot_show_ml_predictions.setObjectName("plot_show_ml_predictions")
        self.gridLayout_4.addWidget(self.plot_show_ml_predictions, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem1)
        self.label_9 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.verticalLayout.addWidget(self.label_9)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.crop_reset_active = QtWidgets.QPushButton(Form)
        self.crop_reset_active.setObjectName("crop_reset_active")
        self.gridLayout_5.addWidget(self.crop_reset_active, 0, 1, 1, 1)
        self.show_bleach_range = QtWidgets.QCheckBox(Form)
        self.show_bleach_range.setChecked(True)
        self.show_bleach_range.setObjectName("show_bleach_range")
        self.gridLayout_5.addWidget(self.show_bleach_range, 2, 0, 1, 1)
        self.show_measurement_range = QtWidgets.QCheckBox(Form)
        self.show_measurement_range.setChecked(True)
        self.show_measurement_range.setObjectName("show_measurement_range")
        self.gridLayout_5.addWidget(self.show_measurement_range, 1, 0, 1, 1)
        self.show_gamma_range = QtWidgets.QCheckBox(Form)
        self.show_gamma_range.setChecked(True)
        self.show_gamma_range.setObjectName("show_gamma_range")
        self.gridLayout_5.addWidget(self.show_gamma_range, 3, 0, 1, 1)
        self.show_crop_range = QtWidgets.QCheckBox(Form)
        self.show_crop_range.setChecked(True)
        self.show_crop_range.setObjectName("show_crop_range")
        self.gridLayout_5.addWidget(self.show_crop_range, 0, 0, 1, 1)
        self.crop_reset_all = QtWidgets.QPushButton(Form)
        self.crop_reset_all.setObjectName("crop_reset_all")
        self.gridLayout_5.addWidget(self.crop_reset_all, 0, 2, 1, 1)
        self.measurment_reset_active = QtWidgets.QPushButton(Form)
        self.measurment_reset_active.setObjectName("measurment_reset_active")
        self.gridLayout_5.addWidget(self.measurment_reset_active, 1, 1, 1, 1)
        self.bleach_reset_active = QtWidgets.QPushButton(Form)
        self.bleach_reset_active.setObjectName("bleach_reset_active")
        self.gridLayout_5.addWidget(self.bleach_reset_active, 2, 1, 1, 1)
        self.gamma_reset_active = QtWidgets.QPushButton(Form)
        self.gamma_reset_active.setObjectName("gamma_reset_active")
        self.gridLayout_5.addWidget(self.gamma_reset_active, 3, 1, 1, 1)
        self.measurment_reset_all = QtWidgets.QPushButton(Form)
        self.measurment_reset_all.setObjectName("measurment_reset_all")
        self.gridLayout_5.addWidget(self.measurment_reset_all, 1, 2, 1, 1)
        self.bleach_reset_all = QtWidgets.QPushButton(Form)
        self.bleach_reset_all.setObjectName("bleach_reset_all")
        self.gridLayout_5.addWidget(self.bleach_reset_all, 2, 2, 1, 1)
        self.gamma_reset_all = QtWidgets.QPushButton(Form)
        self.gamma_reset_all.setObjectName("gamma_reset_all")
        self.gridLayout_5.addWidget(self.gamma_reset_all, 3, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_5)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem2)
        self.label_6 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.plot_downsample = QtWidgets.QComboBox(Form)
        self.plot_downsample.setObjectName("plot_downsample")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.plot_downsample.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plot_downsample)
        self.verticalLayout.addLayout(self.formLayout_2)
        spacerItem3 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem3)
        self.label_4 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.crop_plots = QtWidgets.QCheckBox(Form)
        self.crop_plots.setObjectName("crop_plots")
        self.gridLayout.addWidget(self.crop_plots, 1, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        spacerItem4 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem4)
        self.label_3 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.plot_user_group = QtWidgets.QComboBox(Form)
        self.plot_user_group.setObjectName("plot_user_group")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.plot_user_group.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.plot_user_group)
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.plot_measurement_label = QtWidgets.QComboBox(Form)
        self.plot_measurement_label.setEditable(False)
        self.plot_measurement_label.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.plot_measurement_label.setObjectName("plot_measurement_label")
        self.plot_measurement_label.addItem("")
        self.plot_measurement_label.addItem("")
        self.plot_measurement_label.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.plot_measurement_label)
        self.verticalLayout.addLayout(self.formLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "General Settings"))
        self.plot_normalise.setText(_translate("Form", "Normalise Data [N]"))
        self.plot_split_lines.setText(_translate("Form", "Split Plots With Multiple Lines [S]"))
        self.plot_showx.setText(_translate("Form", "Show X axis [X]"))
        self.plot_showy.setText(_translate("Form", "Show Y axis [Y]"))
        self.show_detected_states.setText(_translate("Form", "Show Detected States"))
        self.label_8.setText(_translate("Form", "Text Overlay Settings"))
        self.plot_show_correction_factors.setText(_translate("Form", "Correction Factors"))
        self.plot_show_ml_predictions.setText(_translate("Form", "ML Predictions"))
        self.label_9.setText(_translate("Form", "Graphics Overlay Settings"))
        self.crop_reset_active.setText(_translate("Form", "Reset (Active Plot)"))
        self.show_bleach_range.setText(_translate("Form", "Bleach Ranges [B]"))
        self.show_measurement_range.setText(_translate("Form", "Measurement Ranges [M]"))
        self.show_gamma_range.setText(_translate("Form", "Gamma Ranges [G]"))
        self.show_crop_range.setText(_translate("Form", "Crop Ranges [C]"))
        self.crop_reset_all.setText(_translate("Form", "Reset (All Plots)"))
        self.measurment_reset_active.setText(_translate("Form", "Reset (Active Plot)"))
        self.bleach_reset_active.setText(_translate("Form", "Reset (Active Plot)"))
        self.gamma_reset_active.setText(_translate("Form", "Reset (Active Plot)"))
        self.measurment_reset_all.setText(_translate("Form", "Reset (All Plots)"))
        self.bleach_reset_all.setText(_translate("Form", "Reset (All Plots)"))
        self.gamma_reset_all.setText(_translate("Form", "Reset (All Plots)"))
        self.label_6.setText(_translate("Form", "Downsample Settings"))
        self.label_7.setText(_translate("Form", "Downsample"))
        self.plot_downsample.setItemText(0, _translate("Form", "1"))
        self.plot_downsample.setItemText(1, _translate("Form", "2"))
        self.plot_downsample.setItemText(2, _translate("Form", "3"))
        self.plot_downsample.setItemText(3, _translate("Form", "4"))
        self.plot_downsample.setItemText(4, _translate("Form", "5"))
        self.plot_downsample.setItemText(5, _translate("Form", "10"))
        self.label_4.setText(_translate("Form", "Crop Settings"))
        self.crop_plots.setText(_translate("Form", "Crop Plots"))
        self.label_3.setText(_translate("Form", "Group Settings"))
        self.label_17.setText(_translate("Form", "Group Label"))
        self.plot_user_group.setItemText(0, _translate("Form", "None"))
        self.plot_user_group.setItemText(1, _translate("Form", "0"))
        self.plot_user_group.setItemText(2, _translate("Form", "1"))
        self.plot_user_group.setItemText(3, _translate("Form", "2"))
        self.plot_user_group.setItemText(4, _translate("Form", "3"))
        self.plot_user_group.setItemText(5, _translate("Form", "4"))
        self.plot_user_group.setItemText(6, _translate("Form", "5"))
        self.plot_user_group.setItemText(7, _translate("Form", "6"))
        self.plot_user_group.setItemText(8, _translate("Form", "7"))
        self.plot_user_group.setItemText(9, _translate("Form", "8"))
        self.plot_user_group.setItemText(10, _translate("Form", "9"))
        self.label_2.setText(_translate("Form", "Measurement Label"))
        self.plot_measurement_label.setItemText(0, _translate("Form", "Label1"))
        self.plot_measurement_label.setItemText(1, _translate("Form", "Label2"))
        self.plot_measurement_label.setItemText(2, _translate("Form", "Label3"))
