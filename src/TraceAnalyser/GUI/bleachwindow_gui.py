# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\turnerp\PycharmProjects\TraceAnalyser\src\TraceAnalyser\GUI\bleachwindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(403, 478)
        Form.setMaximumSize(QtCore.QSize(16777215, 500))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.bleach_dataset = QtWidgets.QComboBox(Form)
        self.bleach_dataset.setObjectName("bleach_dataset")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.bleach_dataset)
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.bleach_user_group = QtWidgets.QComboBox(Form)
        self.bleach_user_group.setObjectName("bleach_user_group")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.bleach_user_group.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.bleach_user_group)
        self.verticalLayout.addLayout(self.formLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.bleach_acceptor_threshold = QtWidgets.QDoubleSpinBox(Form)
        self.bleach_acceptor_threshold.setDecimals(1)
        self.bleach_acceptor_threshold.setMaximum(10000.0)
        self.bleach_acceptor_threshold.setSingleStep(0.1)
        self.bleach_acceptor_threshold.setProperty("value", 0.2)
        self.bleach_acceptor_threshold.setObjectName("bleach_acceptor_threshold")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.bleach_acceptor_threshold)
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setObjectName("label_7")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.bleach_donor_threshold = QtWidgets.QDoubleSpinBox(Form)
        self.bleach_donor_threshold.setDecimals(1)
        self.bleach_donor_threshold.setMaximum(10000.0)
        self.bleach_donor_threshold.setSingleStep(0.1)
        self.bleach_donor_threshold.setProperty("value", 0.2)
        self.bleach_donor_threshold.setObjectName("bleach_donor_threshold")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.bleach_donor_threshold)
        self.bleach_sensitivity = QtWidgets.QLabel(Form)
        self.bleach_sensitivity.setObjectName("bleach_sensitivity")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.bleach_sensitivity)
        self.bleach_events_ignored = QtWidgets.QSpinBox(Form)
        self.bleach_events_ignored.setObjectName("bleach_events_ignored")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.bleach_events_ignored)
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.bleach_event_size = QtWidgets.QSpinBox(Form)
        self.bleach_event_size.setMinimum(1)
        self.bleach_event_size.setProperty("value", 5)
        self.bleach_event_size.setObjectName("bleach_event_size")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.bleach_event_size)
        self.verticalLayout.addLayout(self.formLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.detect_bleach = QtWidgets.QPushButton(Form)
        self.detect_bleach.setObjectName("detect_bleach")
        self.verticalLayout.addWidget(self.detect_bleach)
        self.bleach_progressbar = QtWidgets.QProgressBar(Form)
        self.bleach_progressbar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.bleach_progressbar.setProperty("value", 0)
        self.bleach_progressbar.setObjectName("bleach_progressbar")
        self.verticalLayout.addWidget(self.bleach_progressbar)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Dataset Selection"))
        self.label_3.setText(_translate("Form", "Dataset(s)"))
        self.label_17.setText(_translate("Form", "Group Label"))
        self.bleach_user_group.setItemText(0, _translate("Form", "None"))
        self.bleach_user_group.setItemText(1, _translate("Form", "0"))
        self.bleach_user_group.setItemText(2, _translate("Form", "1"))
        self.bleach_user_group.setItemText(3, _translate("Form", "2"))
        self.bleach_user_group.setItemText(4, _translate("Form", "3"))
        self.bleach_user_group.setItemText(5, _translate("Form", "4"))
        self.bleach_user_group.setItemText(6, _translate("Form", "5"))
        self.bleach_user_group.setItemText(7, _translate("Form", "6"))
        self.bleach_user_group.setItemText(8, _translate("Form", "7"))
        self.bleach_user_group.setItemText(9, _translate("Form", "8"))
        self.bleach_user_group.setItemText(10, _translate("Form", "9"))
        self.label_2.setText(_translate("Form", "Bleach Detection Settings"))
        self.label_4.setText(_translate("Form", "Acceptor Threshold"))
        self.label_7.setText(_translate("Form", "Donor Threshold"))
        self.bleach_sensitivity.setText(_translate("Form", "Events Ignored (#Events)"))
        self.label_5.setText(_translate("Form", "Mininum Event Size (#Frames)"))
        self.detect_bleach.setText(_translate("Form", "Detect Bleaching in Datset Selection"))