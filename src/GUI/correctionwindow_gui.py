# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\turnerp\PycharmProjects\TraceAnalyser\src\GUI\correctionwindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(443, 468)
        Form.setMaximumSize(QtCore.QSize(16777215, 800))
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 450))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_12 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_2.addWidget(self.label_12)
        self.label_13 = QtWidgets.QLabel(self.tab)
        self.label_13.setWordWrap(True)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_2.addWidget(self.label_13)
        self.formLayout_6 = QtWidgets.QFormLayout()
        self.formLayout_6.setObjectName("formLayout_6")
        self.bleach_sensitivity = QtWidgets.QLabel(self.tab)
        self.bleach_sensitivity.setObjectName("bleach_sensitivity")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.bleach_sensitivity)
        self.global_gamma = QtWidgets.QDoubleSpinBox(self.tab)
        self.global_gamma.setDecimals(2)
        self.global_gamma.setMaximum(10000.0)
        self.global_gamma.setSingleStep(1.0)
        self.global_gamma.setProperty("value", 1.0)
        self.global_gamma.setObjectName("global_gamma")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.global_gamma)
        self.verticalLayout_2.addLayout(self.formLayout_6)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem)
        self.label_5 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setObjectName("label_4")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.global_donor_leakage = QtWidgets.QDoubleSpinBox(self.tab)
        self.global_donor_leakage.setDecimals(2)
        self.global_donor_leakage.setMaximum(10000.0)
        self.global_donor_leakage.setSingleStep(0.1)
        self.global_donor_leakage.setProperty("value", 0.0)
        self.global_donor_leakage.setObjectName("global_donor_leakage")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.global_donor_leakage)
        self.verticalLayout_2.addLayout(self.formLayout_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem1)
        self.label_10 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.verticalLayout_2.addWidget(self.label_10)
        self.label_11 = QtWidgets.QLabel(self.tab)
        self.label_11.setWordWrap(True)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_2.addWidget(self.label_11)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_7 = QtWidgets.QLabel(self.tab)
        self.label_7.setObjectName("label_7")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.global_direct_excitation = QtWidgets.QDoubleSpinBox(self.tab)
        self.global_direct_excitation.setDecimals(2)
        self.global_direct_excitation.setMaximum(10000.0)
        self.global_direct_excitation.setSingleStep(0.1)
        self.global_direct_excitation.setProperty("value", 0.0)
        self.global_direct_excitation.setObjectName("global_direct_excitation")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.global_direct_excitation)
        self.verticalLayout_2.addLayout(self.formLayout_5)
        spacerItem2 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem2)
        self.global_correction_active = QtWidgets.QPushButton(self.tab)
        self.global_correction_active.setObjectName("global_correction_active")
        self.verticalLayout_2.addWidget(self.global_correction_active)
        self.global_correction_all = QtWidgets.QPushButton(self.tab)
        self.global_correction_all.setObjectName("global_correction_all")
        self.verticalLayout_2.addWidget(self.global_correction_all)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.correction_dataset = QtWidgets.QComboBox(self.tab_2)
        self.correction_dataset.setObjectName("correction_dataset")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.correction_dataset)
        self.label_17 = QtWidgets.QLabel(self.tab_2)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.correction_user_group = QtWidgets.QComboBox(self.tab_2)
        self.correction_user_group.setObjectName("correction_user_group")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.correction_user_group.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.correction_user_group)
        self.verticalLayout_3.addLayout(self.formLayout)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem3)
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.bleaching_spacer = QtWidgets.QSpinBox(self.tab_2)
        self.bleaching_spacer.setProperty("value", 5)
        self.bleaching_spacer.setObjectName("bleaching_spacer")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.bleaching_spacer)
        self.correction_min_frames = QtWidgets.QSpinBox(self.tab_2)
        self.correction_min_frames.setProperty("value", 5)
        self.correction_min_frames.setObjectName("correction_min_frames")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.correction_min_frames)
        self.bleach_sensitivity_2 = QtWidgets.QLabel(self.tab_2)
        self.bleach_sensitivity_2.setObjectName("bleach_sensitivity_2")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.bleach_sensitivity_2)
        self.gamma_default_frames = QtWidgets.QSpinBox(self.tab_2)
        self.gamma_default_frames.setProperty("value", 25)
        self.gamma_default_frames.setObjectName("gamma_default_frames")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.gamma_default_frames)
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.verticalLayout_3.addLayout(self.formLayout_3)
        self.use_global_factors = QtWidgets.QCheckBox(self.tab_2)
        self.use_global_factors.setObjectName("use_global_factors")
        self.verticalLayout_3.addWidget(self.use_global_factors)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem4)
        self.detect_correction_factors = QtWidgets.QPushButton(self.tab_2)
        self.detect_correction_factors.setObjectName("detect_correction_factors")
        self.verticalLayout_3.addWidget(self.detect_correction_factors)
        self.correction_progressbar = QtWidgets.QProgressBar(self.tab_2)
        self.correction_progressbar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.correction_progressbar.setProperty("value", 0)
        self.correction_progressbar.setObjectName("correction_progressbar")
        self.verticalLayout_3.addWidget(self.correction_progressbar)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_12.setText(_translate("Form", "Gamma Factor"))
        self.label_13.setText(_translate("Form", "The Gamma correction factor accounts for the relative difference in the number of photons measured of the acceptor (DA) and the donor (DD) for the same number of excited states."))
        self.bleach_sensitivity.setText(_translate("Form", "Gamma Factor:"))
        self.label_5.setText(_translate("Form", "Donor Leakage"))
        self.label_2.setText(_translate("Form", "The Donor Leakage correction factor accounts for the amount of leakage of the donor emission (DD) into the acceptor (DA) emission channel upon donor excitation, in the time interval after the acceptor has bleached and before the donor has bleached"))
        self.label_4.setText(_translate("Form", "Donor Leakage:"))
        self.label_10.setText(_translate("Form", "Direct Acceptor"))
        self.label_11.setText(_translate("Form", "The Direct Acceptor correction factor accounts for the direct excitation of the acceptor at the donor wavelength (DA), in the time interval after the donor has bleached and before the acceptor has bleached."))
        self.label_7.setText(_translate("Form", "Direct Acceptor Excitation: "))
        self.global_correction_active.setText(_translate("Form", "Apply Global Correction Factors (Active Trace)"))
        self.global_correction_all.setText(_translate("Form", "Apply Global Correction Factors (All Traces)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Global Correction Factors"))
        self.label.setText(_translate("Form", "Dataset Selection"))
        self.label_3.setText(_translate("Form", "Dataset(s)"))
        self.label_17.setText(_translate("Form", "Group Label"))
        self.correction_user_group.setItemText(0, _translate("Form", "None"))
        self.correction_user_group.setItemText(1, _translate("Form", "0"))
        self.correction_user_group.setItemText(2, _translate("Form", "1"))
        self.correction_user_group.setItemText(3, _translate("Form", "2"))
        self.correction_user_group.setItemText(4, _translate("Form", "3"))
        self.correction_user_group.setItemText(5, _translate("Form", "4"))
        self.correction_user_group.setItemText(6, _translate("Form", "5"))
        self.correction_user_group.setItemText(7, _translate("Form", "6"))
        self.correction_user_group.setItemText(8, _translate("Form", "7"))
        self.correction_user_group.setItemText(9, _translate("Form", "8"))
        self.correction_user_group.setItemText(10, _translate("Form", "9"))
        self.label_6.setText(_translate("Form", "Detect Molecule Dependent Correction Factors"))
        self.label_8.setText(_translate("Form", "#Frame Buffer After Bleaching"))
        self.bleach_sensitivity_2.setText(_translate("Form", "Gamma Factor Default #Frames"))
        self.label_9.setText(_translate("Form", "Correction Factor Min #Frames"))
        self.use_global_factors.setText(_translate("Form", "Use Global Correction Factors (If molecule-dependent factors unavailable)"))
        self.detect_correction_factors.setText(_translate("Form", "Detect Correction Factors"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Form", "Detect Correction Factors"))