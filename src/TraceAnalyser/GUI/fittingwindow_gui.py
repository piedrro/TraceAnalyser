# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\turnerp\PycharmProjects\TraceAnalyser\src\TraceAnalyser\GUI\fittingwindow_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 581)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_11 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_4.addWidget(self.label_11)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setObjectName("label_8")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.hmm_fit_dataset = QtWidgets.QComboBox(self.tab_3)
        self.hmm_fit_dataset.setObjectName("hmm_fit_dataset")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.hmm_fit_dataset)
        self.label_10 = QtWidgets.QLabel(self.tab_3)
        self.label_10.setObjectName("label_10")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.hmm_fit_data = QtWidgets.QComboBox(self.tab_3)
        self.hmm_fit_data.setObjectName("hmm_fit_data")
        self.hmm_fit_data.addItem("")
        self.hmm_fit_data.addItem("")
        self.hmm_fit_data.addItem("")
        self.hmm_fit_data.addItem("")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.hmm_fit_data)
        self.label_19 = QtWidgets.QLabel(self.tab_3)
        self.label_19.setObjectName("label_19")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.hmm_user_group = QtWidgets.QComboBox(self.tab_3)
        self.hmm_user_group.setObjectName("hmm_user_group")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.hmm_user_group.addItem("")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.hmm_user_group)
        self.verticalLayout_4.addLayout(self.formLayout_3)
        self.hmm_crop_plots = QtWidgets.QCheckBox(self.tab_3)
        self.hmm_crop_plots.setObjectName("hmm_crop_plots")
        self.verticalLayout_4.addWidget(self.hmm_crop_plots)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_4.addItem(spacerItem)
        self.label_12 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_4.addWidget(self.label_12)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_14 = QtWidgets.QLabel(self.tab_3)
        self.label_14.setObjectName("label_14")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.hmm_package = QtWidgets.QComboBox(self.tab_3)
        self.hmm_package.setObjectName("hmm_package")
        self.hmm_package.addItem("")
        self.hmm_package.addItem("")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.hmm_package)
        self.label_13 = QtWidgets.QLabel(self.tab_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.hmm_n_states = QtWidgets.QComboBox(self.tab_3)
        self.hmm_n_states.setObjectName("hmm_n_states")
        self.hmm_n_states.addItem("")
        self.hmm_n_states.addItem("")
        self.hmm_n_states.addItem("")
        self.hmm_n_states.addItem("")
        self.hmm_n_states.addItem("")
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.hmm_n_states)
        self.label_15 = QtWidgets.QLabel(self.tab_3)
        self.label_15.setObjectName("label_15")
        self.formLayout_4.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.hmm_n_iterations = QtWidgets.QLineEdit(self.tab_3)
        self.hmm_n_iterations.setObjectName("hmm_n_iterations")
        self.formLayout_4.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.hmm_n_iterations)
        self.label_20 = QtWidgets.QLabel(self.tab_3)
        self.label_20.setObjectName("label_20")
        self.formLayout_4.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.hmm_n_init = QtWidgets.QLineEdit(self.tab_3)
        self.hmm_n_init.setObjectName("hmm_n_init")
        self.formLayout_4.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.hmm_n_init)
        self.label_21 = QtWidgets.QLabel(self.tab_3)
        self.label_21.setObjectName("label_21")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.hmm_fit_algo = QtWidgets.QComboBox(self.tab_3)
        self.hmm_fit_algo.setObjectName("hmm_fit_algo")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.hmm_fit_algo)
        self.label_24 = QtWidgets.QLabel(self.tab_3)
        self.label_24.setObjectName("label_24")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.hmm_mode = QtWidgets.QComboBox(self.tab_3)
        self.hmm_mode.setObjectName("hmm_mode")
        self.hmm_mode.addItem("")
        self.hmm_mode.addItem("")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.hmm_mode)
        self.verticalLayout_4.addLayout(self.formLayout_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_4.addItem(spacerItem1)
        self.label_23 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_4.addWidget(self.label_23)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_22 = QtWidgets.QLabel(self.tab_3)
        self.label_22.setObjectName("label_22")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.hmm_min_length = QtWidgets.QLineEdit(self.tab_3)
        self.hmm_min_length.setObjectName("hmm_min_length")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.hmm_min_length)
        self.label_16 = QtWidgets.QLabel(self.tab_3)
        self.label_16.setObjectName("label_16")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.hmm_max_transitions = QtWidgets.QLineEdit(self.tab_3)
        self.hmm_max_transitions.setObjectName("hmm_max_transitions")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.hmm_max_transitions)
        self.verticalLayout_4.addLayout(self.formLayout_5)
        spacerItem2 = QtWidgets.QSpacerItem(20, 26, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.hmm_detect_states = QtWidgets.QPushButton(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hmm_detect_states.sizePolicy().hasHeightForWidth())
        self.hmm_detect_states.setSizePolicy(sizePolicy)
        self.hmm_detect_states.setObjectName("hmm_detect_states")
        self.verticalLayout_4.addWidget(self.hmm_detect_states)
        self.hmm_progressbar = QtWidgets.QProgressBar(self.tab_3)
        self.hmm_progressbar.setMaximumSize(QtCore.QSize(16777215, 10))
        self.hmm_progressbar.setProperty("value", 0)
        self.hmm_progressbar.setObjectName("hmm_progressbar")
        self.verticalLayout_4.addWidget(self.hmm_progressbar)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.ebfret_connect_matlab = QtWidgets.QPushButton(self.tab)
        self.ebfret_connect_matlab.setObjectName("ebfret_connect_matlab")
        self.verticalLayout_2.addWidget(self.ebfret_connect_matlab)
        self.ebfret_connect_matlab_progress = QtWidgets.QProgressBar(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ebfret_connect_matlab_progress.sizePolicy().hasHeightForWidth())
        self.ebfret_connect_matlab_progress.setSizePolicy(sizePolicy)
        self.ebfret_connect_matlab_progress.setMaximumSize(QtCore.QSize(16777215, 10))
        self.ebfret_connect_matlab_progress.setProperty("value", 0)
        self.ebfret_connect_matlab_progress.setObjectName("ebfret_connect_matlab_progress")
        self.verticalLayout_2.addWidget(self.ebfret_connect_matlab_progress)
        spacerItem3 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem3)
        self.label_2 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.ebfret_fit_dataset = QtWidgets.QComboBox(self.tab)
        self.ebfret_fit_dataset.setObjectName("ebfret_fit_dataset")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ebfret_fit_dataset)
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.ebfret_fit_data = QtWidgets.QComboBox(self.tab)
        self.ebfret_fit_data.setObjectName("ebfret_fit_data")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.ebfret_fit_data.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.ebfret_fit_data)
        self.label_17 = QtWidgets.QLabel(self.tab)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.ebfret_user_group = QtWidgets.QComboBox(self.tab)
        self.ebfret_user_group.setObjectName("ebfret_user_group")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.ebfret_user_group.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ebfret_user_group)
        self.verticalLayout_2.addLayout(self.formLayout)
        self.gridLayout_17 = QtWidgets.QGridLayout()
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.ebfret_max_states = QtWidgets.QComboBox(self.tab)
        self.ebfret_max_states.setObjectName("ebfret_max_states")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.ebfret_max_states.addItem("")
        self.gridLayout_17.addWidget(self.ebfret_max_states, 0, 3, 1, 1)
        self.ebfret_min_states = QtWidgets.QComboBox(self.tab)
        self.ebfret_min_states.setObjectName("ebfret_min_states")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.ebfret_min_states.addItem("")
        self.gridLayout_17.addWidget(self.ebfret_min_states, 0, 1, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.tab)
        self.label_38.setObjectName("label_38")
        self.gridLayout_17.addWidget(self.label_38, 0, 0, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.tab)
        self.label_39.setObjectName("label_39")
        self.gridLayout_17.addWidget(self.label_39, 0, 2, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_17)
        self.ebfret_crop_plots = QtWidgets.QCheckBox(self.tab)
        self.ebfret_crop_plots.setObjectName("ebfret_crop_plots")
        self.verticalLayout_2.addWidget(self.ebfret_crop_plots)
        self.ebfret_run_analysis = QtWidgets.QPushButton(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ebfret_run_analysis.sizePolicy().hasHeightForWidth())
        self.ebfret_run_analysis.setSizePolicy(sizePolicy)
        self.ebfret_run_analysis.setObjectName("ebfret_run_analysis")
        self.verticalLayout_2.addWidget(self.ebfret_run_analysis)
        spacerItem4 = QtWidgets.QSpacerItem(20, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.label_4 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.formLayout_26 = QtWidgets.QFormLayout()
        self.formLayout_26.setObjectName("formLayout_26")
        self.label_66 = QtWidgets.QLabel(self.tab)
        self.label_66.setObjectName("label_66")
        self.formLayout_26.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_66)
        self.ebfret_visualisation_state = QtWidgets.QComboBox(self.tab)
        self.ebfret_visualisation_state.setObjectName("ebfret_visualisation_state")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.ebfret_visualisation_state.addItem("")
        self.formLayout_26.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.ebfret_visualisation_state)
        self.verticalLayout_2.addLayout(self.formLayout_26)
        self.tabWidget.addTab(self.tab, "")
        self.verticalLayout.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.ebfret_max_states.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_11.setText(_translate("Form", "Fit Data"))
        self.label_8.setText(_translate("Form", "Fit Dataset"))
        self.label_10.setText(_translate("Form", "Fit Data"))
        self.hmm_fit_data.setItemText(0, _translate("Form", "Donor"))
        self.hmm_fit_data.setItemText(1, _translate("Form", "Acceptor"))
        self.hmm_fit_data.setItemText(2, _translate("Form", "FRET Efficiency"))
        self.hmm_fit_data.setItemText(3, _translate("Form", "ALEX Efficiency"))
        self.label_19.setText(_translate("Form", "Group Label"))
        self.hmm_user_group.setItemText(0, _translate("Form", "None"))
        self.hmm_user_group.setItemText(1, _translate("Form", "0"))
        self.hmm_user_group.setItemText(2, _translate("Form", "1"))
        self.hmm_user_group.setItemText(3, _translate("Form", "2"))
        self.hmm_user_group.setItemText(4, _translate("Form", "3"))
        self.hmm_user_group.setItemText(5, _translate("Form", "4"))
        self.hmm_user_group.setItemText(6, _translate("Form", "5"))
        self.hmm_user_group.setItemText(7, _translate("Form", "6"))
        self.hmm_user_group.setItemText(8, _translate("Form", "7"))
        self.hmm_user_group.setItemText(9, _translate("Form", "8"))
        self.hmm_user_group.setItemText(10, _translate("Form", "9"))
        self.hmm_crop_plots.setText(_translate("Form", "Crop Plots"))
        self.label_12.setText(_translate("Form", "HMM Fit Settings"))
        self.label_14.setText(_translate("Form", "HMM Implementation"))
        self.hmm_package.setItemText(0, _translate("Form", "HMM Learn"))
        self.hmm_package.setItemText(1, _translate("Form", "Pomegranate"))
        self.label_13.setText(_translate("Form", "N States"))
        self.hmm_n_states.setItemText(0, _translate("Form", "2"))
        self.hmm_n_states.setItemText(1, _translate("Form", "3"))
        self.hmm_n_states.setItemText(2, _translate("Form", "4"))
        self.hmm_n_states.setItemText(3, _translate("Form", "5"))
        self.hmm_n_states.setItemText(4, _translate("Form", "Auto"))
        self.label_15.setText(_translate("Form", "N Iterations"))
        self.hmm_n_iterations.setText(_translate("Form", "1000"))
        self.label_20.setText(_translate("Form", "N Restarts"))
        self.hmm_n_init.setText(_translate("Form", "10"))
        self.label_21.setText(_translate("Form", "HMM Fit Algorithm"))
        self.label_24.setText(_translate("Form", "HMM Mode"))
        self.hmm_mode.setItemText(0, _translate("Form", "Fit ALL traces TOGETHER"))
        self.hmm_mode.setItemText(1, _translate("Form", "Fit EACH trace INDEPENDENTLY"))
        self.label_23.setText(_translate("Form", "HMM Post Processing"))
        self.label_22.setText(_translate("Form", "Min State Length"))
        self.hmm_min_length.setText(_translate("Form", "2"))
        self.label_16.setText(_translate("Form", "Max State Transitions"))
        self.hmm_max_transitions.setText(_translate("Form", "100"))
        self.hmm_detect_states.setText(_translate("Form", "Detect States"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Form", "HMM"))
        self.label.setText(_translate("Form", "Connect MATLAB/ebFRET"))
        self.ebfret_connect_matlab.setText(_translate("Form", "Open MATLAB/ebFRET"))
        self.label_2.setText(_translate("Form", "ebFRET Analysis"))
        self.label_3.setText(_translate("Form", "Fit Dataset"))
        self.label_5.setText(_translate("Form", "Fit Data"))
        self.ebfret_fit_data.setItemText(0, _translate("Form", "Donor"))
        self.ebfret_fit_data.setItemText(1, _translate("Form", "Acceptor"))
        self.ebfret_fit_data.setItemText(2, _translate("Form", "FRET Efficiency"))
        self.ebfret_fit_data.setItemText(3, _translate("Form", "ALEX Efficiency"))
        self.label_17.setText(_translate("Form", "Group Label"))
        self.ebfret_user_group.setItemText(0, _translate("Form", "None"))
        self.ebfret_user_group.setItemText(1, _translate("Form", "0"))
        self.ebfret_user_group.setItemText(2, _translate("Form", "1"))
        self.ebfret_user_group.setItemText(3, _translate("Form", "2"))
        self.ebfret_user_group.setItemText(4, _translate("Form", "3"))
        self.ebfret_user_group.setItemText(5, _translate("Form", "4"))
        self.ebfret_user_group.setItemText(6, _translate("Form", "5"))
        self.ebfret_user_group.setItemText(7, _translate("Form", "6"))
        self.ebfret_user_group.setItemText(8, _translate("Form", "7"))
        self.ebfret_user_group.setItemText(9, _translate("Form", "8"))
        self.ebfret_user_group.setItemText(10, _translate("Form", "9"))
        self.ebfret_max_states.setItemText(0, _translate("Form", "2"))
        self.ebfret_max_states.setItemText(1, _translate("Form", "3"))
        self.ebfret_max_states.setItemText(2, _translate("Form", "4"))
        self.ebfret_max_states.setItemText(3, _translate("Form", "5"))
        self.ebfret_max_states.setItemText(4, _translate("Form", "6"))
        self.ebfret_min_states.setItemText(0, _translate("Form", "2"))
        self.ebfret_min_states.setItemText(1, _translate("Form", "3"))
        self.ebfret_min_states.setItemText(2, _translate("Form", "4"))
        self.ebfret_min_states.setItemText(3, _translate("Form", "5"))
        self.ebfret_min_states.setItemText(4, _translate("Form", "6"))
        self.label_38.setText(_translate("Form", "Min States:"))
        self.label_39.setText(_translate("Form", "Max States:"))
        self.ebfret_crop_plots.setText(_translate("Form", "Crop Plots"))
        self.ebfret_run_analysis.setText(_translate("Form", "Run ebFRET analysis"))
        self.label_4.setText(_translate("Form", "ebFRET Visualisation"))
        self.label_66.setText(_translate("Form", "ebFRET fitted States:"))
        self.ebfret_visualisation_state.setItemText(0, _translate("Form", "2"))
        self.ebfret_visualisation_state.setItemText(1, _translate("Form", "3"))
        self.ebfret_visualisation_state.setItemText(2, _translate("Form", "4"))
        self.ebfret_visualisation_state.setItemText(3, _translate("Form", "5"))
        self.ebfret_visualisation_state.setItemText(4, _translate("Form", "6"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "ebFRET"))
