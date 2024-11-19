#pragma once

#include <QtWidgets/QWidget>
#include "ui_Digit_recognizer_project.h"

#include <QWidget>
#include <QTimer>
#include <QResizeEvent>
#include <QStandardPaths>
#include <QMessageBox>
#include <QImageWriter>
#include <QDir>
#include <QFileDialog>
#include <fstream>
#include <iomanip>

#include "paintscene.h"
import <vector>;
import CNN;
import Layer;
import ConvolutionalLayer;
import FullyConnectedLayer;
import SigmoidFunction;
import ReshapeLayer;
import trainingData;

class Digit_recognizer_project : public QWidget
{
    Q_OBJECT

public:
    Digit_recognizer_project(QWidget *parent = nullptr);
    ~Digit_recognizer_project();

private:
    Ui::Digit_recognizer_projectClass* ui;
    QTimer* timer;
    paintScene* scene;

    Network network;
    CurrImage image;
    d3Matrix train_inputs;
    std::vector<int> true_labels;
    d3Matrix test_inputs;
    std::vector<int> test_true_labels;
    std::filesystem::path save_directory;

private:
    void resizeEvent(QResizeEvent* event);

private slots:
    void slotTimer();
    void on_submitButton_clicked();
    void on_redrawButton_clicked();
    void on_LoadButton_clicked();
    void on_Train_clicked();
    void on_Save_clicked();
    void on_Test_clicked();
};
