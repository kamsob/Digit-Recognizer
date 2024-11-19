#include "Digit_recognizer_project.h"

import CNN;
import Layer;
import ConvolutionalLayer;
import FullyConnectedLayer;
import SigmoidFunction;
import ReshapeLayer;
import trainingData;
import Matrix;
import <iostream>;
import <vector>;
import <string>;

Digit_recognizer_project::Digit_recognizer_project(QWidget* parent) :
    QWidget(parent),
    ui(new Ui::Digit_recognizer_projectClass)
{
    ui->setupUi(this);

    scene = new paintScene();
    scene->setBackgroundBrush(Qt::black);
    ui->graphicsView->setScene(scene);
    timer = new QTimer();
    connect(timer, &QTimer::timeout, this, &Digit_recognizer_project::slotTimer);
    timer->start(100);

    save_directory = std::filesystem::current_path() / "WeightsAndBiases";

    network = Network(save_directory);

    network.addLayer(new ConvolutionalLayer(1, 28, 5, 2, 1)); // Output: 2 x 24
    network.addLayer(new ConvolutionalLayer(2, 24, 3, 8, 1)); // Output: 8 x 20
    network.addLayer(new Reshape(1, 1, 8 * 22 * 22));
    network.addLayer(new Dense(8 * 22 * 22, 40));
    network.addLayer(new Sigmoid());
    network.addLayer(new Dense(40, 10));
    network.addLayer(new Sigmoid());

    train_inputs.resize(33600, d2Matrix(28, std::vector<double>(28)));
    true_labels.resize(33600);
    for (int i = 0; i < train_inputs.size(); ++i)
    {
        image.uploadImage();
        true_labels[i] = image.getDigit();
        std::vector<double> pixels = image.getImage();
        for (int j = 0; j < 28; ++j)
            train_inputs[i][j] = std::vector<double>(pixels.begin() + j * 28, pixels.begin() + (j + 1) * 28);
    }

    test_inputs.resize(8400, d2Matrix(28, std::vector<double>(28)));
    test_true_labels.resize(8400);

    for (int i = 0; i < test_inputs.size(); ++i)
    {
        image.uploadImage();
        test_true_labels[i] = image.getDigit();
        std::vector<double> pixels = image.getImage();
        for (int j = 0; j < 28; ++j)
            test_inputs[i][j] = std::vector<double>(pixels.begin() + j * 28, pixels.begin() + (j + 1) * 28);
    }
}

Digit_recognizer_project::~Digit_recognizer_project()
{
    delete ui;
}

void Digit_recognizer_project::slotTimer()
{
    timer->stop();
    scene->setSceneRect(0, 0, ui->graphicsView->width() - 20, ui->graphicsView->height() - 20);
}

void Digit_recognizer_project::resizeEvent(QResizeEvent* event)
{
    timer->start(100);
    QWidget::resizeEvent(event);
}

void Digit_recognizer_project::on_submitButton_clicked()
{
    QImage originalImage(scene->sceneRect().size().toSize(), QImage::Format_ARGB32);
    QPainter painter(&originalImage);
    scene->render(&painter);
    painter.end();

    QImage scaledImage = originalImage.scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    d2Matrix input(28, std::vector<double>(28));

    for (int y = 0; y < scaledImage.height(); ++y)
        for (int x = 0; x < scaledImage.width(); ++x) 
        {
            QRgb pixel = scaledImage.pixel(x, y);
            input[y][x] = static_cast<double>(qGray(pixel)) / 255;
        }

    ui->result->setNum(network.prediction(input));
}

void Digit_recognizer_project::on_redrawButton_clicked()
{
    scene->clear();
}

void Digit_recognizer_project::on_LoadButton_clicked() 
{
    network.loadWeightsAndBiases();
}

void Digit_recognizer_project::on_Train_clicked() 
{
    int epochs = 1;
    double learning_rate = 0.01;
    network.train(train_inputs, true_labels, epochs, learning_rate);
}

void Digit_recognizer_project::on_Save_clicked()
{
    network.saveWeightsAndBiases();
}

void Digit_recognizer_project::on_Test_clicked()
{
    double accuracy_value = static_cast<double>(network.testOnDataset(test_inputs, test_true_labels)) / static_cast<double>(test_inputs.size()) * 100;
    QString accuracy = QString::number(accuracy_value);
    QString text = QString("accuracy: ") + accuracy + QString("%");
    ui->accuracy->setText(text);
}
