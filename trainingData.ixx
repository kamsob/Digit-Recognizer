#include <fstream>
#include <sstream>

export module trainingData;

import <string>;
import <iostream>;
import <vector>;


export class CurrImage 
{
private:
    int digit, counter;
    std::vector<double>pixels;
    std::ifstream file;
	
public:
    CurrImage();
    ~CurrImage();
    bool uploadImage();
    int getDigit() { return digit; };
    int getCounter() { return counter; }
    std::vector<double>getImage() { return pixels; }
};

CurrImage::CurrImage()
    :counter(0), pixels(784)
{
    file = std::ifstream("train.csv");
    if (!file.is_open())
        std::cout << "nie mozna otworzyc pliku train.csv" << std::endl;
    std::string trash;//indexes of pixels
    std::getline(file, trash);
}

CurrImage::~CurrImage()
{
    file.close();
    std::cout<<"plik train.csv zostal zamkniety" << std::endl;
}

bool CurrImage::uploadImage() 
{
    counter++;
    std::string line;
    if (!std::getline(file, line)) {
        CurrImage::~CurrImage();
        return false;
    }
    std::stringstream ss(line);
    std::string currElement;

    std::getline(ss, currElement, ',');
    digit = std::stoi(currElement);

    int idx = 0;
    while (std::getline(ss, currElement, ',')) {
        pixels[idx] = std::stod(currElement)/255;
        idx++;
    }
    return true;
}

