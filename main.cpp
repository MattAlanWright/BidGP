#include <iostream>

#include "Environment.hpp"
#include "iris.hpp"
#include "shuttle.hpp"

void irisExperiment() {
    IrisDataset * iris = new IrisDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        iris->num_classes,
        iris->num_features,
        100,
        70,
        iris->num_training_samples,
        iris->train_X,
        iris->train_y);

    env.train(500);

    std::cout << "Training Accuracy: " << env.accuracy(iris->train_X, iris->train_y) << std::endl;
    std::cout << "Training Recall: " << env.detectionRate(iris->train_X, iris->train_y) << std::endl;
    std::cout << "Test Accuracy: " << env.accuracy(iris->test_X, iris->test_y) << std::endl;
    std::cout << "Test Recall: " << env.detectionRate(iris->test_X, iris->test_y) << std::endl;
}

void shuttleExperiment() {
    ShuttleStatlogDataset * shuttle = new ShuttleStatlogDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        shuttle->num_classes,
        shuttle->num_features,
        100,
        70,
        100,
        shuttle->train_X,
        shuttle->train_y);

    env.train(500);

    std::cout << "Training Accuracy: " << env.accuracy(shuttle->train_X, shuttle->train_y) << std::endl;
    std::cout << "Training Recall: " << env.detectionRate(shuttle->train_X, shuttle->train_y) << std::endl;
    std::cout << "Test Accuracy: " << env.accuracy(shuttle->test_X, shuttle->test_y) << std::endl;
    std::cout << "Test Recall: " << env.detectionRate(shuttle->test_X, shuttle->test_y) << std::endl;
}

int main() {

    shuttleExperiment();

    return 0;
}
