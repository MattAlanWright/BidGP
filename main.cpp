#include <iostream>

#include "Environment.hpp"
#include "iris.hpp"

int main() {

    IrisDataset * iris = new IrisDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        iris->num_classes,
        iris->num_features,
        100,
        70,
        iris->num_training_samples,
        iris->train_X,
        iris->train_y);

    env.train(200);

    std::cout << "Training Accuracy: " << env.accuracy(iris->train_X, iris->train_y) << std::endl;
    std::cout << "Test Accuracy: " << env.accuracy(iris->test_X, iris->test_y) << std::endl;

    return 0;
}
