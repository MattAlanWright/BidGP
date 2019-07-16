#include <iostream>
#include <fstream>

#include "Environment.hpp"
#include "iris.hpp"
#include "shuttle.hpp"
#include "thyroid.hpp"
#include "tictactoe.hpp"

void irisStandardFitnessExperiment() {
    IrisDataset * iris = new IrisDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        iris->num_classes,
        iris->num_features,
        100,
        70,
        100,
        iris->train_X,
        iris->train_y,
        "iris_standard",
        false);

    env.train(500);

    float training_acc    = env.accuracy(iris->train_X, iris->train_y);
    float training_recall = env.detectionRate(iris->train_X, iris->train_y);
    float test_accuracy   = env.accuracy(iris->test_X, iris->test_y);
    float test_recall     = env.detectionRate(iris->test_X, iris->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("iris_standard_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete iris;
}


void irisFitnessSharingExperiment() {
    IrisDataset * iris = new IrisDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        iris->num_classes,
        iris->num_features,
        100,
        70,
        100,
        iris->train_X,
        iris->train_y,
        "iris_sharing",
        true);

    env.train(500);

    float training_acc    = env.accuracy(iris->train_X, iris->train_y);
    float training_recall = env.detectionRate(iris->train_X, iris->train_y);
    float test_accuracy   = env.accuracy(iris->test_X, iris->test_y);
    float test_recall     = env.detectionRate(iris->test_X, iris->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("iris_sharing_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete iris;
}


void shuttleStandardFitnessExperiment() {
    ShuttleStatlogDataset * shuttle = new ShuttleStatlogDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        shuttle->num_classes,
        shuttle->num_features,
        100,
        70,
        100,
        shuttle->train_X,
        shuttle->train_y,
        "shuttle_standard",
        false);

    env.train(10000);

    float training_acc    = env.accuracy(shuttle->train_X, shuttle->train_y);
    float training_recall = env.detectionRate(shuttle->train_X, shuttle->train_y);
    float test_accuracy   = env.accuracy(shuttle->test_X, shuttle->test_y);
    float test_recall     = env.detectionRate(shuttle->test_X, shuttle->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("shuttle_standard_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete shuttle;
}


void shuttleFitnessSharingExperiment() {
    ShuttleStatlogDataset * shuttle = new ShuttleStatlogDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        shuttle->num_classes,
        shuttle->num_features,
        100,
        70,
        100,
        shuttle->train_X,
        shuttle->train_y,
        "shuttle_sharing",
        true);

    env.train(10000);

    float training_acc    = env.accuracy(shuttle->train_X, shuttle->train_y);
    float training_recall = env.detectionRate(shuttle->train_X, shuttle->train_y);
    float test_accuracy   = env.accuracy(shuttle->test_X, shuttle->test_y);
    float test_recall     = env.detectionRate(shuttle->test_X, shuttle->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("shuttle_sharing_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete shuttle;
}


void thyroidStandardFitnessExperiment() {
    ThyroidDataset * thyroid = new ThyroidDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        thyroid->num_classes,
        thyroid->num_features,
        100,
        70,
        100,
        thyroid->train_X,
        thyroid->train_y,
        "thyroid_standard",
        false);

    env.train(10000);

    float training_acc    = env.accuracy(thyroid->train_X, thyroid->train_y);
    float training_recall = env.detectionRate(thyroid->train_X, thyroid->train_y);
    float test_accuracy   = env.accuracy(thyroid->test_X, thyroid->test_y);
    float test_recall     = env.detectionRate(thyroid->test_X, thyroid->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("thyroid_standard_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete thyroid;
}


void thyroidFitnessSharingExperiment() {
    ThyroidDataset * thyroid = new ThyroidDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        thyroid->num_classes,
        thyroid->num_features,
        100,
        70,
        100,
        thyroid->train_X,
        thyroid->train_y,
        "thyroid_sharing",
        true);

    env.train(10000);

    float training_acc    = env.accuracy(thyroid->train_X, thyroid->train_y);
    float training_recall = env.detectionRate(thyroid->train_X, thyroid->train_y);
    float test_accuracy   = env.accuracy(thyroid->test_X, thyroid->test_y);
    float test_recall     = env.detectionRate(thyroid->test_X, thyroid->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("thyroid_sharing_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete thyroid;
}


void tttStandardFitnessExperiment() {
    TicTacToeDataset * ttt = new TicTacToeDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        ttt->num_classes,
        ttt->num_features,
        100,
        70,
        100,
        ttt->train_X,
        ttt->train_y,
        "ttt_standard",
        false);

    env.train(10000);

    float training_acc    = env.accuracy(ttt->train_X, ttt->train_y);
    float training_recall = env.detectionRate(ttt->train_X, ttt->train_y);
    float test_accuracy   = env.accuracy(ttt->test_X, ttt->test_y);
    float test_recall     = env.detectionRate(ttt->test_X, ttt->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("ttt_standard_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete ttt;
}


void tttFitnessSharingExperiment() {
    TicTacToeDataset * ttt = new TicTacToeDataset();

    ClassificationEnvironment env = ClassificationEnvironment(
        ttt->num_classes,
        ttt->num_features,
        100,
        70,
        100,
        ttt->train_X,
        ttt->train_y,
        "ttt_sharing",
        true);

    env.train(10000);

    float training_acc    = env.accuracy(ttt->train_X, ttt->train_y);
    float training_recall = env.detectionRate(ttt->train_X, ttt->train_y);
    float test_accuracy   = env.accuracy(ttt->test_X, ttt->test_y);
    float test_recall     = env.detectionRate(ttt->test_X, ttt->test_y);

    std::cout << "Training Accuracy: " << training_acc    << std::endl;
    std::cout << "Training Recall: "   << training_recall << std::endl;
    std::cout << "Test Accuracy: "     << test_accuracy   << std::endl;
    std::cout << "Test Recall: "       << test_recall     << std::endl;

    std::ofstream results_file;
    results_file.open("ttt_sharing_results.txt");
    results_file << "Training Accuracy: " << training_acc    << std::endl;
    results_file << "Training Recall: "   << training_recall << std::endl;
    results_file << "Test Accuracy: "     << test_accuracy   << std::endl;
    results_file << "Test Recall: "       << test_recall     << std::endl;
    results_file.close();

    delete ttt;
}


int main() {

    irisStandardFitnessExperiment();
    irisFitnessSharingExperiment();
    shuttleStandardFitnessExperiment();
    shuttleFitnessSharingExperiment();
    thyroidStandardFitnessExperiment();
    thyroidFitnessSharingExperiment();
    tttStandardFitnessExperiment();
    tttFitnessSharingExperiment();

    return 0;
}
