#ifndef _ENVIRONMENT_HPP
#define _ENVIRONMENT_HPP

#include "Point.hpp"
#include "Learner.hpp"
#include "Dataset.hpp"

#include <vector>

struct Outcome {
    float fitness;
    bool  match;
};

class ClassificationEnvironment {

public:
    ClassificationEnvironment(int num_classes,
                              int num_features,
                              int l_size,
                              int l_gap,
                              int tau,
                              const std::vector<std::vector<float>> &X,
                              const std::vector<int>                &y,
                              std::string filename,
                              bool do_fitness_sharing);

    int num_classes;
    int num_features;
    int l_size;
    int l_gap;
    int tau;
    int p_size;
    bool do_fitness_sharing;
    std::string filename;

    BalancedDataset dataset;

    // Solution population
    std::vector<Learner> S;

    // Initialization
    std::vector<Learner> initializeLearners(int action);

    // Population generation
    void generateLearners(std::vector<Learner> &learner_pop);
    std::vector<Point> generatePoints();

    // Let the Hosts bid on the Points and store the results
    void calculateOutcomeMatrix(
        std::vector<Learner>                &learners,
        std::vector<Point>                  &points,
        std::vector< std::vector<Outcome> > &G);

    // Evaluation methods
    void evaluateHosts();
    void evaluatePoints();

    // Remove lowest performing individuals
    void removeHosts();
    void removePoints();

    // Fitness measurements
    float standardFitness(
        std::vector<Learner>              &learners,
        std::vector< std::vector<float> > &G);

    float paretoFitness(
        std::vector<Learner>              &learners,
        std::vector< std::vector<float> > &G);

    float fitnessSharing(
        std::vector<Learner>              &learners,
        std::vector< std::vector<float> > &G);

    // Core training alorithm
    void train(int num_generations);

    // Classify a Point using the entire Solution
    int classify(const std::vector<float> &features);

    // Find accuracy using discovered solution
    float accuracy(
        const std::vector<std::vector<float>> &X,
        const std::vector<int>                &y);

    float detectionRate(
        const std::vector<std::vector<float>> &X,
        const std::vector<int>                &y);
};

#endif //_ENVIRONMENT_HPP
