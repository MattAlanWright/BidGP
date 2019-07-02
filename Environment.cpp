#include "Environment.hpp"

#include <cmath>

// Effolkronium random library
#include "random.hpp"
using Random = effolkronium::random_static;

ClassificationEnvironment::ClassificationEnvironment(
    int num_classes,
    int num_features,
    int p_size,
    int p_gap,
    int tau,
    const std::vector<std::vector<float>> &X,
    const std::vector<int>                &y) :

    num_classes(num_classes),
    num_features(num_features),
    p_size(p_size),
    p_gap(p_gap),
    tau(tau),
    dataset(X, y) {}

std::vector<Learner>
ClassificationEnvironment::initializeLearners(int action) {
    std::vector<Learner> learners;

    for( int i = 0; i < (p_size - p_gap); i++ ) {
        // TODO: Remove magic registers number
        Learner learner(action, 8, num_classes, num_features);
        learners.push_back(learner);
    }

    return learners;
}

void
ClassificationEnvironment::generateLearners(std::vector<Learner> &learner_pop) {

    // Vector to store offspring of current Learner population
    std::vector<Learner> new_learners;
    for( int i = 0; i < p_gap; i++ ) {

        // Copy member of current Learner pop into new_learner
        // TODO: Confirm this works as expected!!
        Learner new_learner = *Random::get(learner_pop);

        // Stochastically mutate the Learner
        new_learner.mutate();

        // Add new Learner to gap population
        new_learners.push_back(new_learner);
    }

    // Add new Learners to the Learner population
    learner_pop.insert(learner_pop.end(), new_learners.begin(), new_learners.end());
}

std::vector<Point>
ClassificationEnvironment::generatePoints() {
    return dataset.getRandomExemplars(tau);
}

void
ClassificationEnvironment::train(int num_generations) {

    // Reserve space for outcome matrix
    std::vector< std::vector<float> > G;
    G.resize(p_size);
    for( int i = 0; i < G.size(); i++ ) {
        G[i].resize(tau);
    }

    for( int action = 0; action < num_classes; action++ ) {

        // Initialize Learners
        std::vector<Learner> learners = initializeLearners(action);

        for(int t = 0; t < num_generations; t++) {

            // Generate new individuals
            generateLearners(learners);

            // Generate new Points
            std::vector<Point> points = generatePoints();

            // Calculate the outcome of each Learner on each Point
            for( int i = 0; i < p_size; i++ ) {
                for( int k = 0; k < tau; k++ ) {
                    float bid = learners[i].bid(points[k].X);
                    if( learners[i].action == points[k].y ) {
                        G[i][k] = bid;
                    } else {
                        G[i][k] = 1.0 - bid;
                    }
                }
            }

            // Sum across rows of G to calculate fitness
            for( int i = 0; i < p_size; i++ ) {
                learners[i].fitness = 0.0;
                for( int k = 0; k < tau; k++ ) {
                    learners[i].fitness += G[i][k];
                }
            }

            // Sort Learners by
            std::sort(learners.begin(), learners.end(), [](Learner &a, Learner &b) {
                return a.fitness > b.fitness;
            });

            // Remove the p_gap worst performing individuals
            // (Technically, we save the (p_size - p_gap) individuals)
            int p_keep = p_size - p_gap;
            learners.erase(learners.begin() + p_keep, learners.end());

            std::cout << '\r' << std::flush;
            std::cout << "Action: " << action << " Gen: " << t << " Best fitness: " << learners[0].fitness;
        }
        std::cout << std::endl;

        S.insert(S.end(), learners.begin(), learners.end());
    }
}

int
ClassificationEnvironment::classify(const std::vector<float> &features) {

    // Let each Learner in the solution bid on the exemplar and
    // find the action of the Learner with the highest bid
    float max_bid    = 0.0;
    int   max_action = -1;
    for( int i = 0; i < S.size(); i++ ) {
        if( S[i].bid(features) > max_bid ) {
            max_bid    = S[i].registers[0];
            max_action = S[i].action;
        }
    }

    if( max_action == -1 ) {
        std::cout << "ClassEnv::classify - Error - max_action == -1" << std::endl;
    }

    return max_action;
}

float
ClassificationEnvironment::accuracy(const std::vector<std::vector<float>> &X,
                                    const std::vector<int>                &y) {

    if( X.size() != y.size() ) {
        std::cout << "ClassEnv::accuracy - Error - X and y are of unequal lengths" << std::endl;
        return -1.0;
    }

    float num_correct = 0.0;
    for( int i = 0; i < X.size(); i++ ) {
        int prediction = classify(X[i]);
        if( prediction == y[i] ) num_correct += 1.0;
    }

    return num_correct / X.size();
}
