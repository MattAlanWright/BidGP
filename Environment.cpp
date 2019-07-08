#include "Environment.hpp"

#include <cmath>

// Effolkronium random library
#include "random.hpp"
using Random = effolkronium::random_static;

// Return true if a Pareto-dominates b
bool paretoDominates(const std::vector<float> &a, const std::vector<float> &b) {

    if( a.size() != b.size() ) {
        std::cout << "paretoDominates - Error - a and b are of unequal length." << std::endl << std::endl;
        return false;
    }

    bool does_dominate = false;
    for( int i = 0; i < a.size(); i++ ) {
        if( a[i] < b[i] ) {
            return false;
        }

        if( a[i] > b[i] ) {
            does_dominate = true;
        }
    }

    return does_dominate;
}

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

        if( Random::get<float>(0.0, 1.0) < 0.8 ) {

            // Copy member of current Learner pop into new_learner
            Learner new_learner = *Random::get(learner_pop);

            // Stochastically mutate the Learner
            new_learner.mutate();

            // Add new Learner to gap population
            new_learners.push_back(new_learner);

        } else {
            Learner new_learner(learner_pop[0].action, 8, num_classes, num_features);

            // Add new Learner to gap population
            new_learners.push_back(new_learner);
        }
    }

    // Add new Learners to the Learner population
    learner_pop.insert(learner_pop.end(), new_learners.begin(), new_learners.end());
}

std::vector<Point>
ClassificationEnvironment::generatePoints() {
    return dataset.getRandomExemplars(tau);
}

float
ClassificationEnvironment::standardFitness(std::vector<Learner>              &learners,
                                           std::vector< std::vector<float> > &G)
{
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

    return learners[0].fitness / tau;
}

float
ClassificationEnvironment::crossEntropyFitness(std::vector<Learner>              &learners,
                                               std::vector< std::vector<float> > &G)
{
    // Sum across rows of G to calculate per-class accuracy
    float max_fitness = 0.0;
    for( int i = 0; i < p_size; i++ ) {
        float fitness = 0.0;
        for( int k = 0; k < tau; k++ ) {
            fitness += G[i][k];
        }
        if( fitness > max_fitness ) {
            max_fitness = fitness;
        }
    }

    // Sum across rows of G to calculate fitness
    for( int i = 0; i < p_size; i++ ) {
        learners[i].fitness = 0.0;
        for( int k = 0; k < tau; k++ ) {
            learners[i].fitness += -std::log(G[i][k]);
        }
    }

    // Sort Learners by
    std::sort(learners.begin(), learners.end(), [](Learner &a, Learner &b) {
        return a.fitness < b.fitness;
    });

    // Remove the p_gap worst performing individuals
    // (Technically, we save the (p_size - p_gap) individuals)
    int p_keep = p_size - p_gap;
    learners.erase(learners.begin() + p_keep, learners.end());

    return max_fitness / tau;
}

float
ClassificationEnvironment::fitnessSharing(std::vector<Learner>              &learners,
                                          std::vector< std::vector<float> > &G)
{
    // Sum across rows of G to calculate per-class accuracy
    float max_fitness = 0.0;
    for( int i = 0; i < p_size; i++ ) {
        float fitness = 0.0;
        for( int k = 0; k < tau; k++ ) {
            fitness += G[i][k];
        }
        if( fitness > max_fitness ) {
            max_fitness = fitness;
        }
    }

    // Sum down columns to create scaling factor. Each entry of G[i][k] is
    // scaled by the sum of the performance of all Learners on point k.
    // This performance is stored in denom[k].
    std::vector<float> denom(G[0].size(), 0.0);
    for( int k = 0; k < tau; k++ ) {
        for( int i = 0; i < p_size; i++ ) {
            denom[k] += G[i][k];
        }
    }

    // Scale entries of G
    for( int k = 0; k < tau; k++ ) {
        for( int i = 0; i < p_size; i++ ) {
            G[i][k] /= denom[k];
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

    return max_fitness / tau;
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

        std::vector<Point> points;
        for(int t = 0; t < num_generations; t++) {

            // Generate new individuals
            generateLearners(learners);

            // Generate new Points
            if( t == 0) points = generatePoints();

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

#ifdef SIMPLE_FITNESS
            float fitness = standardFitness(learners, G);

            std::cout << "Class: " << action
                      << " Gen: " << t
                      << " Class accuracy: "
                      << fitness
                      << "                    "
                      << '\r' << std::flush;

#endif

#ifdef CROSS_ENTROPY
            float fitness = crossEntropyFitness(learners, G);

            std::cout << "Class: " << action
                      << " Gen: " << t
                      << " Class accuracy: "
                      << fitness
                      << "                    "
                      << '\r' << std::flush;

#endif

#ifdef FITNESS_SHARING
            float fitness = fitnessSharing(learners, G);

            std::cout << "Class: " << action
                      << " Gen: " << t
                      << " Class accuracy: "
                      << fitness
                      << "                    "
                      << '\r' << std::flush;
#endif
        }

        std::cout << std::endl;
        S.insert(S.end(), learners.begin(), learners.end());
    }
}

int
ClassificationEnvironment::classify(const std::vector<float> &features) {

    // Let each Learner in the solution bid on the exemplar and
    // find the action of the Learner with the highest bid
    float max_bid    = S[0].bid(features);
    int   max_action = S[0].action;
    for( int i = 1; i < S.size(); i++ ) {
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

float
ClassificationEnvironment::detectionRate(const std::vector<std::vector<float>> &X,
                                         const std::vector<int>                &y) {
    float detection_rate = 0.0;

    std::vector<int> true_positives(num_classes, 0);
    std::vector<int> false_negatives(num_classes, 0);

    for( int i = 0; i < X.size(); i++ ) {
        int prediction = classify(X[i]);
        if( prediction == y[i]) {
            true_positives[y[i]]++;
        } else {
            false_negatives[y[i]]++;
        }
    }

    for( int i = 0; i < num_classes; i++ ) {
        if( true_positives[i] == 0 ) {
            detection_rate += 0;
        } else {
            detection_rate += (float)true_positives[i] / (float)(true_positives[i] + false_negatives[i]);
        }
    }

    return detection_rate / num_classes;
}
