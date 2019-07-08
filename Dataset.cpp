#include "Dataset.hpp"

#include <iostream>

// Effolkronium random library
#include "random.hpp"
using Random = effolkronium::random_static;


UniformDataset::UniformDataset(const std::vector< std::vector<float> > &X,
                               const std::vector<int>                  &y)
{
    if( X.size() != y.size() ) {
        std::cout << "UniformDataset - Error - X and y are of unequal lengths" << std::endl;
    }

    // y[i] = Integer class label
    // X[i] = std::vector<float> of features
    for(int i = 0; i < X.size(); i++) {
        Point p(X[i], y[i]);
        dataset.push_back(p);
    }
}

std::vector<Point> UniformDataset::getRandomExemplars(int tau) {

    // Shuffle dataset uniformly and take the first tau Points
    Random::shuffle(dataset);

    auto start = dataset.begin();
    auto end   = dataset.begin() + tau;
    std::vector<Point> subset(start, end);

    return subset;
}

BalancedDataset::BalancedDataset(const std::vector< std::vector<float> > &X,
                                 const std::vector<int>                  &y,
                                 int num_classes)
    : num_classes(num_classes)
{
    if( X.size() != y.size() ) {
        std::cout << "BalancedDataset - Error - X and y are of unequal lengths" << std::endl;
    }

    dataset.resize(num_classes);

    // y[i] = Integer class label
    // X[i] = std::vector<float> of features
    for(int i = 0; i < X.size(); i++) {
        Point p(X[i], y[i]);
        dataset[y[i]].push_back(p);
    }
}

std::vector<Point> BalancedDataset::getRandomExemplars(int tau) {

    std::vector<Point> subset;

    for(int i = 0; i < num_classes; i++) {
        // Shuffle dataset uniformly and take the first tau Points
        Random::shuffle(dataset[i]);

        int num_points = (dataset[i].size() >= tau) ? tau : dataset[i].size();

        auto start = dataset[i].begin();
        auto end   = dataset[i].begin() + num_points;
        std::vector<Point> class_subset(start, end);

        // Insert class_subset into subset
        subset.insert(subset.end(), class_subset.begin(), class_subset.end());
    }

    Random::shuffle(subset);
    return subset;
}
