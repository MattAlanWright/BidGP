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
    int num_points = (dataset.size() >= tau) ? tau : dataset.size();

    // Shuffle dataset uniformly and take the first tau Points
    Random::shuffle(dataset);

    auto start = dataset.begin();
    auto end   = dataset.begin() + num_points;
    std::vector<Point> subset(start, end);
    return subset;
}

int UniformDataset::getSubsetSize(int tau) {
    return (dataset.size() >= tau) ? tau : dataset.size();;
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
        int num_points = (dataset[i].size() >= tau) ? tau : dataset[i].size();

        // Shuffle dataset uniformly and take the first class_subset_size Points
        Random::shuffle(dataset[i]);

        auto start = dataset[i].begin();
        auto end   = dataset[i].begin() + num_points;
        std::vector<Point> class_subset(start, end);

        // Insert class_subset into subset
        subset.insert(subset.end(), class_subset.begin(), class_subset.end());

        if( num_points < tau ) {
            int num_sub_subsets = tau / num_points - 1;

            for( int j = 0; j < num_sub_subsets; j++ ) {
                auto start = dataset[i].begin();
                auto end   = dataset[i].begin() + num_points;
                std::vector<Point> another_class_subset(start, end);

                // Insert class_subset into subset
                subset.insert(subset.end(), another_class_subset.begin(), another_class_subset.end());
            }
        }
    }

    return subset;
}

int BalancedDataset::getSubsetSize(int tau) {
    int subset_size = 0;

    for(int i = 0; i < num_classes; i++) {
        int num_points = (dataset[i].size() >= tau) ? tau : dataset[i].size();
        subset_size += num_points;

        if( num_points < tau ) {
            int num_sub_subsets = tau / num_points - 1;

            for( int j = 0; j < num_sub_subsets; j++ ) {
                subset_size += num_points;
            }
        }
    }

    return subset_size;
}
