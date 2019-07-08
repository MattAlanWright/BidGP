#ifndef _DATASET_HPP
#define _DATASET_HPP

#include <vector>

#include "Point.hpp"

class UniformDataset {

public:
    UniformDataset(const std::vector< std::vector<float> > &X,
                   const std::vector<int>                  &y);

    // One-dimensional vector, indexed into as:
    //
    //   Point point = dataset[sample number]
    //
    std::vector<Point> dataset;
    std::vector<Point> getRandomExemplars(int tau);
};


class BalancedDataset {

public:
    BalancedDataset(const std::vector< std::vector<float> > &X,
                    const std::vector<int>                  &y,
                    int num_clases);

    int num_classes;

    // Two-dimensional vector, indexed into as:
    //
    //   Point point = dataset[class number][in-class sample number]
    //
    std::vector< std::vector<Point> > dataset;
    std::vector<Point> getRandomExemplars(int tau);
};

#endif //_DATA_HPP
