#ifndef __THYROID_HPP
#define __THYROID_HPP

#include "csv.h"

// shuffle algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::shuffle
#include <vector>       // std::vector
#include <array>
#include <random>       // std::default_random_engine
#include <string>

class ThyroidDataset {

public:
    const int num_training_samples = 3772;
    const int num_test_samples     = 3428;
    const int num_samples          = num_training_samples + num_test_samples;
    const int num_classes          = 3;
    const int num_features         = 21;

    std::vector< std::vector<float> > train_X;
    std::vector<int>                  train_y;

    std::vector< std::vector<float> > test_X;
    std::vector<int>                  test_y;

    ThyroidDataset() {

        // Fields
        float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u;
        int   label;

        std::vector< std::vector<float> > data;
        std::vector<int>                  labels;

        io::CSVReader<22> train_in("ann-train-normalized.data");


        while( train_in.read_row(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,label) )
        {
            data.push_back({a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u});
            labels.push_back(label);
        }

        // Shuffle data
        std::default_random_engine engine;
        engine.seed(42);
        std::shuffle(std::begin(data), std::end(data), engine);
        engine.seed(42);
        std::shuffle(std::begin(labels), std::end(labels), engine);

        // Read values directly into raw float arrays
        train_X.insert(train_X.begin(), data.begin(), data.end());
        train_y.insert(train_y.begin(), labels.begin(), labels.end());

        data.clear();
        labels.clear();

        io::CSVReader<22> test_in("ann-train-normalized.data");

        while( test_in.read_row(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,label) )
        {
            data.push_back({a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u});
            labels.push_back(label);
        }

        engine.seed(42);
        std::shuffle(std::begin(data), std::end(data), engine);
        engine.seed(42);
        std::shuffle(std::begin(labels), std::end(labels), engine);

        test_X.insert(test_X.begin(), data.begin(), data.end());
        test_y.insert(test_y.begin(), labels.begin(), labels.end());
    }
};

#endif //__THYROID_HPP
