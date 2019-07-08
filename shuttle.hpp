#ifndef __SHUTTLE_HPP
#define __SHUTTLE_HPP

#include "csv.h"

// shuffle algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::shuffle
#include <vector>       // std::vector
#include <array>
#include <random>       // std::default_random_engine
#include <string>

class ShuttleStatlogDataset {

public:
    const int num_training_samples = 43500;
    const int num_test_samples     = 14500;
    const int num_samples          = num_training_samples + num_test_samples;
    const int num_classes          = 7;
    const int num_features         = 9;

    std::vector< std::vector<float> > train_X;
    std::vector<int>                  train_y;

    std::vector< std::vector<float> > test_X;
    std::vector<int>                  test_y;

    ShuttleStatlogDataset() {

        // Fields
        float a, b, c, d, e, f, g, h, i;
        int   label;

        std::vector< std::vector<float> > data;
        std::vector<int>                  labels;

        io::CSVReader<10> train_in("shuttle-normalized.trn");


        while( train_in.read_row(a,b,c,d,e,f,g,h,i,label) )
        {
            data.push_back({a,b,c,d,e,f,g,h,i});
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

        io::CSVReader<10> test_in("shuttle-normalized.tst");

        while( test_in.read_row(a,b,c,d,e,f,g,h,i,label) )
        {
            data.push_back({a,b,c,d,e,f,g,h,i});
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

#endif //__SHUTTLE_HPP
