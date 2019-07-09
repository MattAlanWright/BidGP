#! /bin/sh
echo "BALANCED DATA"

echo "Simple fitness"
./build.sh SIMPLE_FITNESS && ./BGP

echo "Fitness sharing"
./build.sh FITNESS_SHARING && ./BGP
