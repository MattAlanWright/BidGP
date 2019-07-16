os=$(uname -s)
if [ $os = Linux ]; then
    g++ -std=c++17 main.cpp Environment.cpp Dataset.cpp Point.cpp Learner.cpp -o BGP -lpthread
    rm *.o
elif [ $os = Darwin ]; then
    echo "Building BidGP on MacOS..."
    g++ -std=c++17 main.cpp Environment.cpp Dataset.cpp Point.cpp Learner.cpp -o BGP
    rm *.o
    echo "Done"
fi
