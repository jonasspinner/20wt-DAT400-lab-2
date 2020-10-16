//
//  nn_mpi.cpp
//  
//  To compile: mpicxx -std=c++11 -O3 -fopenmp -o train_mpi
//  To run: ./train_mpi



#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <cassert>

#include "deep_core.h"
#include "vector_ops.h"


vector <string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector <string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}


template<typename T>
T sum(const std::vector <T> &vals) {
    T s{};
    for (const auto &x : vals)
        s += x;
    return s;
}

template<typename T>
std::pair <T, T> mean_and_std(const std::vector <T> &vals) {
    T m{};
    for (const auto &x : vals)
        m += x;
    m /= vals.size();

    T s2{};
    for (const auto &x : vals)
        s2 += (x - m) * (x - m);
    s2 /= (vals.size() - 1);
    auto s = std::sqrt(s2);
    return {m, s};
}

void load_data(vector<float> &X_train, vector<float> &y_train) {
    string line;
    vector <string> line_v;
    int len, mpirank = 0;
    cout << "Loading data ...\n";

    ifstream myfile("train.txt");
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            line_v = split(line, '\t');
            int digit = strtof((line_v[0]).c_str(), 0);
            for (unsigned i = 0; i < 10; ++i) {
                if (i == digit) {
                    y_train.push_back(1.);
                } else y_train.push_back(0.);
            }
            int size = static_cast<int>(line_v.size());
            for (unsigned i = 1; i < size; ++i) {
                X_train.push_back(strtof((line_v[i]).c_str(), 0));
            }
        }
        X_train = X_train / 255.0;
        myfile.close();
    } else cout << "Unable to open file" << '\n';
}

int main(int argc, char *argv[]) {
#ifdef USE_MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    if (argc > 1) {
        auto &config = vector_ops::internal::config();
        config.num_threads = std::stoi(argv[1]);
    }
#endif

    std::vector<float> X_train, y_train;

    if (mpi_rank == 0) {
        load_data(X_train, y_train);
    }

    // Some hyperparameters for the NN
    int BATCH_SIZE = 256;
    int LOCAL_BATCH_SIZE = BATCH_SIZE / mpi_size;
    float lr = .01 / BATCH_SIZE;
    // Random initialization of the weights
    vector<float> W1 = random_vector(784 * 128);
    vector<float> W2 = random_vector(128 * 64);
    vector<float> W3 = random_vector(64 * 10);

    std::chrono::time_point <std::chrono::system_clock> t1, t2;
    std::vector<double> time_in_dot_history, time_in_epoch_history;
    auto epoch_time = std::chrono::steady_clock::now();
    auto epoch_wtime = omp_get_wtime();

    if (mpi_rank == 0) {
        cout << "Training the model ...\n";
    }
    for (unsigned i = 0; i < 1000; ++i) {
        t1 = std::chrono::system_clock::now();

        // Building batches of input variables (X) and labels (y)
        vector<float> b_X_master, b_X(LOCAL_BATCH_SIZE * 784);
        vector<float> b_y_master, b_y(LOCAL_BATCH_SIZE * 10);

        if (mpi_rank == 0) {
            int randindx = rand() % (42000 - LOCAL_BATCH_SIZE * mpi_size);
            auto X_begin = X_train.begin() + randindx * 784;
            auto X_end   = X_train.begin() + (randindx + LOCAL_BATCH_SIZE * mpi_size) * 784;
            b_X_master.insert(b_X_master.begin(), X_begin, X_end);
            auto y_begin = y_train.begin() + randindx * 10;
            auto y_end   = y_train.begin() + (randindx + LOCAL_BATCH_SIZE * mpi_size) * 10;
            b_y_master.insert(b_y_master.begin(), y_begin, y_end);
            assert(b_X_master.size() == mpi_size * b_X.size());
            assert(b_y_master.size() == mpi_size * b_y.size());
        }

        MPI_Scatter(b_X_master.data(), b_X.size(), MPI_FLOAT, b_X.data(), b_X.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(b_y_master.data(), b_y.size(), MPI_FLOAT, b_y.data(), b_y.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);


        // Feed forward
        vector<float> a1 = relu(dot(b_X, W1, LOCAL_BATCH_SIZE, 784, 128));
        vector<float> a2 = relu(dot(a1, W2, LOCAL_BATCH_SIZE, 128, 64));
        vector<float> yhat = softmax(dot(a2, W3, LOCAL_BATCH_SIZE, 64, 10), 10);

        // Back propagation
        vector<float> dyhat = (yhat - b_y);
        // dW3 = a2.T * dyhat                   64 x 10
        vector<float> dW3 = dot(transform(&a2[0], LOCAL_BATCH_SIZE, 64), dyhat, 64, LOCAL_BATCH_SIZE, 10);
        // dz2 = dyhat * W3.T * relu'(a2)       B x 64
        vector<float> dz2 = dot(dyhat, transform(&W3[0], 64, 10), LOCAL_BATCH_SIZE, 10, 64) * reluPrime(a2);
        // dW2 = a1.T * dz2                     128 x 64
        vector<float> dW2 = dot(transform(&a1[0], LOCAL_BATCH_SIZE, 128), dz2, 128, LOCAL_BATCH_SIZE, 64);
        // dz1 = dz2 * W2.T * relu'(a1)
        vector<float> dz1 = dot(dz2, transform(&W2[0], 128, 64), LOCAL_BATCH_SIZE, 64, 128) * reluPrime(a1);
        // dW1 = X.T * dz1                      784 x 128
        vector<float> dW1 = dot(transform(&b_X[0], LOCAL_BATCH_SIZE, 784), dz1, 784, LOCAL_BATCH_SIZE, 128);

        std::vector<float> dW1_sum(dW1.size());
        std::vector<float> dW2_sum(dW2.size());
        std::vector<float> dW3_sum(dW3.size());
        MPI_Allreduce(dW1.data(), dW1_sum.data(), dW1.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(dW2.data(), dW2_sum.data(), dW2.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(dW3.data(), dW3_sum.data(), dW3.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // Updating the parameters
        W3 = W3 - lr * dW3_sum;
        W2 = W2 - lr * dW2_sum;
        W1 = W1 - lr * dW1_sum;

        if ((mpi_rank == 0) && (i + 1) % 100 == 0) {
            cout << "Predictions:" << "\n";
            print(yhat, 10, 10);
            cout << "Ground truth:" << "\n";
            print(b_y, 10, 10);
            vector<float> loss_m = yhat - b_y;
            float loss = 0.0;
            for (unsigned k = 0; k < BATCH_SIZE * 10; ++k) {
                loss += loss_m[k] * loss_m[k];
            }

            t2 = std::chrono::system_clock::now();
            chrono::duration<double> elapsed_seconds = t2 - t1;

            double ticks = elapsed_seconds.count();
            cout << "Iteration #: " << i << endl;
            cout << "Iteration Time: " << ticks << "s" << endl;
            cout << "Loss: " << loss / BATCH_SIZE << endl;

#ifndef USE_MPI
            std::chrono::nanoseconds total_time_in_dot{0};
            for (auto &kv : vector_ops::internal::dot_timing_info()) {
                auto N = std::get<0>(kv.first);
                auto K = std::get<1>(kv.first);
                auto M = std::get<2>(kv.first);
                auto duration = kv.second;

                std::cout << std::setw(4) << N << " "
                          << std::setw(4) << K << " "
                          << std::setw(4) << M << " "
                          << std::setw(12) << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
                          << " ms\n";

                total_time_in_dot += duration;
            }

            for (auto &kv : vector_ops::internal::dot_timing_info()) {
                kv.second = std::chrono::nanoseconds{};
            }
#endif

            auto prev_epoch_time = epoch_time;
            epoch_time = std::chrono::steady_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_time - prev_epoch_time);

            auto prev_epoch_wtime = epoch_wtime;
            epoch_wtime = omp_get_wtime();
            auto epoch_wduration = epoch_wtime - prev_epoch_wtime;

            std::cout << "total epoch time "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(epoch_duration).count() << " ms\n";
#ifndef USE_MPI
            std::cout << "time spent in dot "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(total_time_in_dot).count() << " ms\n";
            std::cout << total_time_in_dot.count() * 100.0 / epoch_duration.count() << " %\n";
#endif
            std::cout << "omp_get_wtime() duration " << epoch_wduration << "\n";

#ifndef USE_MPI
            time_in_dot_history.push_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(total_time_in_dot).count());
#endif
            time_in_epoch_history.push_back(
                    std::chrono::duration_cast<std::chrono::milliseconds>(epoch_duration).count());

            cout << "*******************************************" << endl;

        };
    };

    auto p2 = mean_and_std(time_in_epoch_history);
    std::cout << "per epoch " << p2.first << " +- " << p2.second << " ms\n";
#ifndef USE_MPI
    auto p1 = mean_and_std(time_in_dot_history);
    std::cout << "dot in one epoch " << p1.first << " +- " << p1.second << " ms\n";
    std::cout << (sum(time_in_dot_history) * 100.0 / sum(time_in_epoch_history)) << " %\n";
#endif

    MPI_Finalize();
    return 0;
}
