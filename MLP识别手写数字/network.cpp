//
// Created by pc on 2026/2/25.
//
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "network.h"

Matrix<double> Network::relu(const Matrix<double>& z)
{
    Matrix<double> result(z.getRows(), z.getCols());
    for (size_t i = 0; i < z.getRows() * z.getCols(); i++)
        result(i / z.getCols(), i % z.getCols()) = std::max(0.0, z(i / z.getCols(), i % z.getCols()));

    return result;
}

Matrix<double> Network::relu_prime(const Matrix<double>& z)
{
    Matrix<double> result(z.getRows(), z.getCols());
    for (size_t i = 0; i < z.getRows() * z.getCols(); i++)
        result(i / z.getCols(), i % z.getCols()) = (z(i / z.getCols(), i % z.getCols()) > 0.0) ? 1.0 : 0.0;

    return result;
}

Network::Network(const std::vector<int>& sizes_) : sizes(sizes_)
{
    if (sizes.size() < 2)
        throw std::invalid_argument("Error: Network size must be at least 2!");

    num_layers = sizes.size();
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 1; i < sizes.size(); i++)
    {
        biases.emplace_back(sizes[i], 1, 0.0);

        // 针对ReLU, 权重使用He初始化
        Matrix<double> w(sizes[i], sizes[i - 1]);
        double stddev = std::sqrt(2.0 / sizes[i - 1]);
        std::normal_distribution<double> w_dist(0.0, stddev);
        for (size_t r = 0; r < w.getRows(); r++)
            for (size_t c = 0; c < w.getCols(); c++)
                w(r, c) = w_dist(gen);
        weights.push_back(std::move(w));
    }
}

Matrix<double> Network::feedforward(const Matrix<double>& a) const
{
    Matrix<double> activation = a;
    for (size_t i = 0; i < biases.size(); i++)
        activation = relu(weights[i] * activation + biases[i]);
    return activation;
}

Matrix<double> Network::cost_derivative(const Matrix<double>& output_activation, int y)
{
    Matrix<double> result = output_activation;

    if (y < 0 || static_cast<size_t>(y) >= result.getRows())
        throw std::out_of_range("Error: Label out of range!");

    result(y, 0) -= 1.0; // 优化写法：等价于 result(y,0) = result(y,0) - 1.0
    return result;
}

void Network::backprop(const Matrix<double>& x, int y,
                    std::vector<Matrix<double>>& delta_nabla_b,
                    std::vector<Matrix<double>>& delta_nabla_w,
                    std::vector<Matrix<double>>& zs,
                    std::vector<Matrix<double>>& activations) const
{
    activations[0] = x;
    // 前向传播并记录 z 和 a
    for (size_t i = 0; i < biases.size(); i++)
    {
        zs[i] = weights[i] * activations[i] + biases[i];
        activations[i + 1] = relu(zs[i]);
    }

    // 反向传播核心算法
    // 输出层误差（修复：显式初始化delta，避免无参构造问题）
    Matrix<double> delta = activations.back();
    delta(y, 0) -= 1.0;

    delta_nabla_b.back() = delta;
    delta_nabla_w.back() = delta * activations[num_layers - 2].transpose();

    // 隐藏层误差（修复：sp显式初始化，避免无参构造问题）
    for (size_t l = 2; l < num_layers; l++)
    {
        Matrix<double> sp = relu_prime(zs[num_layers - 1 - l]);
        delta = (weights[weights.size() - l + 1].transpose() * delta).hadamard(sp);

        delta_nabla_b[num_layers - 1 - l] = delta;
        delta_nabla_w[num_layers - 1 - l] = delta * activations[num_layers - l - 1].transpose();
    }
}

void Network::update_mini_batch(const std::vector<Datapoint>& mini_batch, double eta, double lambda, int n_total)
    {
        std::vector<Matrix<double>> nabla_b;
        std::vector<Matrix<double>> nabla_w;
        for (const auto& b : biases) nabla_b.emplace_back(b.getRows(), b.getCols(), 0.0);
        for (const auto& w : weights) nabla_w.emplace_back(w.getRows(), w.getCols(), 0.0);

        // 每个线程有独立的局部梯度，避免竞争
        #pragma omp parallel
        {
            // 线程局部梯度
            std::vector<Matrix<double>> l_nb, l_nw, zs(num_layers - 1), acts(num_layers);
            std::vector<Matrix<double>> d_nb, d_nw;
            for (const auto& b : biases)
            {
                l_nb.emplace_back(b.getRows(), b.getCols(), 0.0);
                d_nb.emplace_back(b.getRows(), b.getCols(), 0.0);
            }
            for (const auto& w : weights)
            {
                l_nw.emplace_back(w.getRows(), w.getCols(), 0.0);
                d_nw.emplace_back(w.getRows(), w.getCols(), 0.0);
            }

            // 并行遍历mini_batch，每个线程处理部分样本
            #pragma omp for nowait
            for (size_t i = 0; i < mini_batch.size(); i++)
            {
                backprop(mini_batch[i].x, mini_batch[i].y, d_nb, d_nw, zs, acts);
                for (size_t j = 0; j < l_nb.size(); j++)
                {
                    l_nb[j] += d_nb[j];
                    l_nw[j] += d_nw[j];
                }
            }

            // 合并线程局部梯度到全局梯度
            #pragma omp critical
            {
                for (size_t j = 0; j < nabla_b.size(); j++)
                {
                    nabla_b[j] += l_nb[j];
                    nabla_w[j] += l_nw[j];
                }
            }
        }

        // 更新权重和偏置
        double factor = eta / mini_batch.size();
        for (size_t i = 0; i < weights.size(); i++)
        {
            weights[i] = weights[i] * (1.0 - eta * lambda / n_total) - (nabla_w[i] * factor);
            biases[i] = biases[i] - (nabla_b[i] * factor);
        }
    }

int Network::evaluate(const std::vector<Datapoint>& test_data) const
{
    int correct = 0;

    // 多线程并行评估
#pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < test_data.size(); ++i)
    {
        Matrix<double> output = feedforward(test_data[i].x);
        size_t max_idx = 0;
        double max_val = output(0, 0);
        for (size_t j = 1; j < output.getRows(); j++)
            if (output(j, 0) > max_val)
            {
                max_val = output(j, 0);
                max_idx = j;
            }
        if (static_cast<int>(max_idx) == test_data[i].y)
            correct++;
    }
    return correct;
}

void Network::SGD(std::vector<Datapoint>& train_data, int epochs, int mini_batch_size, double eta,
        double lambda, const std::vector<Datapoint>* test_data = nullptr)
{
    int n = train_data.size();
    std::random_device rd;
    std::mt19937 g(rd());

    for (int j = 0; j < epochs; j++)
    {
        // 打乱训练数据
        std::shuffle(train_data.begin(), train_data.end(), g);

        // 分割为 mini-batches 并训练
        for (int k = 0; k < n; k += mini_batch_size)
        {
            int end = std::min(k + mini_batch_size, n);
            std::vector<Datapoint> batch(train_data.begin() + k, train_data.begin() + end);
            update_mini_batch(batch, eta, lambda, n);
        }

        // 优化输出格式，避免换行混乱
        if (test_data)
        {
            int correct = evaluate(*test_data);
            std::cout << "Epoch " << j + 1 << ": " << correct << " / " << test_data->size()
                      << " (" << static_cast<double>(correct) / test_data->size() * 100 << "%)" << std::endl;
        }
        else
            std::cout << "Epoch " << j + 1 << " completed!" << std::endl;
    }
}

void Network::save(const std::string& file_name)
{
    std::ofstream out(file_name + ".bin", std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot open file for writing: " + file_name + ".bin");

    size_t n_layers = weights.size();
    out.write(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));

    for (size_t i = 0; i < n_layers; i++)
    {
        // 写入偏置
        size_t b_rows = biases[i].getRows();
        out.write(reinterpret_cast<char*>(&b_rows), sizeof(b_rows));
        // 优化：一次性写入连续内存的偏置数据
        out.write(reinterpret_cast<const char*>(&biases[i](0, 0)),
            static_cast<std::streamsize>(b_rows * sizeof(double)));

        // 写入权重
        size_t w_rows = weights[i].getRows(), w_cols = weights[i].getCols();
        out.write(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));
        out.write(reinterpret_cast<char*>(&w_cols), sizeof(w_cols));
        // 优化：一次性写入连续内存的权重数据
        out.write(reinterpret_cast<const char*>(&weights[i](0, 0)),
            static_cast<std::streamsize>(w_rows * w_cols * sizeof(double)));
    }
    out.close();
}

void Network::load(const std::string& file_name)
{
    std::ifstream in(file_name + ".bin", std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open file for reading: " + file_name + ".bin");

    size_t saved_layers;
    in.read(reinterpret_cast<char*>(&saved_layers), sizeof(saved_layers));

    // 清空现有参数
    weights.clear();
    biases.clear();

    for (size_t i = 0; i < saved_layers; i++)
    {
        size_t b_rows;
        in.read(reinterpret_cast<char*>(&b_rows), sizeof(b_rows));
        Matrix<double>b(b_rows, 1);
        in.read(reinterpret_cast<char*>(&b(0, 0)), b_rows * sizeof(double));
        biases.push_back(std::move(b));

        size_t w_rows, w_cols;
        in.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));
        in.read(reinterpret_cast<char*>(&w_cols), sizeof(w_cols));
        Matrix<double> w(w_rows, w_cols);
        in.read(reinterpret_cast<char*>(&w(0, 0)), w_rows * w_cols * sizeof(double));
        weights.push_back(std::move(w));
    }
    in.close();
    this->num_layers = weights.size() + 1;
    std::cout << "Successfully loaded " << saved_layers << " layers from model." << std::endl;
}

int Network::predict(const Matrix<double>& input)
{
    Matrix<double> output = feedforward(input);
    int max_idx = 0;
    double max_val = output(0, 0);
    for (size_t i = 1; i < output.getRows(); i++)
        if (output(i, 0) > max_val)
        {
            max_val = output(i, 0);
            max_idx = i;
        }
    return max_idx;
}