//
// Created by Fly on 2026/2/23.
//

#ifndef MACHINELEARNING_NETWORK_H
#define MACHINELEARNING_NETWORK_H

#include "Matrix.h" // 包含你修改后的矩阵类
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdexcept>  // 补充必要头文件
#include <omp.h>      // OpenMP 头文件（多线程需要）

struct Datapoint
{
    Matrix<double> x;
    int y;
    Datapoint() = default;
    Datapoint(const Matrix<double>& x_, int y_) : x(x_), y(y_) {}
};

class Network
{
private:
    int num_layers;
    std::vector<int> sizes;
    std::vector<Matrix<double>> weights;
    std::vector<Matrix<double>> biases;

    /**
     * @brief Sigmoid激活函数(作用于整个矩阵)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在(0, 1)之间)
     */
    static Matrix<double>sigmoid(const Matrix<double>& z)
    {
        Matrix<double> result(z.getRows(), z.getCols());
        for (size_t r = 0; r < z.getRows(); r++)
            for (size_t c = 0; c < z.getCols(); c++)
            {
                double val = z(r, c);
                if (val > 15.0)
                    result(r, c) = 1.0;
                else if (val < -15.0)
                    result(r, c) = 0.0;
                else
                    result(r, c) = 1.0 / (1.0 + std::exp(-val));
            }

        return result;
    }

    /**
     * @brief Sigmoid函数的导数(兼容原有接口，内部调用优化版)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在(0, 1)之间)
     */
    static Matrix<double> sigmoid_prime(const Matrix<double>& a)
    {
        Matrix<double> result(a.getRows(), a.getCols());
        for (size_t r = 0; r < a.getRows(); r++)
            for (size_t c = 0; c < a.getCols(); c++)
            {
                double val = a(r, c);
                result(r, c) = val * (1.0 - val);
            }

        return result;
    }

    /**
     * @brief ReLU激活函数(作用于整个矩阵)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在[0, +∞)之间)
     */
    static Matrix<double> relu(const Matrix<double>& z)
    {
        Matrix<double> result(z.getRows(), z.getCols());
        for (size_t r = 0; r < z.getRows(); r++)
            for (size_t c = 0; c < z.getCols(); c++)
                result(r, c) = std::max(0.0, z(r, c));

        return result;
    }

    /**
     * @brief ReLU函数的导数(作用于整个矩阵)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在[0, 1]之间)
     */
    static Matrix<double> relu_prime(const Matrix<double>& z)
    {
        Matrix<double> result(z.getRows(), z.getCols());
        for (size_t r = 0; r < z.getRows(); r++)
            for (size_t c = 0; c < z.getCols(); c++)
                result(r, c) = (z(r, c) > 0.0) ? 1.0 : 0.0;

        return result;
    }

public:
    /**
     * @brief 构造函数:初始化神经网络的架构和随机权重和偏置
     * @param sizes_ 神经网络的架构, 例如{784, 30, 10} 表示输入层784个神经元, 隐藏层30个神经元, 输出层10个神经元
     */
    Network(const std::vector<int>& sizes_) : sizes(sizes_)
    {
        if (sizes.size() < 2)
            throw std::invalid_argument("Error: Network size must be at least 2!");
        num_layers = sizes.size();

        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 1; i < sizes.size(); i++)
        {
            Matrix<double> b(sizes[i], 1);
            std::uniform_real_distribution<double> b_dist(0.0, 1.0);
            for (size_t r = 0; r < b.getRows(); r++)
                b(r, 0) = b_dist(gen);
            biases.push_back(std::move(b));

            Matrix<double> w(sizes[i], sizes[i - 1]);
            double stddev = std::sqrt(2.0 / (sizes[i - 1] + sizes[i]));
            std::normal_distribution<double> w_dist(0.0, stddev);
            for (size_t r = 0; r < w.getRows(); r++)
                for (size_t c = 0; c < w.getCols(); c++)
                    w(r, c) = w_dist(gen);
            weights.push_back(std::move(w));
        }
    }

    /**
     * @brief 前向传播（修复ReLU激活函数的bug）
     * @param a 输入向量 (列向量)
     * @param idx 激活函数索引 (0: Sigmoid, 1: ReLU)
     * @return 输出向量 (列向量)
     */
    Matrix<double> feedforward(const Matrix<double> a, int idx = 1)
    {
        Matrix<double> activation = a;
        for (size_t i = 0; i < biases.size(); i++)
        {
            Matrix<double> z = weights[i] * activation + biases[i]; // 修复：原代码用了a而非activation
            activation = (idx == 0) ? sigmoid(z) : relu(z);
        }
        return activation;
    }

    /**
     * @brief 计算代价函数的导数 (这里output_activation 是一个列向量)
     * @param output_activation 输出向量 (列向量)
     * @param y 正确的标签 (0到num_classes-1之间的整数)
     * @return 代价函数的导数向量 (列向量)
     */
    Matrix<double> cost_derivative(const Matrix<double>& output_activation, int y)
    {
        Matrix<double> result = output_activation;

        if (y < 0 || static_cast<size_t>(y) >= result.getRows())
            throw std::out_of_range("Error: Label out of range!");

        result(y, 0) -= 1.0; // 优化写法：等价于 result(y,0) = result(y,0) - 1.0
        return result;
    }

    /**
     * @brief 反向传播,返回偏置和权重的梯度（修复delta初始化bug）
     * @param x 输入向量 (列向量)
     * @param y 正确的标签 (0到num_classes-1之间的整数)
     * @param delta_nabla_b 偏置的梯度向量 (每个元素都是一个列向量)
     * @param delta_nabla_w 权重的梯度向量 (每个元素都是一个矩阵)
     * @param idx 激活函数索引 (0: Sigmoid, 1: ReLU)
     */
    void backprop(const Matrix<double>& x, int y,
                    std::vector<Matrix<double>>& delta_nabla_b,
                    std::vector<Matrix<double>>& delta_nabla_w, int idx = 1)
    {
        delta_nabla_b.clear();
        delta_nabla_w.clear();
        for (const auto& b : biases) delta_nabla_b.emplace_back(b.getRows(), b.getCols(), 0.0);
        for (const auto& w : weights) delta_nabla_w.emplace_back(w.getRows(), w.getCols(), 0.0);

        Matrix<double> activation = x;
        std::vector<Matrix<double>> activations;
        activations.reserve(num_layers);
        activations.push_back(x);

        std::vector<Matrix<double>> zs;
        zs.reserve(num_layers - 1);

        // 前向传播并记录 z 和 a
        for (size_t i = 0; i < biases.size(); i++)
        {
            Matrix<double> z = weights[i] * activation + biases[i];
            zs.push_back(z);
            activation = (idx == 0) ? sigmoid(z) : relu(z);
            activations.push_back(activation);
        }

        // 反向传播核心算法
        // 输出层误差（修复：显式初始化delta，避免无参构造问题）
        Matrix<double> cost_grad = activations.back();
        cost_grad(y, 0) -= 1.0;

        Matrix<double> sp_out = (idx == 0) ? sigmoid_prime(activations.back()) : relu_prime(zs.back());
        Matrix<double> delta = cost_grad.hadamard(sp_out);

        delta_nabla_b.back() = delta;
        delta_nabla_w.back() = delta * activations[activations.size() - 2].transpose();

        // 隐藏层误差（修复：sp显式初始化，避免无参构造问题）
        for (size_t l = 2; l < num_layers; l++)
        {
            const Matrix<double> z = zs[zs.size() - l];
            Matrix<double> sp = (idx == 0) ? sigmoid_prime(activations[activations.size() - l]) : relu_prime(z);

            delta = (weights[weights.size() - l + 1].transpose() * delta).hadamard(sp);

            delta_nabla_b[num_layers - 1 - l] = delta;
            delta_nabla_w[num_layers - 1 - l] = delta * activations[activations.size() - l - 1].transpose();
        }
    }

    /**
     * @brief 更新小批量数据的权重和偏置（核心：OpenMP多线程并行）
     * @param mini_batch 小批量数据 (每个元素都是一个 Datapoint)
     * @param eta 学习率
     */
    void update_mini_batch(const std::vector<Datapoint>& mini_batch, double eta)
    {
        std::vector<Matrix<double>> nabla_b;
        std::vector<Matrix<double>> nabla_w;
        for (const auto& b : biases) nabla_b.emplace_back(b.getRows(), b.getCols(), 0.0);
        for (const auto& w : weights) nabla_w.emplace_back(w.getRows(), w.getCols(), 0.0);

        // 每个线程有独立的局部梯度，避免竞争
        #pragma omp parallel
        {
            // 线程局部梯度
            std::vector<Matrix<double>> local_nb;
            std::vector<Matrix<double>> local_nw;
            for (const auto& b : biases) local_nb.emplace_back(b.getRows(), b.getCols(), 0.0);
            for (const auto& w : weights) local_nw.emplace_back(w.getRows(), w.getCols(), 0.0);

            std::vector<Matrix<double>> d_nb(num_layers - 1), d_nw(num_layers - 1);

            // 并行遍历mini_batch，每个线程处理部分样本
            #pragma omp for nowait
            for (size_t i = 0; i < mini_batch.size(); ++i)
            {
                backprop(mini_batch[i].x, mini_batch[i].y, d_nb, d_nw);
                for (size_t j = 0; j < local_nb.size(); j++)
                {
                    local_nb[j] += d_nb[j];
                    local_nw[j] += d_nw[j];
                }
            }

            // 合并线程局部梯度到全局梯度
            #pragma omp critical
            {
                for (size_t j = 0; j < nabla_b.size(); j++)
                {
                    nabla_b[j] += local_nb[j];
                    nabla_w[j] += local_nw[j];
                }
            }
        }

        // 更新权重和偏置
        double factor = eta / mini_batch.size();
        for (size_t i = 0; i < weights.size(); i++)
        {
            weights[i] = weights[i] - (nabla_w[i] * factor);
            biases[i] = biases[i] - (nabla_b[i] * factor);
        }
    }

    /**
     * @brief 评估网络在测试数据上的性能（优化：OpenMP多线程评估）
     * @param test_data 测试数据 (每个元素都是一个 Datapoint)
     * @return 正确分类的样本数
     */
    int evaluate(const std::vector<Datapoint>& test_data)
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

    /**
     * @brief 使用小批量随机梯度下降训练网络（优化输出格式）
     * @param training_data 训练数据 (每个元素都是一个 Datapoint)
     * @param epochs 训练轮数
     * @param mini_batch_size 小批量大小
     * @param eta 学习率
     * @param test_data 测试数据 (每个元素都是一个 Datapoint), 可选
     */
    void SGD(std::vector<Datapoint>& training_data, int epochs, int mini_batch_size, double eta,
        const std::vector<Datapoint>* test_data = nullptr)
    {
        int n_test = test_data ? static_cast<int>(test_data->size()) : 0;
        int n = static_cast<int>(training_data.size());

        std::random_device rd;
        std::mt19937 g(rd());

        for (int j = 0; j < epochs; j++)
        {
            // 打乱训练数据
            std::shuffle(training_data.begin(), training_data.end(), g);

            // 分割为 mini-batches 并训练
            for (int k = 0; k < n; k += mini_batch_size)
            {
                int end_idx = std::min(k + mini_batch_size, n);
                std::vector<Datapoint> mini_batch(training_data.begin() + k, training_data.begin() + end_idx);
                update_mini_batch(mini_batch, eta);
            }

            // 优化输出格式，避免换行混乱
            if (test_data)
            {
                int num = evaluate(*test_data);
                double accuracy = static_cast<double>(num) / n_test * 100.0;
                std::cout << "Epoch " << j + 1 << ": " << num << " / " << n_test
                          << " (" << std::fixed << std::setprecision(2) << accuracy << "%)" << std::endl;
            }
            else
                std::cout << "Epoch " << j + 1 << " completed!" << std::endl;
        }
    }

    /**
     * @brief 保存网络模型到文件（优化：二进制写入效率）
     * @param file_name 文件名 (不包含扩展名)
     */
    void save(const std::string& file_name)
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

    /**
     * @brief 从文件加载网络模型（优化：二进制读取效率）
     * @param file_name 文件名 (不包含扩展名)
     */
    void load(const std::string& file_name)
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

    int predict(const Matrix<double>& input)
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
};

#endif //MACHINELEARNING_NETWORK_H