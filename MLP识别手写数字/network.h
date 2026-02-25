//
// Created by Fly on 2026/2/23.
//

#ifndef MACHINELEARNING_NETWORK_H
#define MACHINELEARNING_NETWORK_H

#include "Matrix.h"
#include <vector>

struct Datapoint
{
    Matrix<double> x;
    int y = 0;
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
     * @brief ReLU激活函数(作用于整个矩阵)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在[0, +∞)之间)
     */
    static Matrix<double> relu(const Matrix<double>& z);

    /**
     * @brief ReLU函数的导数(作用于整个矩阵)
     * @param z 输入矩阵
     * @return 输出矩阵 (每个元素都在[0, 1]之间)
     */
    static Matrix<double> relu_prime(const Matrix<double>& z);

public:
    /**
     * @brief 构造函数:初始化神经网络的架构和随机权重和偏置
     * @param sizes_ 神经网络的架构, 例如{784, 30, 10} 表示输入层784个神经元, 隐藏层30个神经元, 输出层10个神经元
     */
    Network(const std::vector<int>& sizes_);

    /**
     * @brief 前向传播（修复ReLU激活函数的bug）
     * @param a 输入向量 (列向量)
     * @return 输出向量 (列向量)
     */
    [[nodiscard]] Matrix<double> feedforward(const Matrix<double>& a) const;

    /**
     * @brief 计算代价函数的导数 (这里output_activation 是一个列向量)
     * @param output_activation 输出向量 (列向量)
     * @param y 正确的标签 (0到num_classes-1之间的整数)
     * @return 代价函数的导数向量 (列向量)
     */
    static Matrix<double> cost_derivative(const Matrix<double>& output_activation, int y);

    /**
     * @brief 反向传播,返回偏置和权重的梯度（修复delta初始化bug）
     * @param x 输入向量 (列向量)
     * @param y 正确的标签 (0到num_classes-1之间的整数)
     * @param delta_nabla_b 偏置的梯度向量 (每个元素都是一个列向量)
     * @param delta_nabla_w 权重的梯度向量 (每个元素都是一个矩阵)
     * @param zs 存储每个层的净输入 (每个元素都是一个列向量)
     * @param activations 存储每个层的激活输出 (每个元素都是一个列向量)
     */
    void backprop(const Matrix<double>& x, int y,
                    std::vector<Matrix<double>>& delta_nabla_b,
                    std::vector<Matrix<double>>& delta_nabla_w,
                    std::vector<Matrix<double>>& zs,
                    std::vector<Matrix<double>>& activations) const;

    /**
     * @brief 更新小批量数据的权重和偏置（核心：OpenMP多线程并行）
     * @param mini_batch 小批量数据 (每个元素都是一个 Datapoint)
     * @param eta 学习率
     * @param lambda 正则化参数
     * @param n_total 总样本数 (用于正则化)
     */
    void update_mini_batch(const std::vector<Datapoint>& mini_batch, double eta, double lambda, int n_total);

    /**
     * @brief 评估网络在测试数据上的性能（优化：OpenMP多线程评估）
     * @param test_data 测试数据 (每个元素都是一个 Datapoint)
     * @return 正确分类的样本数
     */
    [[nodiscard]] int evaluate(const std::vector<Datapoint>& test_data) const;

    /**
     * @brief 使用小批量随机梯度下降训练网络（优化输出格式）
     * @param train_data 训练数据 (每个元素都是一个 Datapoint)
     * @param epochs 训练轮数
     * @param mini_batch_size 小批量大小
     * @param eta 学习率
     * @param lambda 正则化参数
     * @param test_data 测试数据 (每个元素都是一个 Datapoint), 可选
     */
    void SGD(std::vector<Datapoint>& train_data, int epochs, int mini_batch_size, double eta,
        double lambda, const std::vector<Datapoint>* test_data);

    /**
     * @brief 保存网络模型到文件（优化：二进制写入效率）
     * @param file_name 文件名 (不包含扩展名)
     */
    void save(const std::string& file_name);

    /**
     * @brief 从文件加载网络模型（优化：二进制读取效率）
     * @param file_name 文件名 (不包含扩展名)
     */
    void load(const std::string& file_name);

     /**
     * @brief 预测输入样本的类别（修复：返回类别索引而不是概率）
     * @param input 输入向量 (列向量)
     * @return 预测的类别索引 (0到num_classes-1之间的整数)
     */
    int predict(const Matrix<double>& input);
};

#endif //MACHINELEARNING_NETWORK_H