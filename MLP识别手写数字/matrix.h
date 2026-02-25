//
// Created by Fly on 2026/2/21.
//

#ifndef MACHINELEARNING_MATRIX_H
#define MACHINELEARNING_MATRIX_H

#include <vector>
#include <iomanip>
#include <stdexcept>
#include <string>

template<typename T>
class Matrix
{
private:
    size_t rows_;
    size_t cols_;
    std::vector<T> data_;

    // 辅助函数：计算行优先索引（对外隐藏，不影响原有接口）
    [[nodiscard]] inline size_t getIndex(size_t row, size_t col) const
    {
        return row * cols_ + col;
    }

public:
    // 默认构造函数, 构造一个空矩阵
    Matrix() : rows_(0), cols_(0), data_() {}

    /**
     * @brief 构造一个指定行数和列数, 并用默认值填充的矩阵
     * @param rows 矩阵的行数
     * @param cols 矩阵的列数
     * @param initial_value 初始化值，默认值为T()
     */
    Matrix(size_t rows, size_t cols, const T& initial_value = T())
        : rows_(rows), cols_(cols), data_(rows * cols, initial_value) {}

    /**
     * @brief 使用初始化列表构造矩阵（保留逻辑，适配新存储）
     * @param list 嵌套的初始化列表，例如{ {1, 2, 3}, {4, 5, 6} }
     */
    Matrix(std::initializer_list<std::initializer_list<T>> list)
    {
        if (list.empty())
            throw std::invalid_argument("Error: Cannot initialize matrix with empty list");

        rows_ = list.size();
        cols_ = list.begin()->size();
        data_.reserve(rows_ * cols_);  // 预分配连续内存

        for (const auto& row_list : list)
        {
            if (row_list.size() != cols_)
                throw std::invalid_argument(
                    "Error: Matrix row size mismatch!"
                    "row" + std::to_string(row_list.size()) +
                    "expected " + std::to_string(cols_) + " columns!");

            data_.insert(data_.end(), row_list.begin(), row_list.end());
        }
    }

    // 拷贝构造函数
    Matrix(const Matrix<T>& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}

    /**
     * @brief 移动构造函数
     * @param other 要移动的矩阵
     */
    Matrix(Matrix<T>&& other) noexcept : rows_(other.rows_),cols_(other.cols_), data_(std::move(other.data_))
    {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    /**
     * @brief 获取矩阵的行数,列数
     * @return 矩阵的行数,列数
     */
    [[nodiscard]] inline size_t getRows() const { return rows_; }
    [[nodiscard]] inline size_t getCols() const { return cols_; }

    /**
     * @brief 重载()运算符, 用于读取和修改元素内容(例如mat(0, 1) = 5)
     * 接口完全不变，内部适配连续内存
     * @param row 要访问的元素的行索引
     * @param col 要访问的元素的列索引
     * @return 对指定位置元素的引用
     */
    T& operator()(size_t row, size_t col)
    {
        if (row >= rows_ || col >= cols_)
            throw std::out_of_range("Matrix index out of range");
        return data_[getIndex(row, col)];
    }

    /**
     * @brief 重载()运算符的只读版本（修复原bug：row > rows_ → row >= rows_）
     * @param row 要访问的元素的行索引
     * @param col 要访问的元素的列索引
     * @return 指定位置元素的副本
     */
    const T& operator()(size_t row, size_t col) const
    {
        if (row >= rows_ || col >= cols_)  // 修复原bug：避免越界
            throw std::out_of_range("Matrix index out of range");
        return data_[getIndex(row, col)];  // 核心修改5：用索引访问连续内存
    }

    /**
     * @brief 重载+=运算符, 用于矩阵加法
     * @param other 要相加的矩阵
     * @return 当前矩阵的引用
     */
    Matrix<T>& operator+=(const Matrix<T>& other)
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Error: The dimension of the two matrices must be the same!");

        for (size_t i = 0; i < rows_ * cols_; i++)
            data_[i] += other.data_[i];

        return *this;
    }

    /**
     * @brief 重载+运算符, 用于矩阵加法
     * @param rhs 要相加的矩阵
     * @return 两个矩阵对应元素之和的新矩阵
     */
    Matrix<T> operator+(const Matrix<T>& rhs) const
    {
        Matrix<T> res = *this;
        res += rhs;
        return res;
    }

    /**
     * @brief 重载-=运算符, 用于矩阵减法
     * @param other 要相减的矩阵
     * @return 当前矩阵的引用
     */
    Matrix<T>& operator-=(const Matrix<T>& other)
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Error: The dimension of the two matrices must be the same!");

        for (size_t i = 0; i < data_.size(); i++)
            data_[i] -= other.data_[i];

        return *this;
    }

    /**
     * @brief 重载-运算符, 用于矩阵减法
     * @param other 要相减的矩阵
     * @return 两个矩阵对应元素之差的新矩阵
     */
    Matrix<T> operator-(const Matrix<T>& other) const
    {
        Matrix<T> res = *this;
        res -= other;
        return res;
    }

    /**
     * @brief 重载*运算符, 用于矩阵乘法
     * @param other 要相乘的矩阵
     * @return 两个矩阵的乘积矩阵
     */
    Matrix<T> operator*(const Matrix<T>& other) const
    {
        if (cols_ != other.rows_)
            throw std::invalid_argument("Error: Matrix multiplication dimension mismatch!");

        Matrix<T> result(rows_, other.cols_, T());

        for (size_t i = 0; i < rows_; ++i)
            for (size_t k = 0; k < cols_; ++k)
            {
                T val = data_[getIndex(i, k)];  // 只读取一次，减少重复访问
                for (size_t j = 0; j < other.cols_; ++j)
                    result.data_[result.getIndex(i, j)] += val * other.data_[other.getIndex(k, j)];
            }
        return result;
    }

    /**
     * @brief 重载*运算符, 用于矩阵与标量的乘法
     * @param scalar 要相乘的标量
     * @return 每个元素与标量乘积的新矩阵
     */
    Matrix<T> operator*(const T& scalar) const
    {
        Matrix<T> result(rows_, cols_);
        for (size_t i = 0; i < rows_ * cols_; ++i)
            result.data_[i] = scalar * data_[i];

        return result;
    }



    /**
     * @brief 赋值运算符
     * @param other 要赋值的矩阵
     * @return 当前矩阵的引用
     */
    Matrix<T>& operator=(const Matrix<T>& other)
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;  // 适配新存储
        }
        return *this;
    }

    /**
     * @brief 移动赋值运算符
     * @param other 要移动的矩阵
     * @return 当前矩阵的引用
     */
    Matrix<T>& operator=(Matrix<T>&& other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    /**
    * @brief Hadamard乘积 (两个矩阵/向量逐元素相乘)
    * @param other 输入矩阵/向量
    * @return 输出矩阵/向量 (每个元素都是对应位置元素的乘积)
    */
    Matrix<T> hadamard(const Matrix<T>& other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Error: Hadamard product dimension mismatch!");
        Matrix<T> result(rows_, cols_);
        for (size_t i = 0; i < data_.size(); i++)
            result.data_[i] = this->data_[i] * other.data_[i];

        return result;
    }

    /**
     * @brief 转置矩阵（适配新存储）
     * @return 转置后的矩阵
     */
    Matrix<T> transpose() const
    {
        Matrix<T> result(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result.data_[result.getIndex(j, i)] = data_[getIndex(i, j)];

        return result;
    }

    /*
     * @brief 重载<<运算符, 用于输出矩阵
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat)
    {
        for (size_t i = 0; i < mat.rows_; i++)
        {
            for (size_t j = 0; j < mat.cols_; j++)
                os << std::setw(8) << mat.data_[mat.getIndex(i, j)] << " ";
            os << "\n";
        }
        return os;
    }
};

#endif //MACHINELEARNING_MATRIX_H