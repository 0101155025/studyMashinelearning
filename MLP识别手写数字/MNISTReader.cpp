//
// Created by pc on 2026/2/25.
//

#include "MNISTReader.h"
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <string>

std::vector<Datapoint> MNISTReader::load(const std::string& image_path, const std::string& label_path)
    {
        std::vector<Datapoint> dataset;

        std::ifstream image_file(image_path, std::ios::binary);
        std::ifstream label_file(label_path, std::ios::binary);

        if (!image_file.is_open())
            throw std::runtime_error("Failed to open image file");
        if (!label_file.is_open())
            throw std::runtime_error("Failed to open label file");

        uint32_t img_magic, num_images, rows, cols;
        image_file.read(reinterpret_cast<char*>(&img_magic), 4);
        image_file.read(reinterpret_cast<char*>(&num_images), 4);
        image_file.read(reinterpret_cast<char*>(&rows), 4);
        image_file.read(reinterpret_cast<char*>(&cols), 4);

        img_magic = swap_endian(img_magic);
        num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (img_magic != 2051)
            throw std::runtime_error("Invalid MNIST image file magic number: " + std::to_string(img_magic));

        uint32_t lbl_magic, num_labels;
        label_file.read(reinterpret_cast<char*>(&lbl_magic), 4);
        label_file.read(reinterpret_cast<char*>(&num_labels), 4);

        lbl_magic = swap_endian(lbl_magic);
        num_labels = swap_endian(num_labels);

        if (lbl_magic != 2049)
            throw std::runtime_error("Invalid MNIST label file magic number: " + std::to_string(lbl_magic));
        if (num_images != num_labels)
            throw std::runtime_error("Image count (" + std::to_string(num_images) +
                                    ") does not match label count (" + std::to_string(num_labels) + ")");


        size_t image_size = rows * cols;
        dataset.reserve(num_images);

        for (uint32_t i = 0; i < num_images; i++)
        {
            Datapoint dp;
            // 1.读取图像并归一化
            dp.x = Matrix<double>(image_size, 1);
            for (int p = 0; p < image_size; p++)
            {
                unsigned char pixel = 0;
                if (!image_file.read(reinterpret_cast<char*>(&pixel), 1))
                    throw std::runtime_error("Failed to read pixel data for sample " + std::to_string(i));
                dp.x(p, 0) = static_cast<double>(pixel) / 255.0;
            }

            // 2.读取标签
            unsigned char label = 0;
            if (!label_file.read(reinterpret_cast<char*>(&label), 1))
                throw std::runtime_error("Failed to read label data for sample " + std::to_string(i));
            dp.y = static_cast<int>(label);

            dataset.emplace_back(std::move(dp.x), dp.y);
        }
        return dataset;
    }