//
// Created by Fly on 2026/2/23.
//

#ifndef MACHINELEARNING_MNISTREADER_H
#define MACHINELEARNING_MNISTREADER_H

#include <string>
#include <cstdint>
#include "network.h"

inline uint32_t swap_endian(uint32_t val)
{
    return ((val >> 24) & 0x000000FF) |  // 最高位字节移到最低位
           ((val >> 8)  & 0x0000FF00) |  // 次高位字节移到次低位
           ((val << 8)  & 0x00FF0000) |  // 次低位字节移到次高位
           ((val << 24) & 0xFF000000);   // 最低位字节移到最高位
}

class MNISTReader
{
public:
    /**
     * @brief 从MNIST文件加载数据集
     * @param image_path 图像文件路径
     * @param label_path 标签文件路径
     * @return 包含Datapoint的向量, 每个Datapoint包含一个图像和对应的标签
     */
    static std::vector<Datapoint> load(const std::string& image_path, const std::string& label_path);
};

#endif //MACHINELEARNING_MNISTREADER_H