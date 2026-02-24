#include <iostream>
#include "MNISTReader.h"
#include "Network.h"

// 打印数字图像
void print_digit(const Matrix<double>& x) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // 根据像素值亮度选择字符
            double val = x(i * 28 + j, 0);
            if (val < 0.1) std::cout << "  ";
            else if (val < 0.5) std::cout << "..";
            else if (val < 0.8) std::cout << "++";
            else std::cout << "##";
        }
        std::cout << "\n";
    }
}

// 推理模式：允许用户输入索引查看模型预测结果
void run_inference(Network& net, const std::vector<Datapoint>& test_data) {
    std::cout << "\n--- Inference Mode ---" << std::endl;
    std::cout << "Pick a sample index (0-9999): ";
    int idx;
    std::cin >> idx;

    if (idx < 0 || idx >= test_data.size()) {
        std::cout << "Invalid index!" << std::endl;
        return;
    }

    const auto& sample = test_data[idx];
    print_digit(sample.x);
    int p = net.predict(sample.x);

    std::cout << "Target Label: " << sample.y << std::endl;
    std::cout << "Network Prediction: " << p << std::endl;
    std::cout << (p == sample.y ? "✅ Correct!" : "❌ Wrong!") << std::endl;
}

int main() {
    try {
        std::string model_file = "mnist_model_weights";
        std::vector<int> architecture = {784, 30, 10};
        Network net(architecture);

        // 尝试加载现有模型
        std::ifstream f(model_file + ".bin");
        if (f.good()) {
            f.close();
            std::cout << "Found existing model. Load it? (y/n): ";
            char choice; std::cin >> choice;
            if (choice == 'y') net.load(model_file);
        }

        std::cout << "Loading MNIST data..." << std::endl;
        auto train_data = MNISTReader::load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        auto test_data = MNISTReader::load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

        std::cout << "\nChoose action: \n1. Train new model\n2. Test on sample\n3. Evaluate Accuracy\nChoice: ";
        int action; std::cin >> action;

        if (action == 1) {
            std::cout << "Training with ReLU..." << std::endl;
            net.SGD(train_data, 10, 10, 0.01, &test_data); // 跑5个Epoch试试
            net.save(model_file);
        } else if (action == 2) {
            run_inference(net, test_data);
        } else {
            int correct = net.evaluate(test_data);
            std::cout << "Accuracy: " << (double)correct/test_data.size() * 100 << "%" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }
    return 0;
}