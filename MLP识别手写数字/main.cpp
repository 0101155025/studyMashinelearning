#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "MNISTReader.h"
#include "network.h"

// 打印数字图像函数保持不变...
void print_digit(const Matrix<double>& x) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            double val = x(i * 28 + j, 0);
            if (val < 0.1) std::cout << "  ";
            else if (val < 0.5) std::cout << "..";
            else if (val < 0.8) std::cout << "++";
            else std::cout << "##";
        }
        std::cout << "\n";
    }
}

// 推理模式保持不变...
void run_inference(Network& net, const std::vector<Datapoint>& test_data) {
    std::cout << "\n--- Inference Mode ---" << std::endl;
    std::cout << "Pick a sample index (0-9999): ";
    int idx;
    std::cin >> idx;

    if (idx < 0 || idx >= static_cast<int>(test_data.size()))
    {
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
        std::vector<int> architecture;

        // --- 1. 动态配置网络架构 ---
        std::cout << "=== Neural Network Configuration ===" << std::endl;
        architecture.push_back(784); // 输入层固定为 784 (MNIST 像素)

        int num_hidden_layers;
        std::cout << "Enter the number of HIDDEN layers: ";
        std::cin >> num_hidden_layers;

        for (int i = 0; i < num_hidden_layers; ++i) {
            int nodes;
            std::cout << "Enter number of nodes for hidden layer " << i + 1 << ": ";
            std::cin >> nodes;
            architecture.push_back(nodes);
        }

        architecture.push_back(10); // 输出层固定为 10 (0-9 数字)

        // 初始化网络
        Network net(architecture);

        // --- 2. 加载模型逻辑 ---
        std::ifstream f(model_file + ".bin");
        if (f.good()) {
            f.close();
            std::cout << "\nFound existing model file. Load it? (y/n): ";
            char load_choice; std::cin >> load_choice;
            if (load_choice == 'y') {
                net.load(model_file);
                // 注意：如果加载的模型文件与你刚才输入的架构不匹配，Network::load 内部会抛出异常或覆盖架构
            }
        }

        // --- 3. 数据加载 ---
        std::cout << "\nLoading MNIST data..." << std::endl;
        auto train_data = MNISTReader::load("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        auto test_data = MNISTReader::load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

        // --- 4. 操作选择 ---
        while (true) {
            std::cout << "\nChoose action: \n1. Train Model\n2. Test Single Sample\n3. Evaluate Total Accuracy\n4. Exit\nChoice: ";
            int action; std::cin >> action;

            if (action == 1) {
                // --- 5. 动态设置训练参数 ---
                int epochs, batch_size;
                double eta, lambda;

                std::cout << "\n--- Training Hyperparameters ---" << std::endl;
                std::cout << "Enter number of Epochs (e.g., 10): ";
                std::cin >> epochs;
                std::cout << "Enter Mini-batch size (e.g., 10): ";
                std::cin >> batch_size;
                std::cout << "Enter Learning Rate (eta, e.g., 0.01 for ReLU): ";
                std::cin >> eta;
                std::cout << "Enter Regularization (lambda, e.g., 5.0): ";
                std::cin >> lambda;

                std::cout << "Training started..." << std::endl;
                net.SGD(train_data, epochs, batch_size, eta, lambda, &test_data);

                std::cout << "Training complete. Save model? (y/n): ";
                char save_choice; std::cin >> save_choice;
                if (save_choice == 'y') net.save(model_file);

            } else if (action == 2) {
                run_inference(net, test_data);
            } else if (action == 3) {
                int correct = net.evaluate(test_data);
                std::cout << "Accuracy on Test Set: " << static_cast<double>(correct) / static_cast<double>(test_data.size()) * 100.0 << "%" << std::endl;
            } else {
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[Fatal Error]: " << e.what() << std::endl;
    }
    return 0;
}