#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define DIM 1000

struct Complex {
    float r, i;
};

float magnitude(struct Complex a) {
    return ((a.r * a.r) + (a.i * a.i));
}

void add(struct Complex a, struct Complex b, struct Complex *res) {
    res->r = a.r + b.r;
    res->i = a.i + b.i;
}

void mul(struct Complex a, struct Complex b, struct Complex *res) {
    res->r = a.r * b.r - a.i * b.i;
    res->i = a.r * b.r + a.i * b.i;
}

int julia(int x, int y, struct Complex c) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    struct Complex a, r1, r2;
    a.r = jx, a.i = jy;

    for (int i = 0; i < 1000; i++) {
        mul(a, a, &r1);
        add(r1, c, &r2);

        if (magnitude(r2) > 4.0f) {
            return 0;
        }
        a.r = r2.r;
        a.i = r2.i;
    }
    return 1;
}

void kernelFun(unsigned char *nice, struct Complex c) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            int julia_val = julia(x, y, c);
            nice[offset * 4 + 0] = 0;
            nice[offset * 4 + 1] = 0;
            nice[offset * 4 + 2] = 255 * julia_val;
            nice[offset * 4 + 3] = 255;
        }
    }
}

void saveImage(const string &filename, struct Complex c) {
    cv::Mat image(DIM, DIM, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    unsigned char *nice = image.data;
    kernelFun(nice, c);
    cv::imwrite(filename, image);
}

int main() {
    struct Complex c1, c2;
    c1.r = -0.70176, c1.i = -0.3842; // Test case 1: Connected Julia set
    c2.r = 0.285, c2.i = 0.01;     // Test case 2: Fatou dust pattern

    auto cpu_start = chrono::high_resolution_clock::now();
    saveImage("connected.png", c1);
    saveImage("fatou_dust.png", c2);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    cout << "CPU Execution Time: " << cpu_duration.count() << " seconds\n";

    return 0;
}

//For a popout window.
//cv::namedWindow("Julia Set", cv::WINDOW_NORMAL);
//cv::imshow("Julia Set", image);

//To run the Code,
//nvcc -o julia_sets julia_sets.cu `pkg-config --cflags --libs opencv4` -diag-suppress=611
