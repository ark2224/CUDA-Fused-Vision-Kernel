#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ImageRGBA {
    int w = 0, h = 0, c = 4;
    std::vector<uint8_t> data;
};

ImageRGBA loadrgba(const std::string& path);
void save_gray_png(const std::string& path, int w, int h, const std::vector<uint8_t>& gray);