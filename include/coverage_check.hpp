#ifndef COVERAGE_CHECK_HPP
#define COVERAGE_CHECK_HPP

#include <vector>
#include <unordered_map>
#include <string>

struct IntervalInfo {
    int interval;
    int count;
    bool ok;
};

struct FeatureReport {
    int feature_idx;
    float sharpness;
    int n_intervals;
    std::vector<IntervalInfo> intervals;
    int ok_count;
    int warning_count;
    float recommended_sharpness;
};

struct CoverageReport {
    int n_samples;
    int min_required;
    std::vector<FeatureReport> features;
};

// Функция 1: Дискретизация одного признака
std::vector<int> discretize_feature(const std::vector<float>& data, float rezkost);

// Функция 2: Проверка покрытия для одного признака
FeatureReport check_feature_coverage(
    const std::vector<float>& raw_data,
    int feature_idx,
    float current_sharpness,
    int min_required);

// Функция 3: Полная проверка для всех признаков
CoverageReport check_data_coverage(
    const std::vector<std::vector<float>>& data,
    float sharpness,
    int min_per_interval = 5,
    const std::vector<int>& feature_indices = {});

#endif