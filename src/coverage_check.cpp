#include "coverage_check.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<int> discretize_feature(const std::vector<float>& data, float rezkost) {
    if (data.empty()) return {};

    // 1. Находим min/max, игнорируя NaN
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < data.size(); i++) {
        if (!std::isnan(data[i])) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
    }

    // 2. Если все значения NaN
    if (min_val == std::numeric_limits<float>::max()) {
        return std::vector<int>(data.size(), -1);
    }

    // 3. Вычисляем интервалы
    int n_intervals = static_cast<int>(std::round(2.0f / rezkost));
    if (n_intervals < 2) n_intervals = 2;
    
    float step = (max_val - min_val) / n_intervals;
    if (step < 1e-10f) step = 1.0f;

    // 4. Дискретизация: NaN → -1, числа → 0, 1, 2...
    std::vector<int> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        if (std::isnan(data[i])) {
            result[i] = -1;
        } else {
            int idx = static_cast<int>((data[i] - min_val) / step);
            idx = std::max(0, std::min(idx, n_intervals - 1));
            result[i] = idx;
        }
    }
    return result;
}

FeatureReport check_feature_coverage(
    const std::vector<float>& raw_data,
    int feature_idx,
    float current_sharpness,
    int min_required) {

    FeatureReport report;
    report.feature_idx = feature_idx;
    report.sharpness = current_sharpness;
    report.ok_count = 0;
    report.warning_count = 0;

    auto binned = discretize_feature(raw_data, current_sharpness);
    std::unordered_map<int, int> counts;
    for (int val : binned) counts[val]++;

    report.n_intervals = counts.size();

    for (const auto& pair : counts) {
        IntervalInfo info;
        info.interval = pair.first;
        info.count = pair.second;
        info.ok = (info.count >= min_required);

        if (info.ok) report.ok_count++;
        else report.warning_count++;

        report.intervals.push_back(info);
    }

    std::sort(report.intervals.begin(), report.intervals.end(),
        [](const IntervalInfo& a, const IntervalInfo& b) {
            return a.interval < b.interval;
        });

    if (report.warning_count == 0) {
        report.recommended_sharpness = current_sharpness;
    } else {
        int new_n_intervals = report.n_intervals - report.warning_count + 1;
        new_n_intervals = std::max(2, new_n_intervals);
        report.recommended_sharpness = 2.0f / new_n_intervals;
    }

    return report;
}

CoverageReport check_data_coverage(
    const std::vector<std::vector<float>>& data,
    float sharpness,
    int min_per_interval,
    const std::vector<int>& feature_indices) {

    CoverageReport report;
    report.n_samples = data.size();
    report.min_required = min_per_interval;

    std::vector<int> indices = feature_indices;
    if (indices.empty()) {
        for (size_t i = 0; i < data[0].size(); i++) indices.push_back(i);
    }

    for (int idx : indices) {
        std::vector<float> feature_data;
        for (const auto& row : data) feature_data.push_back(row[idx]);

        auto feat_report = check_feature_coverage(
            feature_data, idx, sharpness, min_per_interval);
        report.features.push_back(feat_report);
    }

    return report;
}
