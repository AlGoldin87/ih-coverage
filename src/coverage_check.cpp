#include "coverage_check.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<int> discretize_feature(const std::vector<float>& data, float rezkost) {
    std::cout << "\n========== DISCRETIZATION DEBUG ==========" << std::endl;
    
    // 1. Входные параметры
    std::cout << "rezkost = " << rezkost << std::endl;
    std::cout << "data.size() = " << data.size() << std::endl;
    if (!data.empty()) {
        std::cout << "data[0..4] = ";
        for (size_t i = 0; i < std::min(data.size(), size_t(5)); i++) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
    
    if (data.empty()) return {};
    
    // 2. Min / Max
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    std::cout << "min_val = " << min_val << std::endl;
    std::cout << "max_val = " << max_val << std::endl;
    std::cout << "range = " << (max_val - min_val) << std::endl;
    
    // 3. Количество интервалов
    float n_intervals_float = 2.0f / rezkost;
    int n_intervals = static_cast<int>(std::round(n_intervals_float));
    if (n_intervals < 2) n_intervals = 2;
    std::cout << "n_intervals_float = " << n_intervals_float << std::endl;
    std::cout << "n_intervals (after round) = " << n_intervals << std::endl;
    
    // 4. Размер интервала
    float step = (max_val - min_val) / n_intervals;
    std::cout << "step = " << step << std::endl;
    
    // 5. Вычисление индексов для первых 5 значений
    std::cout << "\nFirst 5 values calculation:" << std::endl;
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); i++) {
        float val = data[i];
        float raw_idx = (val - min_val) / step;
        int idx = static_cast<int>(raw_idx);
        if (idx >= n_intervals) idx = n_intervals - 1;
        if (idx < 0) idx = 0;
        std::cout << "  val=" << val 
                  << " (val-min)=" << (val - min_val)
                  << " raw_idx=" << raw_idx
                  << " idx=" << idx << std::endl;
    }
    std::cout << "==========================================\n" << std::endl;
    
    // 6. Основной расчёт
    std::vector<int> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        int idx = static_cast<int>((data[i] - min_val) / step);
        if (idx >= n_intervals) idx = n_intervals - 1;
        if (idx < 0) idx = 0;
        result[i] = idx;
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
