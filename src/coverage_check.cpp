#include "coverage_check.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

#include <fstream>
#include <chrono>
#include <ctime>

std::vector<int> discretize_feature(const std::vector<float>& data, float rezkost) {
    // Открываем лог-файл (создаётся в /tmp в Colab)
    std::ofstream log("/tmp/discretize_debug.log", std::ios::app);
    
    // 1. Временная метка и версия
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    log << "\n========== " << std::ctime(&now_c);
    log << "VERSION: 2025-03-21-final-debug\n";
    
    // 2. Входные параметры
    log << "rezkost = " << rezkost << "\n";
    log << "data.size() = " << data.size() << "\n";
    if (!data.empty()) {
        log << "data[0..4] = ";
        for (size_t i = 0; i < std::min(data.size(), size_t(5)); i++) {
            log << data[i] << " ";
        }
        log << "\n";
    }
    
    if (data.empty()) {
        log << "ERROR: data is empty\n";
        log.close();
        return {};
    }
    
    // 3. Min / Max
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    log << "min_val = " << min_val << "\n";
    log << "max_val = " << max_val << "\n";
    log << "range = " << (max_val - min_val) << "\n";
    
    // 4. Количество интервалов
    float n_intervals_float = 2.0f / rezkost;
    int n_intervals = static_cast<int>(std::round(n_intervals_float));
    if (n_intervals < 2) n_intervals = 2;
    log << "n_intervals_float = " << n_intervals_float << "\n";
    log << "n_intervals (after round) = " << n_intervals << "\n";
    
    // 5. Размер интервала
    float step = (max_val - min_val) / n_intervals;
    log << "step = " << step << "\n";
    
    // 6. Проверка на вырожденный случай
    if (step < 1e-10f) {
        log << "WARNING: step is too small, setting to 1.0\n";
        step = 1.0f;
    }
    
    // 7. Вычисление индексов для первых 5 значений (подробно)
    log << "\nFirst 5 values calculation:\n";
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); i++) {
        float val = data[i];
        float val_minus_min = val - min_val;
        float raw_idx = val_minus_min / step;
        int idx = static_cast<int>(raw_idx);
        if (idx >= n_intervals) idx = n_intervals - 1;
        if (idx < 0) idx = 0;
        log << "  val=" << val 
            << " (val-min)=" << val_minus_min
            << " / step=" << step
            << " = raw_idx=" << raw_idx
            << " -> idx=" << idx << "\n";
    }
    
    // 8. Основной расчёт (с сохранением всех индексов)
    std::vector<int> result(data.size());
    log << "\nAll indices (first 20):\n";
    for (size_t i = 0; i < data.size(); i++) {
        int idx = static_cast<int>((data[i] - min_val) / step);
        if (idx >= n_intervals) idx = n_intervals - 1;
        if (idx < 0) idx = 0;
        result[i] = idx;
        if (i < 20) log << idx << " ";
    }
    if (data.size() > 20) log << "...";
    log << "\n";
    
    log << "==========================================\n";
    log.close();
    
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
