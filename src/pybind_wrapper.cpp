#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "coverage_check.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace py = pybind11;

// Вспомогательная функция для преобразования vector<vector<float>> из numpy
std::vector<std::vector<float>> numpy_to_vector(py::array_t<float> input) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];
    
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i][j] = ptr[i * cols + j];
        }
    }
    return result;
}

// Функция для определения минимального шага по данным
float detect_min_step(const std::vector<float>& data) {
    if (data.size() < 2) return 1.0f;
    
    // Сортируем и ищем минимальную разницу между соседними уникальными значениями
    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    float min_step = 1e10f;
    for (size_t i = 1; i < sorted.size(); i++) {
        float diff = sorted[i] - sorted[i-1];
        if (diff > 1e-10f && diff < min_step) {
            min_step = diff;
        }
    }
    
    // Если все значения одинаковы или разница слишком мала
    if (min_step > 1e9f) return 1.0f;
    
    return min_step;
}

// Функция для построения эталонной дискретизации с минимальным шагом
std::vector<int> discretize_reference(const std::vector<float>& data) {
    float min_step = detect_min_step(data);
    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());
    
    // Число интервалов = (max - min) / min_step
    int n_intervals = static_cast<int>(std::ceil((max_val - min_val) / min_step));
    if (n_intervals < 2) n_intervals = 2;
    
    std::vector<int> result(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        int idx = static_cast<int>((data[i] - min_val) / min_step);
        idx = std::max(0, std::min(idx, n_intervals - 1));
        result[i] = idx;
    }
    return result;
}

// Функция для одного столбца - рекомендуемая резкость (ВЕРСИЯ С ICC)
float suggest_sharpness_1d(py::array_t<float> data, float alpha = 1.0f) {
    auto buf = data.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;
    std::vector<float> vec(ptr, ptr + size);
    
    if (vec.empty()) return 1.0f;
    
    // ---- 1. Эталонная энтропия (минимальный шаг по данным) ----
    auto binned_ref = discretize_reference(vec);
    std::unordered_map<int, int> counts_ref;
    for (int val : binned_ref) counts_ref[val]++;
    
    float n = (float)vec.size();
    float h_sum_ref = 0.0f;
    for (const auto& pair : counts_ref) {
        float p = pair.second / n;
        if (p > 0) h_sum_ref += p * log2(p);
    }
    float H_ref = -h_sum_ref;
    
    // ---- 2. Перебор резкостей ----
    float best_icc = 1e10f;
    float best_sharpness = 1.0f;
    
    for (float s = 0.1f; s <= 1.0f + 0.001f; s += 0.05f) {
        auto binned = discretize_feature(vec, s);
        
        // H_current
        std::unordered_map<int, int> counts;
        for (int val : binned) counts[val]++;
        
        float h_sum = 0.0f;
        for (const auto& pair : counts) {
            float p = pair.second / n;
            if (p > 0) h_sum += p * log2(p);
        }
        float H_current = -h_sum;
        
        // Loss (потеря информации при огрублении относительно эталона)
        float loss = H_ref - H_current;
        
        // Penalty (штраф за ненадёжность интервалов)
        float penalty = 0.0f;
        for (const auto& pair : counts) {
            penalty += 1.0f / pair.second;
        }
        
        float icc = loss + alpha * penalty;
        
        if (icc < best_icc) {
            best_icc = icc;
            best_sharpness = s;
        }
    }
    
    return best_sharpness;
}

// Обёртка для Python
py::dict check_coverage_py(py::array_t<float> data,
                           float sharpness,
                           int min_per_interval = 5,
                           py::list feature_indices = py::list()) {

    auto cpp_data = numpy_to_vector(data);

    std::vector<int> indices;
    for (auto item : feature_indices) {
        indices.push_back(item.cast<int>());
    }

    auto report = check_data_coverage(cpp_data, sharpness, min_per_interval, indices);

    // Преобразуем отчёт в Python-словарь
    py::dict result;
    result["n_samples"] = report.n_samples;
    result["min_required"] = report.min_required;

    py::list features;
    float max_recommended = 0.0f;

    for (size_t i = 0; i < report.features.size(); i++) {
        const auto& feat = report.features[i];
        py::dict fdict;
        fdict["feature"] = feat.feature_idx;
        fdict["sharpness"] = feat.sharpness;
        fdict["n_intervals"] = feat.n_intervals;
        fdict["ok_count"] = feat.ok_count;
        fdict["warning_count"] = feat.warning_count;
        fdict["recommended_sharpness"] = feat.recommended_sharpness;

        if (feat.warning_count > 0) {
            max_recommended = std::max(max_recommended, feat.recommended_sharpness);
        }
        
        py::list intervals;
        for (size_t j = 0; j < feat.intervals.size(); j++) {
            const auto& iv = feat.intervals[j];
            py::dict idict;
            idict["interval"] = iv.interval;
            idict["count"] = iv.count;
            idict["ok"] = iv.ok;
            intervals.append(idict);
        }
        fdict["intervals"] = intervals;

        features.append(fdict);
    }

    result["features"] = features;
    result["recommended_sharpness"] = max_recommended;

    return result;
}

// Модуль
PYBIND11_MODULE(ih_coverage, m) {
    m.doc() = "Coverage check and sharpness optimization for IH library";

    m.def("check_coverage", &check_coverage_py,
          py::arg("data"),
          py::arg("sharpness"),
          py::arg("min_per_interval") = 5,
          py::arg("feature_indices") = py::list(),
          "Check coverage of intervals for given features\n\n"
          "Args:\n"
          "    data: 2D numpy array (rows x cols)\n"
          "    sharpness: current sharpness value\n"
          "    min_per_interval: minimum observations per interval (default 5)\n"
          "    feature_indices: list of feature indices to check (default all)\n\n"
          "Returns:\n"
          "    dict with coverage report and recommendations");

    // ========== ИСПРАВЛЕННАЯ ФУНКЦИЯ DISCRETIZE ==========
    m.def("discretize", [](py::array_t<float> data, float sharpness) {
        auto buf = data.request();
        float* ptr = static_cast<float*>(buf.ptr);
        size_t rows = buf.shape[0];
        size_t cols = buf.shape[1];
        
        // Результат: rows × cols
        std::vector<std::vector<int>> result(rows, std::vector<int>(cols));
        
        // Дискретизируем каждый столбец
        for (size_t j = 0; j < cols; j++) {
            // Собираем столбец j
            std::vector<float> column(rows);
            for (size_t i = 0; i < rows; i++) {
                column[i] = ptr[i * cols + j];
            }
            
            // Дискретизируем столбец
            auto binned = discretize_feature(column, sharpness);
            
            // Записываем сразу в результат
            for (size_t i = 0; i < rows; i++) {
                result[i][j] = binned[i];
            }
        }
        
        return result;
    }, py::arg("data"), py::arg("sharpness"),
       "Discretize data using given sharpness");

    m.def("_suggest_sharpness", &suggest_sharpness_1d,
          py::arg("data"),
          
          py::arg("alpha") = 1.0f,
          "Suggest optimal sharpness using ICC criterion\n\n"
          "Args:\n"
          "    data: 1D numpy array\n"
          
          "    alpha: penalty coefficient for interval unreliability (default 1.0)\n\n"
          "Returns:\n"
          "    optimal sharpness value");
}
