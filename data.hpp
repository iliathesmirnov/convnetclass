#include <random>
#include "shapes.hpp"

#ifndef DATA_H
#define DATA_H

struct Labeled_shape {
    Shape* shape;
    Shape_type type;

    void print_to_file(std::ofstream& to, bool bbox = true);
    Labeled_shape(Shape* shape = 0, Shape_type type = TRIANGLE);
};

enum Deformation_level {LOW, MED, HIGH, NONE};
enum Dataset_type {TRAIN, VALID, TEST};

class Data_set {
friend class kNN; friend class Softmax; friend class NN_trainer;
friend class GPU_Data_set;
private:
    std::normal_distribution<float> diag_low = std::normal_distribution<float>(1, 0.1);
    std::normal_distribution<float> off_diag_low = std::normal_distribution<float>(0, 0.1);
    std::normal_distribution<float> diag_med = std::normal_distribution<float>(1, 0.3);
    std::normal_distribution<float> off_diag_med = std::normal_distribution<float>(0, 1.0);
    std::normal_distribution<float> diag_high = std::normal_distribution<float>(0, 1);
    std::normal_distribution<float> off_diag_high = std::normal_distribution<float>(0, 6.28);
    std::uniform_real_distribution<float> diag_def = std::uniform_real_distribution<float>(-1.2, 1.2);
    std::uniform_real_distribution<float> off_diag_def = std::uniform_real_distribution<float>(-3.14, 3.14);
    std::uniform_real_distribution<float> scale = std::uniform_real_distribution<float>(0.25, 1.25);
    int REFLECTION_PROB = 10;
protected:
    int SIZE; Deformation_level DEF_LVL;
    std::vector<Labeled_shape> data;
    void transform_shape(Shape* model_shape, Shape* transf_shape, std::default_random_engine& gen);
    std::vector<Labeled_shape*> return_pointers();
public:
    void gen_bitmap();
    void normalize(float mean = 0.0, float stdev = 1.0);
    void print(std::ofstream& to, int from, int till, bool bbox = true);
    void print_sample(std::ofstream& to, int num);
    Data_set(int SIZE = 0, Deformation_level DEF_LVL = NONE);
    ~Data_set();
};

class Given_set : public Data_set {
public:
    Given_set(int SIZE, Deformation_level DEF_LVL,
              std::default_random_engine& gen);
};

class Testing_set : public Data_set {
public:
    Testing_set(int SIZE, Deformation_level DEF_LVL,
                std::default_random_engine& gen);
};

class Given_set_with_rounding : public Data_set {
friend class Given_set_expanded;
public:
    Given_set_with_rounding (int SIZE, Deformation_level DEF_LVL,
                             std::default_random_engine& gen);
};

//class Testing_set_with_rounding : public Data_set {
//public:
//    Testing_set_with_rounding (int size, Deformation_level DEF_LVL,
//                               std::default_random_engine& gen);
//}

class Given_set_expanded : public Data_set {
public:
    Given_set_with_rounding *gset_low, *gset_med;
    Given_set_expanded (int SIZE, std::default_random_engine& gen);
};

// ====================================== GPU Data Set ==============================================

struct GPU_Labeled_shape {
    float* bitmap;
    Shape_type* type; Shape_type CPU_type;

    GPU_Labeled_shape (float* bitmap = 0, Shape_type* type = 0);
};

class GPU_Data_set {
    friend class NN_trainer;
protected:
    int SIZE; Deformation_level DEF_LVL;
    std::vector<GPU_Labeled_shape> data;
    void copy_from_host(Data_set& data_dev);
public:
    GPU_Data_set (int SIZE = 0, Deformation_level DEF_LVL = NONE);
    ~GPU_Data_set ();
};

class GPU_Given_set : public GPU_Data_set {
public:
    GPU_Given_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen);
};

class GPU_Testing_set : public GPU_Data_set {
public:
    GPU_Testing_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen);
};

class GPU_Given_set_with_rounding : public GPU_Data_set {
public:
    GPU_Given_set_with_rounding (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen);
};

class GPU_Given_set_expanded : public GPU_Data_set {
public:
    GPU_Given_set_expanded (int SIZE, std::default_random_engine& gen);
};

#endif /* DATA_H */
