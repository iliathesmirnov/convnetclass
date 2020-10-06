#include "data.hpp"
#include "heap.hpp"
#include <random>

class kNN {
friend class kNN_Analyzer;
private:
    int GIVEN_SET_SIZE,  TESTING_SET_SIZE,  k;
    Deformation_level def_lvl;
    Given_set* given_set;
    Testing_set* testing_set;
    Data_set* nn;
    Labeled_value* dist;
    Shape_type* kNN_class;
    float prop_correct;
public:
    void find_distances();
    void find_nn();
    void print(std::ofstream& to, int num);
    kNN (int GIVEN_SET_SIZE, int TESTING_SET_SIZE, int k, Deformation_level def_lvl, std::default_random_engine& gen);
    ~kNN();
};

class kNN_Analyzer {
private:
    std::default_random_engine gen;
public:
    void gen_and_test_kNN         (const int GIVEN, const int TEST, const int k, const Deformation_level DEF_LVL,
                                   float* prop, int i);
    void plot_vary_k              (const int GIVEN_SET_SIZE, const int TESTING_SET_SIZE,
                                   const int k_init, const int k_step, const int k_iters,
                                   Deformation_level def_lvl, const int n_samples = 1, bool compute_stdev = true);
    void plot_vary_given_set_size (int given_init, int given_multiple, int given_iters,
                                   const int TEST, const int k, const Deformation_level DEF_LVL,
                                   const int n_samples, const int n_threads, bool compute_stdev = true);

    kNN_Analyzer();
};
