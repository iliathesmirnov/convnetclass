#include "data.hpp"
#include <random>
#include <fstream>

class Softmax_class {
public:
    float prob[NUM_SHAPES];
    float max; void find_max();
    Shape_type type;
};

class Softmax{
private:
    int GIVEN_SET_SIZE, TESTING_SET_SIZE, MINIBATCH_SIZE, INPUT_PARAMS;
    float ALPHA, LAMBDA;
    Deformation_level def_lvl;
    Given_set* given_set;
    Testing_set* validation_set;
    Testing_set* testing_set;
    std::normal_distribution<float> theta_dist = std::normal_distribution<float>(0.1, 0.01);
    float *theta, *dtheta, *dot_products;
    float dot_product_sum;
    void compute_dot_products(Labeled_shape& s);
    float compute_loss(), find_acc(Dataset_type D_TYPE);
public:
    void train(const int NUM_EPOCHS, std::default_random_engine& gen);
    Softmax_class classify(Labeled_shape& s);
    void visualize_params(std::ofstream& to, Shape_type SHAPE);
    Softmax (int GIVEN_SET_SIZE, int TESTING_SET_SIZE, std::default_random_engine& gen,
                 int MINIBATCH_SIZE=1, int INPUT_PARAMS=128*128, float ALPHA=0.01, float LAMBDA=0.05,
                 Deformation_level def_lvl=LOW);
    ~Softmax();
};
