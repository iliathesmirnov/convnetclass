#include "softmax.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

void Softmax_class::find_max() {
    max = -INFTY;
    for (int s = 0; s < NUM_SHAPES; s++) {
        if (prob[s] > max) {
            max = prob[s];
            type = static_cast<Shape_type>(s);
        }
    }
}

void Softmax::train (const int NUM_EPOCHS, std::default_random_engine& gen) {
    Labeled_shape cur_shape;
    Softmax_class sc;
    int w, dirac_delta;
    float cur_dp_ratio;
    float train_acc, test_acc;
    std::string train_rec = "", test_rec = "";
    std::ofstream to ("softmax-none14.txt");

    for (int ep = 1; ep <= NUM_EPOCHS; ep++) {
        std::shuffle(given_set->data.begin(), given_set->data.end(), gen);
        for (int n = 0; n < given_set->data.size(); n += MINIBATCH_SIZE) {
            for (int w = 0; w < NUM_SHAPES * (INPUT_PARAMS+1); w++)
                dtheta[w] = 0.0;

            for (int m = 0; m < MINIBATCH_SIZE; m++) {
                cur_shape = given_set->data[n+m];
                compute_dot_products(cur_shape);
                for (int s = 0; s < NUM_SHAPES; s++) {
                    dirac_delta = cur_shape.type == static_cast<Shape_type>(s) ? 1 : 0;
                    cur_dp_ratio = dirac_delta - dot_products[s]/dot_product_sum;
                    for (int p = 0; p < INPUT_PARAMS; p++) {
                        w = p + s*(INPUT_PARAMS+1);
                        dtheta[w] += cur_dp_ratio * cur_shape.shape->bitmap[p];
                    }
                    w = INPUT_PARAMS + s*(INPUT_PARAMS+1);
                    dtheta[w] += cur_dp_ratio;
                }
            }
            for (int w = 0; w < NUM_SHAPES * (INPUT_PARAMS+1); w++)
                theta[w] += ALPHA * (dtheta[w] - LAMBDA * theta[w]);
        }
        train_acc = find_acc(TRAIN); test_acc = find_acc(TEST);

        std::cout << "EPOCH: " << ep << std::endl
                  << "TRAIN: " << train_acc << std::endl
                  << "TEST:  " << test_acc << std::endl
                  << "===================================" << std::endl;

        train_rec += std::to_string(ep) + " " + std::to_string(train_acc) + ",";
        test_rec  += std::to_string(ep) + " " + std::to_string(test_acc)  + ",";
    }
    to << train_rec + ";" + test_rec + ";!";
}

void Softmax::compute_dot_products(Labeled_shape& s) {
    Shape* shape = s.shape;
    for (int c = 0; c < NUM_SHAPES; c++) {
        dot_products[c] = 0.0;
        for (int p = 0; p < INPUT_PARAMS; p++)
            dot_products[c] += theta[c * (INPUT_PARAMS+1) + p] * shape->bitmap[p];
        dot_products[c] += theta[c*(INPUT_PARAMS+1) + INPUT_PARAMS];     // bias term
    }
    float max = -INFTY;
    for (int c = 0; c < NUM_SHAPES; c++) if (dot_products[c] > max) max = dot_products[c];
    dot_product_sum = 0.0;
    for (int c = 0; c < NUM_SHAPES; c++) {
        dot_products[c] -= max;
        dot_products[c] = exp(dot_products[c]);
        dot_product_sum += dot_products[c];
    }
}

float Softmax::compute_loss() {
    dot_product_sum = 0.0;
    float loss = 0.0, term;
    for (int i = 0; i < GIVEN_SET_SIZE; i++) {
        compute_dot_products(given_set->data[i]);
        dot_product_sum = 0.0;
        for (int c = 0; c < NUM_SHAPES; c++) dot_product_sum += dot_products[c];
        loss -= log(dot_products[given_set->data[i].type] / dot_product_sum);
    }
    return loss/GIVEN_SET_SIZE;
}

float Softmax::find_acc(Dataset_type D_TYPE) {
    int num_correct = 0; Softmax_class sc;
    Data_set* data_set_ptr;
    int size;
    switch(D_TYPE) {
        case TRAIN    :    size = GIVEN_SET_SIZE;
                           data_set_ptr = given_set;
                           break;
        case TEST     :    size = TESTING_SET_SIZE;
                           data_set_ptr = validation_set;
                           break;
    }
    for (int i = 0; i < size; i++) {
        sc = classify(data_set_ptr->data[i]);
        if ( data_set_ptr->data[i].type == sc.type ) num_correct++;
    }
    return (float) num_correct/size;
}

Softmax_class Softmax::classify(Labeled_shape& s) {
    Softmax_class sc;
    compute_dot_products(s);
    for (int c = 0; c < NUM_SHAPES; c++) sc.prob[c] = dot_products[c] / dot_product_sum;
    sc.find_max();
    return sc;
}

void Softmax::visualize_params(std::ofstream& to, Shape_type SHAPE) {
    int shape_num = static_cast<int>(SHAPE);
    int offset = shape_num * (INPUT_PARAMS+1);
    float min = INFTY, max = -INFTY;
    for (int i = 0; i < INPUT_PARAMS; i++) {
        min = std::min(min, theta[offset+i]);
        max = std::max(max, theta[offset+i]);
    }
    float r_i = 0, r_f = 255,
          g_i = 0, g_f = 255,
          b_i = 0, b_f = 255;
    float r, g, b;
    std::string red_string = "", green_string = "", blue_string = "";
    float t, param;
    for (int i = 0; i < INPUT_PARAMS; i++) {
        param = theta[offset+i];
        t = (param-min)/(max-min);
        r = std::floor((1-t)*r_i + t*r_f);
        g = std::floor((1-t)*g_i + t*g_f);
        b = std::floor((1-t)*b_i + t*b_f);
        red_string += std::to_string(r) + ",";
        blue_string += std::to_string(b) + ",";
        green_string += std::to_string(g) + ",";
    }
    to << red_string + ";"
       << green_string + ";"
       << blue_string + ";!";
}

Softmax::Softmax( int GIVEN_SET_SIZE,  int TESTING_SET_SIZE,  std::default_random_engine& gen,
                  int MINIBATCH_SIZE,  int INPUT_PARAMS,  float ALPHA,  float LAMBDA,
                  Deformation_level def_lvl ) {
    this->GIVEN_SET_SIZE = GIVEN_SET_SIZE;
    this->TESTING_SET_SIZE = TESTING_SET_SIZE;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->ALPHA = (float) ALPHA / MINIBATCH_SIZE;
    this->LAMBDA = LAMBDA;
    this->INPUT_PARAMS = INPUT_PARAMS;
    this->def_lvl = def_lvl;

    given_set      = new Given_set(GIVEN_SET_SIZE, def_lvl, gen);
    validation_set = new Testing_set(TESTING_SET_SIZE, def_lvl, gen);
    theta  = new float[NUM_SHAPES * (INPUT_PARAMS+1)];    // +1 for the bias
    dtheta = new float[NUM_SHAPES * (INPUT_PARAMS+1)];
    dot_products = new float[NUM_SHAPES];

    given_set->gen_bitmap();
    validation_set->gen_bitmap();

    for (int i = 0; i < NUM_SHAPES * (INPUT_PARAMS+1); i++) theta[i] = theta_dist(gen);
}

Softmax::~Softmax() {
    delete given_set; delete validation_set;
    delete[] theta; delete[] dtheta;
    delete[] dot_products;
    std::cout << "Softmax cleaned up" << std::endl;
}
