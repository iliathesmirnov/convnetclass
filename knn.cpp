#include "knn.hpp"
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

void kNN::find_distances () {
    int l;

    for (int i = 0; i < TESTING_SET_SIZE; i++) {
        for (int j = 0; j < GIVEN_SET_SIZE; j++) {
            l = i*GIVEN_SET_SIZE + j;
            dist[l].label = j;
            dist[l].value = testing_set->data[i].shape->find_L2_distance_from(given_set->data[j].shape);
        }
    }
    std::cout << "Distances found" << std::endl;
}

void kNN::find_nn () {
    Labeled_heap heap;
    Labeled_value min;
    Labeled_shape neighbor;
    int label_count[NUM_SHAPES],  high_count = 0;

    std::vector<Shape_type> high_count_type;
    int correct = 0;
    int coin;

    for (int t = 0; t < TESTING_SET_SIZE; t++) {
        Labeled_value cur_dist[GIVEN_SET_SIZE];
        for (int i = 0; i < GIVEN_SET_SIZE; i++) cur_dist[i] = dist[t*GIVEN_SET_SIZE + i];
        for (int i = 0; i < NUM_SHAPES; i++) label_count[i] = 0;
        high_count = 0;

        heap = Labeled_heap(cur_dist, GIVEN_SET_SIZE);

        nn->data.push_back(testing_set->data[t]);
        for (int n = 0; n < k; n++) {
            min = heap.extract_min();
            neighbor = given_set->data[min.label];
            nn->data.push_back(neighbor);
            label_count[neighbor.type]++;

            if (label_count[neighbor.type] > high_count) {
                high_count = label_count[neighbor.type];
                high_count_type.clear();
                high_count_type.push_back(neighbor.type);
            }
            else if (label_count[neighbor.type] == high_count)
                high_count_type.push_back(neighbor.type);
        }
        coin = rand() % high_count_type.size();     // If several classes have the same count,
                                                    // break tie uniformly at random
        kNN_class[t] = high_count_type[coin];
        if (testing_set->data[t].type == kNN_class[t]) correct++;
    }
    prop_correct = (float) correct/TESTING_SET_SIZE;

    std::cout << "Proportion correct identifications: " << prop_correct << std::endl;
    std::cout << "Nearest neighbours found" << std::endl;
}

void kNN::print (std::ofstream& to, int num) {
    to << GIVEN_SET_SIZE << "!"
       << TESTING_SET_SIZE << "!"
       << k << "!"
       << prop_correct << "!";
    for (int i = 0; i < num; i++) to << name_by_shape_type(kNN_class[i]) << ",";
    to << "!";
    nn->print(to, 0, (num-1)*(k+1));
}

kNN::kNN (int GIVEN_SET_SIZE,  int TESTING_SET_SIZE,  int k,
      Deformation_level def_lvl,  std::default_random_engine& gen) {
    this->GIVEN_SET_SIZE = GIVEN_SET_SIZE;
    this->TESTING_SET_SIZE = TESTING_SET_SIZE;
    this->k = k;
    this->def_lvl = def_lvl;

    given_set = new Given_set(GIVEN_SET_SIZE, def_lvl, gen);
    testing_set = new Testing_set(TESTING_SET_SIZE, def_lvl, gen);
    nn = new Data_set(TESTING_SET_SIZE * (k+1));
    dist = new Labeled_value[GIVEN_SET_SIZE * TESTING_SET_SIZE];
    kNN_class = new Shape_type[TESTING_SET_SIZE];

    given_set->gen_bitmap();
    testing_set->gen_bitmap();
    find_distances();
    find_nn();
}

kNN::~kNN () {
    delete given_set;
    delete testing_set;
    delete[] dist;
    delete[] kNN_class;
}


// =============================================================================================================

void kNN_Analyzer::plot_vary_k (const int GIVEN_SET_SIZE, const int TESTING_SET_SIZE,
                 const int k_init, const int k_step, const int k_iters,
                 Deformation_level def_lvl, const int n_samples, bool compute_stdev) {

    std::string file_name = "plot-given" + std::to_string(GIVEN_SET_SIZE)
               + "-testing" + std::to_string(TESTING_SET_SIZE)
               + "-k" + std::to_string(k_init) + "-" + std::to_string(k_step) + "-" + std::to_string(k_init + k_iters*k_step)
               + "-def" + std::to_string(def_lvl)
               + "-nsamples" + std::to_string(n_samples)
               + ".txt";
    std::ofstream to (file_name);

    float mean, stdev;
    int k = k_init;
    std::string data_minus_stdev = "", data = "", data_plus_stdev = "";

    kNN* kNN_instance;
    for (int i = 0; i < k_iters; i++) {
        mean = 0.0;  stdev = 0.0;
        for (int n = 0; n < n_samples; n++) {
            kNN_instance = new kNN(GIVEN_SET_SIZE, TESTING_SET_SIZE, k, def_lvl, gen);
            mean += kNN_instance->prop_correct;                                      // The variables mean and stdev are used for holding
            stdev += kNN_instance->prop_correct * kNN_instance->prop_correct;        // intermediate values here
            delete kNN_instance;
        }
        mean = (float) mean/n_samples;
        stdev = ((float) (stdev - n_samples*mean*mean)/(n_samples-1) );     // Variance
        stdev = sqrt(stdev);

        data_minus_stdev += (std::to_string(k) + " " + std::to_string(mean - stdev) + ",");
        data += (std::to_string(k) + " " + std::to_string(mean) + ",");
        data_plus_stdev += (std::to_string(k) + " " + std::to_string(mean + stdev) + ",");
        k += k_step;
    }

    data_minus_stdev += ";";
    data += ";";
    data_plus_stdev += ";";
    to << data_minus_stdev << data << data_plus_stdev << "!";
    to.close();
}

void kNN_Analyzer::gen_and_test_kNN (int GIVEN, const int TEST, const int k, const Deformation_level DEF_LVL,
                                     float* prop, int i) {
    kNN* kNN_instance = new kNN(GIVEN, TEST, k, DEF_LVL, gen);
    prop[i] = kNN_instance->prop_correct;
    delete kNN_instance;
}

void kNN_Analyzer::plot_vary_given_set_size (const int given_init, const int given_multiple, const int given_iters,
                                             const int TEST, const int k, const Deformation_level DEF_LVL,
                                             const int n_samples, const int n_threads, bool compute_stdev) {
    float mean, stdev;
    int given = given_init;
    std::string data_minus_stdev = "", data = "", data_plus_stdev = "";

    std::vector<std::thread> threads(n_threads);
    float* prop = (float*) malloc(n_threads * sizeof(float));

    for (int g = 0; g < given_iters; g++) {
        std::cout << "Computing kNN statistics: GIVEN = " << given
                                          << "; TEST = "  << TEST
                                          << "; k = "     << k << std::endl;
        mean = 0.0; stdev = 0.0;
        for (int n = 0; n < n_samples; n += n_threads) {
            for (int i = 0; i < n_threads; i++)
               threads[i] = std::thread( &kNN_Analyzer::gen_and_test_kNN, this, given, TEST, k, DEF_LVL, prop, i );
            for (int i = 0; i < n_threads; i++)
               threads[i].join();
            for (int i = 0; i < n_threads; i++) {
                mean  += prop[i];
                stdev += prop[i] * prop[i];
            }
        }
        mean  = (float) mean / n_samples;
        stdev = ((float) (stdev - n_samples*mean*mean)/(n_samples-1));
        stdev = sqrt(stdev);

        data_minus_stdev += std::to_string(given) + " " + std::to_string(mean - stdev) + ",";
        data             += std::to_string(given) + " " + std::to_string(mean)         + ",";
        data_plus_stdev  += std::to_string(given) + " " + std::to_string(mean + stdev) + ",";

        given *= given_multiple;
    }
    data_minus_stdev += ";"; data += ";"; data_plus_stdev += ";";

    std::string file_name = "test.txt";
    std::ofstream to (file_name);
    to << data_minus_stdev << data << data_plus_stdev << "!";
    to.close();

    free(prop);
}


kNN_Analyzer::kNN_Analyzer() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::default_random_engine(seed);
}
