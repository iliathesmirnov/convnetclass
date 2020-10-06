#include <iostream>
#include <random>
#include <chrono>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <string>
#include <thread>
#include "knn.hpp"
#include "neural.hpp"
#include "softmax.hpp"

int main(int argc, char* argv[]) {

    // Set up random number generators
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen_host(seed);
    curandGenerator_t gen;
    curandStatus_t error;
    error = curandCreateGenerator (&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (error != CURAND_STATUS_SUCCESS) std::cout << "Error initializing curandGen" << std::endl;
    error = curandSetPseudoRandomGeneratorSeed (gen, seed);
    if (error != CURAND_STATUS_SUCCESS) std::cout << "Error seeding gen" << std::endl;


    int MINIBATCH_SIZE = 128;

// =================================================================================
// ===== Feedforward net ===========================================================
/*    int HIDDEN_UNITS = 2;
    int NODES[HIDDEN_UNITS+2] = {128*128, 256, 256, 16};

    FF_net fnet(HIDDEN_UNITS, NODES, MINIBATCH_SIZE, gen);
    FF_net fnet_drop(HIDDEN_UNITS, NODES, MINIBATCH_SIZE, gen, true);

    NN_trainer nntr(65536, 1024, 1, HIGH, MINIBATCH_SIZE, gen_host);
        nntr.net = &fnet;
        nntr.ALPHA = 0.01;
        nntr.REG_TYPE = L2;
        nntr.LAMBDA = 0.0025;
    nntr.train (256, gen_host, false);
    nntr.output_to_file ();
    nntr.reset();

//    Uncomment to store weights: ==================================================
    std::ofstream to ("weights-1024-1024-rounding.txt");
    fnet.store_weights(to);
    to.close();

    Triangle t;
    t.gen_bitmap();
    fnet.classify(t.bitmap);

    Square s;
    s.gen_bitmap();
    fnet.classify(s.bitmap);

    Spade sp;
    sp.gen_bitmap();
    fnet.classify(sp.bitmap);*/



// =================================================================================
// ===== Conv net ==================================================================
    convdim input_dim(128, 128, 1);

    int CONV_BLOCKS = 3;                                       // "Large" architecture
    int WIDTH[2*CONV_BLOCKS] = {7,6,5,5,5,5};
    int STRIDE[2*CONV_BLOCKS] = {3,2,1,1,1,1};
    int FILTERS[2*CONV_BLOCKS] = {20,20,20,40,40,40};
    int PADDING[2*CONV_BLOCKS] = {1,1,3,2,3,2};
    int FF_BLOCKS = 0;
    int FF_NODES[1] = {1024};

//    int CONV_BLOCKS = 2;                                     // "Small" architecture
//    int WIDTH[2*CONV_BLOCKS] = {7,6,5,5};
//    int STRIDE[2*CONV_BLOCKS] = {3,2,1,1};
//    int FILTERS[2*CONV_BLOCKS] = {20,20,20,20};
//    int PADDING[2*CONV_BLOCKS] = {1,1,3,2};
//    int FF_BLOCKS = 0;
//    int FF_NODES[1] = {1};

// To train without dropout: =======================================================
    Conv_net cnet(CONV_BLOCKS, WIDTH, STRIDE, FILTERS, PADDING,
                  FF_BLOCKS, FF_NODES, input_dim, gen, MINIBATCH_SIZE, (float) 0.01);

    NN_trainer nntr(32768, 1024, 1, MED, MINIBATCH_SIZE, gen_host);
        nntr.net = &cnet;
        nntr.ALPHA = 0.005;
        nntr.REG_TYPE = L2;
        nntr.LAMBDA = 0.0001;
    nntr.train (128, gen_host, false);
    nntr.output_to_file ();
    nntr.reset();

//    To train with dropout: =======================================================
//    Conv_net cnet_drop(CONV_BLOCKS, WIDTH, STRIDE, FILTERS, PADDING,
//                  FF_BLOCKS, FF_NODES, input_dim, gen, MINIBATCH_SIZE, true, (float) 0.01);
//    NN_trainer nntr(32768, 1024, 1, LOW, MINIBATCH_SIZE, gen_host);
//        nntr.net = &cnet_drop;
//        nntr.ALPHA = 0.005;
//        nntr.REG_TYPE = L2;
//        nntr.LAMBDA = 0.0001;
//    nntr.train (250, gen_host, true);
//    nntr.output_to_file ();
//    nntr.reset();

//    Uncomment to store weights ======================================================
    std::ofstream to("conv-weights.txt");
    to.precision(6);
    cnet.store_weights(to);
    to.close();
/*

// =================================================================================
// ===== kNN =======================================================================
/*    kNN_Analyzer analyzer;
    analyzer.plot_vary_given_set_size (256, 2, 10, 1024, 1, HIGH, 12, 3);
    //analyzer.plot_vary_k(32768, 128, 1, 1, 20, , 10); */

/*    int data_set_size[6] = {1024, 2048, 4096, 8192, 16384, 32768};
    Deformation_level DEF_LVL;
    for (int s = 0; s < 6; s++) {
        for (int d = 0; d < 3; d++) {
            DEF_LVL = static_cast<Deformation_level>(d);
            kNN knn(data_set_size[s], 1024, 9, DEF_LVL, gen_host);
            std::ofstream to ("knn/" + std::to_string(data_set_size[s]) + "-" + std::to_string(d) + ".txt");
            knn.print(to, 64);
            to.close();
        }
    } */

// =================================================================================
// ===== Softmax / Logistic regression =============================================
/*    Softmax sftmax(32768, 1024, gen_host, 32, 128*128, 0.01, 0.0005, NONE);
    sftmax.train(512, gen_host);
    std::ofstream to;
    Shape_type shape;
    for (int s = 0; s < NUM_SHAPES; s++) {
        shape = static_cast<Shape_type>(s);
        to = std::ofstream("softmax/" + file_name_by_shape_type(shape) + "-overfit.txt");
        sftmax.visualize_params(to, shape);
        to.close();
    } */

   /* for (int i = 1; i < 6; i++) {
        std::ofstream to ("samples/med" + std::to_string(i) + ".txt");
        Given_set data_set(32768, HIGH, gen_host);
        data_set.gen_bitmap();
        data_set.print_sample(to, 256);
        to.close();
    } */


// =================================================================================
// ===== Other ======================= =============================================
/*    Given_set_with_rounding set(16384, LOW, gen_host);
    std::ofstream to("test-rounded-set.txt");
    set.print(to);
    to.close(); */

/*    const int KNN_DATASET_SIZE = 4096;
    Given_set gset (KNN_DATASET_SIZE, LOW, gen_host);
    std::ofstream to ("knn_data_set.txt");
    gset.gen_bitmap();

    gset.print(to, 0, KNN_DATASET_SIZE, false);
    to << "!";
    to.close(); */



    return 0;
}
