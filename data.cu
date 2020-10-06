#include "data.hpp"
#include <fstream>
#include <iostream>

void Labeled_shape::print_to_file(std::ofstream& to, bool bbox) {
    shape->print_to_file(to, bbox);
    //to << name_by_shape_type(type) << "!";
    to << type << ";";
}

Labeled_shape::Labeled_shape(Shape* shape, Shape_type type) {
    this->shape = shape; this->type = type;
}

// ======================= Data set ==========================
void Data_set::transform_shape(Shape* model_shape, Shape* transf_shape, std::default_random_engine& gen) {
    Affine_map T, S;
    Affine_map R1 (1, 0, 0, -1, 0, 0);
    Affine_map R2 (-1, 0, 0, 1, 0, 0);
    float sc; int p;

    bool generated = false;
    while (!generated) {
        sc = scale(gen);
        S = Affine_map (sc, 0, 0, sc, 0, 0);
        switch (DEF_LVL) {
            case LOW  :    T = Affine_map (diag_low(gen), off_diag_low(gen), off_diag_low(gen), diag_low(gen), 0, 0);
                           break;
            case MED  :    T = Affine_map (diag_med(gen), off_diag_med(gen), off_diag_med(gen), diag_med(gen), 0, 0);
                           break;
            case HIGH :    T = Affine_map (diag_high(gen), off_diag_high(gen), off_diag_high(gen), diag_high(gen), 0, 0);
                           break;
            case NONE :    T = Affine_map (1, 0, 0, 1, 0, 0);
                           break;
            default   :    T = Affine_map (diag_def(gen), off_diag_def(gen), off_diag_def(gen), diag_def(gen), 0, 0);
                           break;
        }
        if ( (T.det < -0.5) || (T.det > 0.5) ) {      // Check that T does not squish shapes too much
            transf_shape->copy(model_shape);
            try {
                transf_shape->transform(T);
                if (DEF_LVL != NONE) {
                    p = rand() % REFLECTION_PROB;
                    if (p == 0) transf_shape->transform(R1);
                    if (p == 1) transf_shape->transform(R2);
                    transf_shape->transform(S);
                    T = transf_shape->gen_translation();
                    transf_shape->transform(T);
                }
                generated = true;
            } catch (Generr) {}
        }
    }
}

void Data_set::gen_bitmap () {
    for (std::vector<Labeled_shape>::iterator it = data.begin(); it != data.end(); it++)
        (*it).shape->gen_bitmap();
    std::cout << "Bitmaps generated" << std::endl;
}

std::vector<Labeled_shape*> Data_set::return_pointers() {
    std::vector<Labeled_shape*> pointers;
    for (std::vector<Labeled_shape>::iterator it = data.begin(); it != data.end(); it++)
        pointers.push_back(&(*it));
    return pointers;
}

void Data_set::print(std::ofstream& to, int from, int till, bool bbox) {
    std::vector<Labeled_shape>::iterator it;
    for (int iter = from; iter < till; iter++) {
        it = data.begin()+iter;
        (*it).print_to_file(to, bbox);
    }
    std::cout << "Printed to file" << std::endl;
}

void Data_set::print_sample(std::ofstream& to, int num) {
    int n = 0, index;
    while (n < num) {
        index = rand() % (SIZE-n);
        data[index].print_to_file(to);
        data.erase(data.begin()+index);
        n++;
    }
}

Data_set::Data_set(int SIZE, Deformation_level DEF_LVL) {
    this->SIZE = SIZE; this->DEF_LVL = DEF_LVL;
}

Data_set::~Data_set() {
    std::cout << "Data set (CPU) deletion called" << std::endl;
    for (std::vector<Labeled_shape>::iterator it = data.begin(); it != data.end(); it++)
        delete (*it).shape;
}

// ======================= Given set ==========================
Given_set::Given_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen)
       : Data_set (SIZE, DEF_LVL) {
    const int NUM_SAMPLES_PER_SHAPE = ceil((float) SIZE / NUM_SHAPES);

    Shape_type type;
    Shape *transf_shape, *model_shape;

    for (int s = 0; s < NUM_SHAPES; s++) {
        type = static_cast<Shape_type>(s);
        model_shape = allocate_mem_by_shape_type(type);
        for (int n = 0; n < NUM_SAMPLES_PER_SHAPE; n++) {
            transf_shape = allocate_mem_by_shape_type(type);
            transform_shape(model_shape, transf_shape, gen);
            data.push_back(Labeled_shape(transf_shape, type));
        }
        delete model_shape;
    }
    std::cout << "Given data set (CPU) generated" << std::endl;
}

// ======================= Testing set ==========================
Testing_set::Testing_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen)
        : Data_set (SIZE, DEF_LVL) {
    Shape_type type;
    int n;
    Shape *testing_shape, *model_shape;

    std::uniform_int_distribution<> test_sample(0, NUM_SHAPES-1);

    for (int i = 0; i < SIZE; i++) {
        n = test_sample(gen);
        type = static_cast<Shape_type>(n);
        model_shape = allocate_mem_by_shape_type(type);
        testing_shape = allocate_mem_by_shape_type(type);
        transform_shape(model_shape, testing_shape, gen);
        data.push_back(Labeled_shape(testing_shape, type));
        delete model_shape;
    }
    std::cout << "Testing set (CPU) generated" << std::endl;
}

// ======================= Given set (with rounding) ==========================
Given_set_with_rounding::Given_set_with_rounding (int SIZE, Deformation_level DEF_LVL,
                                                  std::default_random_engine& gen)
       : Data_set (SIZE, DEF_LVL) {
    const int NUM_R = 4;
    const int NUM_SAMPLES_PER_SHAPE = ceil((float) SIZE / NUM_SHAPES);

    Shape_type type;
    Shape *model_shape, *transf_shape, *rounded_shape;

    int triangle_r[NUM_R]   = {1,3,5,7};
    int square_r[NUM_R]     = {1,3,5,7};
    int pentagon_r[NUM_R]   = {1,3,5,7};
    int hexagon_r[NUM_R]    = {1,3,5,7};
    int circle_r[NUM_R]     = {1,1,1,2};
    int circleii_r[NUM_R]   = {1,1,1,2};
    int circleiii_r[NUM_R]  = {1,1,1,2};
    int circleiv_r[NUM_R]   = {1,1,1,2};
    int rhombus_r[NUM_R]    = {1,2,3,4};
    int rhombusii_r[NUM_R]  = {1,1,2,3};
    int rhombusiii_r[NUM_R] = {1,1,2,3};
    int rhombusiv_r[NUM_R]  = {1,2,3,5};
    int heart_r[NUM_R]      = {1,2,3,4};
    int diamond_r[NUM_R]    = {1,2,3,4};
    int club_r[NUM_R]       = {1,2,3,4};
    int spade_r[NUM_R]      = {1,2,3,4};
    int* radii[NUM_SHAPES] = {triangle_r, square_r, pentagon_r, hexagon_r,
                              circle_r, circleii_r, circleiii_r, circleiv_r,
                              rhombus_r, rhombusii_r, rhombusiii_r, rhombusiv_r,
                              heart_r, diamond_r, club_r, spade_r};

    int dice;

    for (int s = 0; s < NUM_SHAPES; s++) {
        type = static_cast<Shape_type>(s);
        model_shape = allocate_mem_by_shape_type(type);
        for (int n = 0; n < NUM_SAMPLES_PER_SHAPE; n++) {
            transf_shape = allocate_mem_by_shape_type(type);
            transform_shape(model_shape, transf_shape, gen);
            transf_shape->gen_bitmap();
            dice = rand() % NUM_R;
            transf_shape->rounden(radii[s][dice]);
            data.push_back(Labeled_shape(transf_shape, type));

/*            for (int r = 0; r < NUM_R; r++) {
                rounded_shape = allocate_mem_by_shape_type(type);
                rounded_shape->gen_bitmap();
                for (int i = 0; i < INPUT_DIM*INPUT_DIM; i++) {
                    rounded_shape->bitmap[i] = transf_shape->bitmap[i];
                }
                rounded_shape->rounden(radii[s][r]);
                data.push_back(Labeled_shape(rounded_shape, type));
            }
            delete transf_shape; */
        }
        delete model_shape;
    }
    std::cout << "Given data set (with rounding) (CPU) generated" << std::endl;
}
/*
// ======================= Testing set ==========================
Testing_set::Testing_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen)
        : Data_set (SIZE, DEF_LVL) {
    Shape_type type;
    int n;
    Shape *testing_shape, *model_shape;

    std::uniform_int_distribution<> test_sample(0, NUM_SHAPES-1);

    for (int i = 0; i < SIZE; i++) {
        n = test_sample(gen);
        type = static_cast<Shape_type>(n);
        model_shape = allocate_mem_by_shape_type(type);
        testing_shape = allocate_mem_by_shape_type(type);
        transform_shape(model_shape, testing_shape, gen);
        data.push_back(Labeled_shape(testing_shape, type));
        delete model_shape;
    }
    std::cout << "Testing set (CPU) generated" << std::endl;
} */

// =============== GPU Given set with rounding =====================

Given_set_expanded::Given_set_expanded (int SIZE, std::default_random_engine& gen)
    : Data_set (0, NONE) {
    std::cout << "Generating expanded Given set (CPU)" << std::endl;
    gset_low = new Given_set_with_rounding(SIZE/2, LOW, gen);
    this->data.insert(std::end(this->data), std::begin(gset_low->data), std::end(gset_low->data));
    gset_med = new Given_set_with_rounding(SIZE/2, MED, gen);
    this->data.insert(std::end(this->data), std::begin(gset_med->data), std::end(gset_med->data));
    std::cout << "Expanded Given set (CPU) generated" << std::endl;
}

// ====================================================================================================
// ======================================== GPU Functions =============================================

// Copy bitmaps from data file generated by CPU for the moment

GPU_Labeled_shape::GPU_Labeled_shape (float* bitmap, Shape_type* type) {
    this->bitmap = bitmap; this->type = type;
}

// ======================= GPU Data set ==========================

GPU_Data_set::GPU_Data_set (int SIZE, Deformation_level DEF_LVL) {
    this->SIZE = SIZE; this->DEF_LVL = DEF_LVL;
}

GPU_Data_set::~GPU_Data_set () {
    std::cout << "Data set deletion (GPU) called" << std::endl;
    for (std::vector<GPU_Labeled_shape>::iterator it = data.begin(); it != data.end(); it++) {
        cudaFree((*it).bitmap);
        cudaFree((*it).type);
    }
}

void GPU_Data_set::copy_from_host (Data_set& data_host) {
    data = std::vector<GPU_Labeled_shape>(data_host.data.size());
    std::vector<GPU_Labeled_shape>::iterator it_dev = data.begin();
    for (std::vector<Labeled_shape>::iterator it_host = data_host.data.begin();
         it_host != data_host.data.end(); it_host++) {
        cudaMalloc((void**) &((*it_dev).bitmap), INPUT_DIM * INPUT_DIM * sizeof(float));
        cudaMemcpy((*it_dev).bitmap, (*it_host).shape->bitmap, INPUT_DIM * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &((*it_dev).type), sizeof(Shape_type));
        cudaMemcpy((*it_dev).type, &(*it_host).type, sizeof(Shape_type), cudaMemcpyHostToDevice);
        (*it_dev).CPU_type = (*it_host).type;
        it_dev++;
    }
}

// ====================== GPU Given set ==========================

GPU_Given_set::GPU_Given_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen)
         : GPU_Data_set (SIZE, DEF_LVL) {
    Given_set data_host(SIZE, DEF_LVL, gen);
    data_host.gen_bitmap();
    copy_from_host(data_host);
    std::cout << "Given set (GPU) generated" << std::endl;
}

// ====================== GPU Testing set ==========================

GPU_Testing_set::GPU_Testing_set (int SIZE, Deformation_level DEF_LVL, std::default_random_engine& gen)
         : GPU_Data_set (SIZE, DEF_LVL) {
    Testing_set data_host(SIZE, DEF_LVL, gen);
    data_host.gen_bitmap();
    copy_from_host(data_host);
    std::cout << "Testing set (GPU) generated" << std::endl;
}

// =============== GPU Given set with rounding =====================

GPU_Given_set_with_rounding::GPU_Given_set_with_rounding (int SIZE, Deformation_level DEF_LVL,
                                                          std::default_random_engine& gen)
        : GPU_Data_set (SIZE, DEF_LVL) {
    Given_set_with_rounding data_host(SIZE, DEF_LVL, gen);
    copy_from_host(data_host);
    std::cout << "Given set with rounding (GPU) generated" << std::endl;
}

// =============== GPU Given set expanded =====================

GPU_Given_set_expanded::GPU_Given_set_expanded (int SIZE, std::default_random_engine& gen)
        : GPU_Data_set (SIZE, NONE) {
    Given_set_expanded data_host(SIZE, gen);
    copy_from_host(data_host);
    std::cout << "Given set with rounding (GPU) generated" << std::endl;
}


