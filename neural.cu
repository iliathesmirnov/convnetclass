#include "neural.hpp"
#include <iostream>
#include <random>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>  //CUDA Random-generation
#include <vector>
#include <algorithm>
#include <tuple>
#include <chrono>
#include <fstream>
#include <thread>
#include <unistd.h>

cudaError_t cuda_st;
void check_cuda(std::string id) {
    cudaDeviceSynchronize();
    cuda_st = cudaGetLastError();
    if (cuda_st != cudaSuccess) {
        std::cout << id << std::endl;
        usleep(2000000);
    }
}

int convdim::vol() { return w*h*d; }
convdim::convdim (int w, int h, int d) { this->w = w; this->h = h; this->d = d; }

// ==================================================================
// ======================  Ledger  ==================================
// ==================================================================

namespace Ledger_kernels {
    __global__ void init_acl(float* tpc, float* tloss, float* vpc, float* vloss);
    __global__ void init_singles_acl(float* pc, float* loss, float* best_tpc, float* best_tloss,
                                     float* best_vpc, float* best_vloss);
    __global__ void finalize_acl(float* tpc, float* tloss, float* vpc, float *vloss,
                                 float* pc, float* loss, float* best_tpc, float* best_tloss,
                                 float* best_vpc, float* best_vloss, int, int, int, int t);
    __global__ void print(int from, int to, std::string* out);
}

__global__ void Ledger_kernels::init_acl (float* tpc, float* tloss,
                                          float* vpc, float* vloss) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    tpc[n] = 0.0; tloss[n] = 0.0;
    vpc[n] = 0.0; vloss[n] = 0.0;
}

__global__ void Ledger_kernels::init_singles_acl (float* pc, float* loss,
                                                  float* best_tpc, float* best_tloss,
                                                  float* best_vpc, float* best_vloss) {
    *pc = 0.0; *loss = 0.0;
    *best_tpc = 0.0; *best_tloss = INFTY;
    *best_vpc = 0.0; *best_vloss = INFTY;
}

__global__ void Ledger_kernels::finalize_acl (float* tpc, float* tloss,
                                              float* vpc, float* vloss,
                                              float* pc, float* loss,
                                              float* best_tpc, float* best_tloss,
                                              float* best_vpc, float* best_vloss,
                                              int TRAIN, int VALID, int TEST, int t) {
    tpc[t] /= TRAIN; tloss[t] /= TRAIN;
    vpc[t] /= VALID; vloss[t] /= VALID;
    *pc /= TEST; *loss /= TEST;

    printf("EPOCH %d\nTRAIN pc = %f, loss = %f\nVALID pc = %f, loss = %f\n",
                    t, tpc[t], tloss[t], vpc[t], vloss[t]);

    if (tpc[t] > *best_tpc) *best_tpc = tpc[t];
    if (tloss[t] < *best_tloss) *best_tloss = tloss[t];
    if (vpc[t] > *best_vpc) *best_vpc = vpc[t];
    if (vloss[t] < *best_vloss) *best_vloss = vloss[t];

    printf("=======================\n");
}

void Ledger::end_cur_epoch (int t) {
    if (t == cur_item) {
        Ledger_kernels::finalize_acl <<<1,1>>> (tpc, tloss, vpc, vloss, pc, loss,
                                                best_tpc, best_tloss, best_vpc, best_vloss,
                                                TRAIN_SET_SIZE, VALID_SET_SIZE, TEST_SET_SIZE, t);
        cur_item++;
        if (cur_item == max_size) expand(max_size*2);
    }
    else
        std::cout << "Error: trainer and ledger time inconsistent" << std::endl;
}

void Ledger::expand(int new_max) {
    int old_max = this->max_size;
    this->max_size = new_max;
    std::cout << "Expanding ledger... New max: " << new_max << std::endl;

    float *old_tpc = tpc, *old_tloss = tloss,
          *old_vpc = vpc, *old_vloss = vloss;

    cudaMalloc ((void**) &tpc,   new_max * sizeof(float));
    cudaMalloc ((void**) &tloss, new_max * sizeof(float));
    cudaMalloc ((void**) &vpc,   new_max * sizeof(float));
    cudaMalloc ((void**) &vloss, new_max * sizeof(float));
    reset(); this->cur_item = old_max;

    cudaMemcpy(tpc,   old_tpc,   old_max * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(tloss, old_tloss, old_max * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vpc,   old_vpc,   old_max * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(vloss, old_vloss, old_max * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(old_tpc); cudaFree(old_tloss); cudaFree(old_vpc); cudaFree(old_vloss);
}

void Ledger::reset() {
    int t = std::min(32, max_size);
    Ledger_kernels::init_acl <<<max_size/t, t>>> (tpc, tloss, vpc, vloss);
    Ledger_kernels::init_singles_acl <<<1,1>>> (pc, loss, best_tpc, best_tloss, best_vpc, best_vloss);
    this->cur_item = 1;
}

float Ledger::get_best(Ledger_entry ent) {
    float host_copy;
    switch(ent) {
        case TPC      :    cudaMemcpy(&host_copy, best_tpc, sizeof(float), cudaMemcpyDeviceToHost);
                           return host_copy;
        case TLOSS    :    cudaMemcpy(&host_copy, best_tloss, sizeof(float), cudaMemcpyDeviceToHost);
                           return host_copy;
        case VPC      :    cudaMemcpy(&host_copy, best_vpc, sizeof(float), cudaMemcpyDeviceToHost);
                           return host_copy;
        case VLOSS    :    cudaMemcpy(&host_copy, best_vloss, sizeof(float), cudaMemcpyDeviceToHost);
                           return host_copy;
        default       :    return -1;    // Error
    }
}

std::string Ledger::print(int from, int to) {
    int span = to - from + 1;
    float host_tpc[span], host_tloss[span],
          host_vpc[span], host_vloss[span],
          host_pc,  host_loss;

    cudaMemcpy(host_tpc,   tpc   + (from-1) * sizeof(float), span * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_tloss, tloss + (from-1) * sizeof(float), span * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vpc,   vpc   + (from-1) * sizeof(float), span * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vloss, vloss + (from-1) * sizeof(float), span * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_pc,    pc,                                     sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_loss,  loss,                                   sizeof(float), cudaMemcpyDeviceToHost);

    std::string tpc_string = "", tloss_string = "",
                vpc_string = "", vloss_string = "";
    for (int i = 0; i < span; i++) {
        tpc_string   += std::to_string(from + i) + " " + std::to_string(host_tpc[i]) + ",";
        tloss_string += std::to_string(from + i) + " " + std::to_string(host_tloss[i]) + ",";
        vpc_string   += std::to_string(from + i) + " " + std::to_string(host_vpc[i]) + ",";
        vloss_string += std::to_string(from + i) + " " + std::to_string(host_vloss[i]) + ",";
    }
    return       tpc_string    + ";"
               + vpc_string    + ";"
               + tloss_string  + ";"
               + vloss_string  + ";!"
               + std::to_string(host_pc)   + ";"
               + std::to_string(host_loss) + "@";
}

Ledger::Ledger (int TRAIN, int VALID, int TEST, int max) {
    this->TRAIN_SET_SIZE = TRAIN; this->VALID_SET_SIZE = VALID; this->TEST_SET_SIZE = TEST;
    this->max_size = max; this->cur_item = 1;

    cudaMalloc((void**) &tpc,   max_size * sizeof(float));
    cudaMalloc((void**) &tloss, max_size * sizeof(float));
    cudaMalloc((void**) &vpc,   max_size * sizeof(float));
    cudaMalloc((void**) &vloss, max_size * sizeof(float));
    cudaMalloc((void**) &pc,         sizeof(float));
    cudaMalloc((void**) &loss,       sizeof(float));
    cudaMalloc((void**) &best_tpc,   sizeof(float));
    cudaMalloc((void**) &best_tloss, sizeof(float));
    cudaMalloc((void**) &best_vpc,   sizeof(float));
    cudaMalloc((void**) &best_vloss, sizeof(float));
    cudaDeviceSynchronize();
    reset();
}

Ledger::~Ledger () {
    cudaFree(tpc); cudaFree(tloss);
    cudaFree(vpc); cudaFree(vloss);
    cudaFree(pc); cudaFree(loss);
    cudaFree(best_tpc); cudaFree(best_tloss);
    cudaFree(best_vpc); cudaFree(best_vloss);
}

// ==================================================================
// ===================== Feedforward layer ==========================

const std::string FF_layer::layer_id = "Feedforward";

namespace FF_layer_kernels {
    __global__ void fw_pass(const float* input,
                            const float* weight,
                            float* output, const int PREV, const int CUR);
    __global__ void compute_delta (const float* delta_in, const float* weight,
                                   float* delta, const int PREV, const int CUR);
    __global__ void compute_dw (const float* input, const float* delta_in,
                                float* dw, const int PREV, const int CUR,
                                const int MINIBATCH_SIZE);
    __global__ void update_weights(const float ALPHA, const float LAMBDA,
                            const int MINIBATCH_SIZE, const Reg_type REG_TYPE, const int PREV,
                            float* weight, const float* dw);
    __global__ void reset_dw(const int PREV, float* dw);
}


__global__ void FF_layer_kernels::fw_pass(const float* input,
                                          const float* weight,
                                          float* output, const int PREV, const int CUR) {
    int n = blockIdx.x,    // Index of current node in minibatch
        b = threadIdx.x,   // Index of minibatch
        output_index = n + b * CUR,
        input_index, w_index;
    output[output_index] = 0.0;
    for (int m = 0; m < PREV; m++) {
        w_index = m + n * PREV;
        input_index  = m + b * PREV;
        output[output_index] += weight[w_index] * input[input_index];
    }
}

__global__ void FF_layer_kernels::compute_delta (const float* delta_in, const float* weight,
                                                 float* delta, const int PREV, const int CUR) {
    int block = blockIdx.x,
        batch = threadIdx.x,
        m = threadIdx.y,             // Position of prev. node in block
        p_node, c_node, w_index;     // Indices of previous node, current node, weight

    float d = 0.0;
    for (int n = 0; n < CUR; n++) {
        c_node = n + batch * CUR;
        w_index = m + n * PREV;
        d += delta_in[c_node] * weight[w_index];
    }
    p_node = m + block * blockDim.y + batch * PREV;
    delta[p_node] = d;
}

__global__ void FF_layer_kernels::compute_dw (const float* input, const float* delta_in,
                                              float* dw, const int PREV, const int CUR,
                                              const int MINIBATCH_SIZE) {
    int m = blockIdx.x,                               // Index of prev. node in mbatch
        n = threadIdx.x + blockDim.x * blockIdx.y,    // Index of cur. node in mbatch
        p_node, c_node, w_index;

    float change = 0.0;
    for (int batch = 0; batch < MINIBATCH_SIZE; batch++) {
        p_node = m + PREV * batch;
        c_node = n + CUR * batch;
        change += delta_in[c_node] * input[p_node];
    }
    w_index = m + n * PREV;
    dw[w_index] = change;
}

__global__ void FF_layer_kernels::update_weights (const float ALPHA, const float LAMBDA,
                            const int MINIBATCH_SIZE, const Reg_type REG_TYPE, const int PREV,
                            float* weight, const float* dw) {
    int m = blockIdx.x,                               // Index of prev. node in mbatch
        n = threadIdx.x + blockDim.x * blockIdx.y,    // Index of cur. node in mbatch
        w_index = m + n * PREV;
    weight[w_index] -= ALPHA * (((float) dw[w_index]/MINIBATCH_SIZE)
                                     + LAMBDA * weight[w_index]);
}

__global__ void FF_layer_kernels::reset_dw (const int PREV, float* dw) {
    int m = blockIdx.x,                               // Index of prev. node in mbatch
        n = threadIdx.x + blockDim.x * blockIdx.y,    // Index of cur. node in mbatch
        w_index = m + n * PREV;
    dw[w_index] = 0.0;
}

void FF_layer::fw_pass () {
    FF_layer_kernels::fw_pass <<<CUR, MINIBATCH_SIZE>>> (input, weight, output, PREV, CUR);
}

void FF_layer::back_pass () {
    FF_layer_kernels::compute_delta <<<dim_grid_delta, dim_block_delta>>> (delta_in, weight, delta, PREV, CUR);
    FF_layer_kernels::compute_dw <<<dim_grid_dw, dim_block_dw>>> (input, delta_in, dw, PREV, CUR, MINIBATCH_SIZE);
}

void FF_layer::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {
    FF_layer_kernels::update_weights <<<dim_grid_dw, dim_block_dw>>>
        (ALPHA, LAMBDA, MINIBATCH_SIZE, REG_TYPE, PREV, weight, dw);
}

void FF_layer::reset_dw() {
    FF_layer_kernels::reset_dw <<<dim_grid_dw, dim_block_dw>>> (PREV, dw);
}

void FF_layer::set_msize (int MINIBATCH_SIZE) {}

void FF_layer::store_weights (std::ofstream& to, float p) {
    float q = 1.0-p;
    float* weight_host = (float*) malloc(PREV * CUR * sizeof(float));
    cudaMemcpy( weight_host, weight, PREV* CUR * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < PREV * CUR; i++) {
        to << (weight_host[i] * q) << ",";
    }
    free(weight_host);
}

FF_layer::FF_layer (int PREV, int CUR, int MINIBATCH_SIZE,
                    curandGenerator_t& gen) {
    this->PREV = PREV; this->CUR = CUR;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;

    cudaMalloc((void**) &output, MINIBATCH_SIZE * CUR * sizeof(float));
    cudaMalloc((void**) &delta, MINIBATCH_SIZE * PREV * sizeof(float));
    cudaMalloc((void**) &weight, PREV * CUR * sizeof(float));
    cudaMalloc((void**) &dw, PREV * CUR * sizeof(float));

    //curandGenerateNormal(gen, weight, PREV * CUR, 0, 0.01);
    float He_init = sqrt( (float) 2/PREV);
    curandGenerateNormal(gen, weight, PREV * CUR, 0, He_init);

    // CUDA variables
    this->NODES_PER_BLOCK_delta = 1024 / MINIBATCH_SIZE;
    dim_block_delta = dim3(MINIBATCH_SIZE, NODES_PER_BLOCK_delta);
    dim_grid_delta  = dim3( std::max(PREV / NODES_PER_BLOCK_delta,1) );

    this->NODES_PER_BLOCK_dw = std::min(64, CUR);
    dim_block_dw = dim3(NODES_PER_BLOCK_dw);
    dim_grid_dw  = dim3(PREV, CUR / NODES_PER_BLOCK_dw);
}

FF_layer::~FF_layer() {
    cudaFree(output); cudaFree(delta);
    cudaFree(weight); cudaFree(dw);
}

// ==================================================================
// ======================== ReLU layer ==============================

const std::string ReLU_layer::layer_id = "ReLU";

namespace ReLU_layer_kernels {
    __global__ void fw_pass (const int CUR, const float LEAK, const float* input, float* output);
    __global__ void back_pass (const int CUR, const float LEAK, const float* delta_in, float* delta, const float* output);
}

__global__ void ReLU_layer_kernels::fw_pass (const int CUR, const float LEAK,
                                             const float* input, float* output) {
    int n = blockIdx.x,    // Index of current node in minibatch
        b = threadIdx.x,   // Index of minibatch
        node_index = n + b * CUR;
    output[node_index] = input[node_index];
    if (input[node_index] < 0) output[node_index] *= LEAK;
}

__global__ void ReLU_layer_kernels::back_pass (const int CUR, const float LEAK,
                                               const float* delta_in, float* delta,
                                               const float* input) {
    int n = blockIdx.x,    // Index of current node in minibatch
        b = threadIdx.x,   // Index of minibatch
        node_index = n + b * CUR;
    delta[node_index] = delta_in[node_index];
    if (input[node_index] < 0) delta[node_index] *= LEAK;
}

void ReLU_layer::fw_pass () {
    ReLU_layer_kernels::fw_pass <<<CUR, MINIBATCH_SIZE>>> (CUR, LEAK, input, output);
}

void ReLU_layer::back_pass () {
    ReLU_layer_kernels::back_pass <<<CUR, MINIBATCH_SIZE>>> (CUR, LEAK, delta_in, delta, input);
}

void ReLU_layer::update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {}
void ReLU_layer::reset_dw() {}
void ReLU_layer::set_msize (int MINIBATCH_SIZE) {}

ReLU_layer::ReLU_layer (int CUR, int MINIBATCH_SIZE, float LEAK) {
    this->input = input; this->delta_in = delta_in;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->CUR = CUR; this->LEAK = LEAK;
    cudaMalloc ((void**) &output, MINIBATCH_SIZE * CUR * sizeof(float));
    cudaMalloc ((void**) &delta, MINIBATCH_SIZE * CUR * sizeof(float));
}

ReLU_layer::~ReLU_layer () {
    cudaFree(output); cudaFree(delta);
}

// ==================================================================
// ======================= Softmax layer ============================

const std::string Softmax_layer::layer_id = "Softmax";

namespace Softmax_layer_kernels {
    __global__ void fw_pass (const float* input, float* output, const int CUR,
                             float* loss, Shape_type* type);
    __global__ void back_pass (const float* output, float* delta, const int* target,
                               const int CUR);
    __global__ void cast_target_to_int (Shape_type** target_shape, int* target_int);
}

__global__ void Softmax_layer_kernels::fw_pass ( const float* input, float* output,
                                                 const int CUR, float* loss, Shape_type* type ) {
    int b = threadIdx.x,   // Index of minibatch
        node_index;
    float max = -INFTY, sum = 0.0;
    int max_node_index;
    for (int n = 0; n < CUR; n++) {
        node_index = n + b * CUR;
        if ( input[node_index] > max) {
            max = input[node_index];
            max_node_index = node_index;
        }
    }
    type[b] = static_cast<Shape_type>(max_node_index % CUR);
    for (int n = 0; n < CUR; n++) {
        node_index = n + b * CUR;
        output[node_index] = input[node_index] - max;
        output[node_index] = exp(output[node_index]);
        sum += output[node_index];
    }
    loss[b] = log(sum) + max;
    for (int n = 0; n < CUR; n++) {
        node_index = n + b * CUR;
        output[node_index] /= sum;
    }
}

__global__ void Softmax_layer_kernels::back_pass (const float* output, float* delta,
                                                  const int* target, const int CUR) {
    int n = blockIdx.x,    // Index of node in minibatch
        b = threadIdx.x,   // Index of minibatch
        node_index = n + b * CUR;
    delta[node_index] = output[node_index];
    if (n == target[b]) delta[node_index] -= 1;
}

__global__ void Softmax_layer_kernels::cast_target_to_int (Shape_type** target, int* target_int) {
    int b = threadIdx.x;
    target_int[b] = static_cast<int>(*(target[b]));
}

void Softmax_layer::fw_pass () {
    Softmax_layer_kernels::fw_pass <<<1, MINIBATCH_SIZE>>> (input, output, CUR, loss, type);
}

void Softmax_layer::back_pass () {
    Softmax_layer_kernels::cast_target_to_int <<<1, MINIBATCH_SIZE>>> (target, target_int);
    Softmax_layer_kernels::back_pass <<<CUR, MINIBATCH_SIZE>>> (output, delta, target_int, CUR);
}

void Softmax_layer::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {}
void Softmax_layer::reset_dw() {}
void Softmax_layer::set_msize (int MINIBATCH_SIZE) {}

void Softmax_layer::set_targets (Shape_type** target) {
    this->target = target;
}

Softmax_layer::Softmax_layer (int PREV, int CUR, int MINIBATCH_SIZE) {
    this->PREV = PREV; this->CUR = CUR;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;

    cudaMalloc((void**) &output,     MINIBATCH_SIZE * CUR * sizeof(float));
    cudaMalloc((void**) &delta,      MINIBATCH_SIZE * PREV * sizeof(float));
    cudaMalloc((void**) &target,     MINIBATCH_SIZE * sizeof(Shape_type*));
    cudaMalloc((void**) &target_int, MINIBATCH_SIZE * sizeof(int));
    cudaMalloc((void**) &loss,       MINIBATCH_SIZE * sizeof(float));
    cudaMalloc((void**) &type,       MINIBATCH_SIZE * sizeof(Shape_type));
}

Softmax_layer::~Softmax_layer() {
    cudaFree(output); cudaFree(delta);
    cudaFree(target); cudaFree(target_int);
    cudaFree(loss); cudaFree(type);
}

// ==================================================================
// =============== Helper functions for Convnets ====================

__device__ int coords_to_index (int w, int h, int d, int b, int image_width, int num_filters) {
    return w + h*image_width + d*image_width*image_width + b*image_width*image_width*num_filters;
}

__device__ int coords_to_weight (int i, int j, int d, int n, int window_width, int input_depth) {
    return i + j*window_width + d*window_width*window_width + n*window_width*window_width*input_depth;
}

// ==================================================================
// ===================== Maxpool layer ==============================

const std::string Maxpool_layer::layer_id = "Maxpool";

namespace Maxpool_layer_kernels {
    __global__ void fw_pass (int win_width, int win_stride,
                    int input_image_width, int pooled_image_width, int num_filters,
                    const float* input, float* output, int* max_index_loc);
    __global__ void nil_delta (int input_image_width, int num_filters, float* delta);
    __global__ void back_pass (int pooled_image_width, int num_filters,
                    float* delta, const float* delta_in, const int* max_index_loc);
}

__global__ void Maxpool_layer_kernels::fw_pass (int win_width, int win_stride,
                    int input_image_width, int pooled_image_width, int num_filters,
                    const float* input, float* output, int* max_index_loc) {
    int b = threadIdx.x,    // Index of minibatch
       x0 = blockIdx.x * win_stride,
       y0 = blockIdx.y * win_stride,
        d = blockIdx.z,
        ind;
    float cur_max = -INFTY; int max_index;
    for (int i = 0; i < win_width; i++) {
        for (int j = 0; j < win_width; j++) {
            ind = coords_to_index (x0 + i, y0 + j, d, b, input_image_width, num_filters);
            if ( input[ind] > cur_max ) {
                cur_max = input[ind]; max_index = ind;
            }
        }
    }
    int output_index = coords_to_index(blockIdx.x, blockIdx.y, d, b, pooled_image_width, num_filters);
    output[output_index] = cur_max;
    max_index_loc[output_index] = max_index;
}

__global__ void Maxpool_layer_kernels::nil_delta (int input_image_width, int num_filters, float* delta) {
    int b = threadIdx.x,
        x = blockIdx.x,
        y = blockIdx.y,
        d = blockIdx.z,
        input_index = coords_to_index(x, y, d, b, input_image_width, num_filters);
    delta[input_index] = 0.0;
}

__global__ void Maxpool_layer_kernels::back_pass (int pooled_image_width, int num_filters,
                            float* delta, const float* delta_in, const int* max_index_loc) {
    int b = threadIdx.x,
        x = blockIdx.x,
        y = blockIdx.y,
        d = blockIdx.z,
        output_index = coords_to_index(x, y, d, b, pooled_image_width, num_filters);
    delta[max_index_loc[output_index]] = delta_in[output_index];
}

void Maxpool_layer::fw_pass () {
    Maxpool_layer_kernels::fw_pass <<<grid_dim_out, MINIBATCH_SIZE>>> (window_width, window_stride, input_dim.w, output_dim.w, input_dim.d,
                                                                       input, output, max_index_loc);
}

void Maxpool_layer::back_pass () {
    Maxpool_layer_kernels::nil_delta <<<grid_dim_in, MINIBATCH_SIZE>>> (input_dim.w, input_dim.d, delta);
    cudaDeviceSynchronize();
    Maxpool_layer_kernels::back_pass <<<grid_dim_out, MINIBATCH_SIZE>>> (output_dim.w, output_dim.d, delta, delta_in, max_index_loc);
}

void Maxpool_layer::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {}
void Maxpool_layer::reset_dw () {}
void Maxpool_layer::set_msize (int MINIBATCH_SIZE) {}

Maxpool_layer::Maxpool_layer (int window_width, int window_stride, convdim input_dim,
                              int MINIBATCH_SIZE) {
    this->window_width = window_width;
    this->window_stride = window_stride;
    this->input_dim = input_dim;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    output_dim = convdim( (input_dim.w - window_width) / window_stride + 1,
                          (input_dim.h - window_width) / window_stride + 1,
                           input_dim.d);
    grid_dim_out   = dim3(output_dim.w, output_dim.h, output_dim.d);
    grid_dim_in    = dim3(input_dim.w, input_dim.h, input_dim.d);
    cudaMalloc((void**) &output,        MINIBATCH_SIZE * output_dim.vol() * sizeof(float));
    cudaMalloc((void**) &delta,         MINIBATCH_SIZE * input_dim.vol()  * sizeof(float));
    cudaMalloc((void**) &max_index_loc, MINIBATCH_SIZE * output_dim.vol() * sizeof(int));
}

Maxpool_layer::~Maxpool_layer () {
    cudaFree(output); cudaFree(delta); cudaFree(max_index_loc);
}

// ==================================================================
// ===================== Conv layer =================================

const std::string Conv_layer::layer_id = "Convolutional";

namespace Conv_layer_kernels {
    __global__ void pad_input (int input_image_width, int padded_image_width, int input_depth, int pad,
                               const float* input, float* padded_input);
    __global__ void fw_pass (int window_width, int window_stride, int input_image_width, int output_image_width,
                             int input_depth, int num_filters, const float* weight, const float* padded_input, float* output);
    __global__ void compute_delta (int input_image_width, int output_image_width, int input_depth, int num_filters,
                                   int window_width, int window_stride, int pad,
                                   float* delta, const float* delta_in, const float* weight);
    __global__ void compute_dw (int window_width, int window_stride, int input_image_width, int output_image_width,
                                int input_depth, int num_filters, const float* delta_in, const float* padded_input,
                                float* dw, int MINIBATCH_SIZE);
    __global__ void update_weights (float ALPHA, float LAMBDA, int MINIBATCH_SIZE, Reg_type REG_TYPE,
                                    int window_width, int input_depth, float* weight, const float* dw);
    __global__ void reset_dw (int window_width, int input_depth, float* dw);
}

__global__ void Conv_layer_kernels::pad_input (int input_image_width, int padded_image_width, int input_depth, int pad,
                                               const float* input, float* padded_input) {
    int b = threadIdx.x,
        x = blockIdx.x,
        y = blockIdx.y,
        d = blockIdx.z,
        input_index = coords_to_index(x, y, d, b, input_image_width, input_depth),
        padded_index = coords_to_index(x+pad, y+pad, d, b, padded_image_width, input_depth);
    padded_input[padded_index] = input[input_index];
}

__global__ void Conv_layer_kernels::fw_pass (int window_width, int window_stride, int input_image_width, int output_image_width,
                                             int input_depth, int num_filters, const float* weight, const float* padded_input,
                                             float* output) {
    int b = threadIdx.x,
        x0 = blockIdx.x,
        y0 = blockIdx.y,
        n = blockIdx.z,
        output_index = coords_to_index(x0, y0, n, b, output_image_width, num_filters),
        input_index, weight_index;
    float out = 0.0;

    for (int i = 0; i < window_width; i++) {
        for (int j = 0; j < window_width; j++) {
            for (int d = 0; d < input_depth; d++) {
                input_index = coords_to_index(x0*window_stride + i, y0*window_stride + j, d, b, input_image_width, input_depth);
                weight_index = coords_to_weight(i, j, d, n, window_width, input_depth);
                out += weight[weight_index] * padded_input[input_index];
            }
        }
    }
    output[output_index] = out;
}

__global__ void Conv_layer_kernels::compute_delta (int input_image_width, int output_image_width, int input_depth, int num_filters,
                                                   int window_width, int window_stride, int pad,
                                                   float* delta, const float* delta_in, const float* weight) {
    int b = threadIdx.x,
        x = blockIdx.x,
        y = blockIdx.y,
        d = blockIdx.z;

    if (x >= pad && y >= pad && x-pad < input_image_width && y-pad < input_image_width) {
        int input_index = coords_to_index(x-pad, y-pad, d, b, input_image_width, input_depth),
            output_index, weight_index;
        float delta_prep = 0.0;
        int wmin = (int) fmaxf ( ceilf((x-window_width+1)/window_stride), 0),
            wmax = (int) fminf (floorf ( x/window_stride ), output_image_width-1),
            hmin = (int) fmaxf (ceilf( (y-window_width+1)/window_stride), 0),
            hmax = (int) fminf (floorf ( y/window_stride ), output_image_width-1);

        for (int w = wmin; w <= wmax; w++) {
            for (int h = hmin; h <= hmax; h++) {
                for (int n = 0; n < num_filters; n++) {
                    output_index = coords_to_index(w, h, n, b, output_image_width, num_filters);
                    weight_index = coords_to_weight(x - w*window_stride, y - h*window_stride, d, n, window_width, input_depth);
                    delta_prep += delta_in[output_index] * weight[weight_index];
                }
            }
        }
        delta[input_index] = delta_prep;
    }
}

__global__ void Conv_layer_kernels::compute_dw (int window_width, int window_stride, int input_image_width, int output_image_width,
                                                int input_depth, int num_filters, const float* delta_in, const float* padded_input,
                                                float* dw, int MINIBATCH_SIZE) {
    int n = threadIdx.x,
        i = blockIdx.x,
        j = blockIdx.y,
        d = blockIdx.z,
        weight_index = coords_to_weight(i, j, d, n, window_width, input_depth),
        input_index, output_index;
    float diff = 0.0;

    for (int x = 0; x < output_image_width; x++) {
        for (int y = 0; y < output_image_width; y++) {
            for (int b = 0; b < MINIBATCH_SIZE; b++) {
                output_index = coords_to_index(x, y, n, b, output_image_width, num_filters);
                input_index = coords_to_index(x*window_stride + i, y*window_stride + j, d, b, input_image_width, input_depth);
                diff += delta_in[output_index] * padded_input[input_index];
            }
        }
    }
    dw[weight_index] = diff;
}

__global__ void Conv_layer_kernels::update_weights (float ALPHA, float LAMBDA, int MINIBATCH_SIZE, Reg_type REG_TYPE,
                                                    int window_width, int input_depth, float* weight, const float* dw) {
    int n = threadIdx.x,
        i = blockIdx.x,
        j = blockIdx.y,
        d = blockIdx.z,
        weight_index = coords_to_weight(i, j, d, n, window_width, input_depth);
    weight[weight_index] -= ALPHA * ( ((float) dw[weight_index]/MINIBATCH_SIZE) + LAMBDA * weight[weight_index]);
}

__global__ void Conv_layer_kernels::reset_dw (int window_width, int input_depth, float* dw) {
    int n = threadIdx.x,
        i = blockIdx.x,
        j = blockIdx.y,
        d = blockIdx.z,
        weight_index = coords_to_weight(i, j, d, n, window_width, input_depth);
    dw[weight_index] = 0.0;
}

void Conv_layer::fw_pass () {
    Conv_layer_kernels::pad_input <<<grid_dim_in, MINIBATCH_SIZE>>>
            (input_dim.w, padded_dim.w, input_dim.d, pad, input, padded_input);
    cudaDeviceSynchronize();
    Conv_layer_kernels::fw_pass <<<grid_dim_out, MINIBATCH_SIZE>>>
             (window_width, window_stride, padded_dim.w, output_dim.w,
             padded_dim.d, output_dim.d, weight, padded_input, output);
}

void Conv_layer::back_pass() {
    Conv_layer_kernels::compute_dw <<<grid_dim_weight, num_filters>>>
            (window_width, window_stride, padded_dim.w, output_dim.w,
             padded_dim.d, output_dim.d, delta_in, padded_input, dw, MINIBATCH_SIZE);
    Conv_layer_kernels::compute_delta <<<grid_dim_pad, MINIBATCH_SIZE>>>
            (input_dim.w, output_dim.w, input_dim.d, output_dim.d,
             window_width, window_stride, pad, delta, delta_in, weight);
}

void Conv_layer::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {
    Conv_layer_kernels::update_weights <<<grid_dim_weight, num_filters>>>
            (ALPHA, LAMBDA, MINIBATCH_SIZE, REG_TYPE, window_width, padded_dim.d, weight, dw);
}

void Conv_layer::reset_dw() {
    Conv_layer_kernels::reset_dw <<<grid_dim_weight, num_filters>>> (window_width, padded_dim.d, dw);
}

void Conv_layer::set_msize (int MINIBATCH_SIZE) {}

void Conv_layer::store_weights (std::ofstream& to, float p) {
    float q = 1.0-p;
    int n_weights = window_width * window_width * input_dim.d * num_filters;
    float* weight_host = (float*) malloc(n_weights * sizeof(float));
    cudaMemcpy( weight_host, weight, n_weights * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_weights; i++) {
          to << (weight_host[i] * q) << ",";
    }
    free(weight_host);
}

Conv_layer::Conv_layer (int window_width, int window_stride, int num_filters, int pad,
                        convdim input_dim, int MINIBATCH_SIZE, curandGenerator_t& gen) {
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->window_width = window_width; this->window_stride = window_stride;
    this->num_filters = num_filters; this->pad = pad;
    this->input_dim = input_dim;
    this->padded_dim = convdim (input_dim.w + 2*pad,
                                input_dim.h + 2*pad,
                                input_dim.d);
    this->output_dim = convdim( ((float) (padded_dim.w - window_width) / window_stride ) + 1,
                                ((float) (padded_dim.h - window_width) / window_stride ) + 1,
                                num_filters );

    int n_weights = num_filters * window_width * window_width * input_dim.d;

    cudaMalloc ((void**) &padded_input, MINIBATCH_SIZE * padded_dim.vol() * sizeof(float));
    cudaMalloc ((void**) &delta,        MINIBATCH_SIZE * input_dim.vol() * sizeof(float));
    cudaMalloc ((void**) &output,       MINIBATCH_SIZE * output_dim.vol() * sizeof(float));
    cudaMalloc ((void**) &weight, n_weights * sizeof(float));
    cudaMalloc ((void**) &dw,     n_weights * sizeof(float));

    cudaMemset (padded_input, 0, MINIBATCH_SIZE * padded_dim.vol() * sizeof(float));

    grid_dim_pad  = dim3 (padded_dim.w, padded_dim.h, padded_dim.d);
    grid_dim_in   = dim3 (input_dim.w,  input_dim.h,  input_dim.d);
    grid_dim_out  = dim3 (output_dim.w, output_dim.h, output_dim.d);
    grid_dim_weight = dim3 (window_width, window_width, input_dim.d);

    float He_init = sqrt( (float) 2/input_dim.vol() );
    curandGenerateNormal(gen, weight, n_weights, 0, He_init);
}

Conv_layer::~Conv_layer () {
    cudaFree(padded_input); cudaFree(output); cudaFree(delta);
    cudaFree(weight); cudaFree(dw);
}

// ==================================================================
// ===================== Dropout layer ==============================

const std::string Dropout_layer::layer_id = "Dropout";
bool Dropout_layer::training = true;

namespace Dropout_layer_kernels {
    __global__ void fw_pass_training (const float* mask, float p, const float* input, float* output);
    __global__ void fw_pass_testing  (const float* input, float* output, float p);
    __global__ void back_pass        (const float* mask, float p, const float* delta_in, float* delta);
}

__global__ void Dropout_layer_kernels::fw_pass_training (const float* mask, float p, const float* input, float* output) {
    int node = blockIdx.x + gridDim.x * threadIdx.x;
    output[node] = (mask[node] > p ? input[node] : 0.0);
}

__global__ void Dropout_layer_kernels::fw_pass_testing (const float* input, float* output, float p) {
    int node = blockIdx.x + gridDim.x * threadIdx.x;
    output[node] = input[node] * p;
}

__global__ void Dropout_layer_kernels::back_pass (const float* mask, float p, const float* delta_in, float* delta) {
    int node = blockIdx.x + gridDim.x * threadIdx.x;
    delta[node] = (mask[node] > p ? delta_in[node] : 0.0);
}

void Dropout_layer::fw_pass () {
    if (training) {
        curandGenerateUniform(*gen, mask, MINIBATCH_SIZE * CUR);
        cudaDeviceSynchronize();
        Dropout_layer_kernels::fw_pass_training <<<CUR, MINIBATCH_SIZE>>> (mask, p, input, output);
    }
    else
        Dropout_layer_kernels::fw_pass_testing  <<<CUR, MINIBATCH_SIZE>>> (input, output, 1-p);
}

void Dropout_layer::back_pass () {
    Dropout_layer_kernels::back_pass <<<CUR, MINIBATCH_SIZE>>> (mask, p, delta_in, delta);
}

void Dropout_layer::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {}
void Dropout_layer::reset_dw () {}
void Dropout_layer::set_msize (int MINIBATCH_SIZE) {}

Dropout_layer::Dropout_layer (int CUR, int MINIBATCH_SIZE, float p, curandGenerator_t* gen) {
    this->CUR = CUR; this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->p = p; this->gen = gen;

    cudaMalloc ((void**) &output, MINIBATCH_SIZE * CUR * sizeof(float));
    cudaMalloc ((void**) &delta,  MINIBATCH_SIZE * CUR * sizeof(float));
    cudaMalloc ((void**) &mask,   MINIBATCH_SIZE * CUR * sizeof(float));
}

Dropout_layer::~Dropout_layer () {
    cudaFree(output); cudaFree(delta); cudaFree(mask);
}

// ===================================================================
// ======================  Neural network  ===========================

namespace Neural_net_kernels {
    __global__ void load_input (float* input, float** bitmap, const int INPUT_DIM);
    __global__ void update_acl (float* sm_loss, Shape_type* sm_type,
             const float* output, Shape_type** actual_type,
             float* tpc, float* tloss, float* vpc, float* vloss, float* pc, float* loss,
             Dataset_type D_TYPE, int t, int MINIBATCH_SIZE);
    __global__ void max(float* layer);
}

__global__ void Neural_net_kernels::load_input (float* input, float** bitmap, const int INPUT_DIM) {
    int x = blockIdx.x,
        y = blockIdx.y,
        b = threadIdx.x,
        bit_index = x + INPUT_DIM*y,
        input_index = bit_index + INPUT_DIM*INPUT_DIM*b;
    input[input_index] = bitmap[b][bit_index];
}

__global__ void Neural_net_kernels::update_acl (float* sm_loss, Shape_type* sm_type,
             const float* output, Shape_type** actual_type,
             float* tpc, float* tloss, float* vpc, float* vloss, float* pc, float* loss,
             Dataset_type D_TYPE, int t, int MINIBATCH_SIZE) {
    int target_node; float l = 0.0, p = 0.0;
    for (int b = 0; b < MINIBATCH_SIZE; b++) {
        if (sm_type[b] == *(actual_type[b])) p++;
        target_node = static_cast<int>(*(actual_type[b]));
        l += sm_loss[b] - output[target_node+b*NUM_SHAPES];
    }
    switch (D_TYPE) {
        case TRAIN    :    tpc[t] += p; tloss[t] += l;  break;
        case VALID    :    vpc[t] += p; vloss[t] += l;  break;
        case TEST     :    *pc += p;    *loss += l;     break;
        default       :    printf("Error in Neural_net_kernels::update_acl --- Invalid data type\n"); break;
    }
}

__global__ void Neural_net_kernels::max (float* layer) {
    printf("Starting max f'n\n");
    float max = -INFTY; int max_class;
    for (int s = 0; s < NUM_SHAPES; s++) {
        printf("%d === %f\n", s, layer[s]);
        if (layer[s] > max) {
           max = layer[s];
           max_class = s;
        }
    }
    printf("%d\n", max_class);
}

void Neural_net::classify(float* input) {
    float* input_dev;
    cudaMalloc((void**) &input_dev, 128*128*sizeof(float));
    cudaMemset(input_dev, 0, 128*128*sizeof(float));
    cudaMemcpy(input_dev, input, 128*128*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    float* input2[MINIBATCH_SIZE];
    for (int b = 0; b < MINIBATCH_SIZE; b++)
        input2[b] = input_dev;
    float** input2_dev;
    cudaMalloc((void**)&input2_dev, MINIBATCH_SIZE * sizeof(float*));
    cudaMemcpy(input2_dev, input2, MINIBATCH_SIZE * sizeof(float*), cudaMemcpyHostToDevice);
    load_input(input2_dev);
    cudaDeviceSynchronize();
    fw_pass();
    Neural_net_kernels::max<<<1,1>>>(layer[LAYERS]->output);

    cudaFree(input_dev); cudaFree(input2_dev);
}

void Neural_net::load_input (float* input) {
    layer[0]->input = input;
}

void Neural_net::load_input (float** bitmap) {
    dim3 grid(INPUT_DIM, INPUT_DIM);
    Neural_net_kernels::load_input <<<grid, MINIBATCH_SIZE>>> (this->input, bitmap, INPUT_DIM);
    cudaDeviceSynchronize();
    load_input(this->input);
}

void Neural_net::fw_pass () {
    for (int r = 0; r < LAYERS+1; r++) {
        layer[r]->fw_pass();
        check_cuda(layer[r]->get_id() + " fw pass");
    }
}

void Neural_net::back_pass (Shape_type** target) {
    Softmax_layer* softmax_head = static_cast<Softmax_layer*>(layer[LAYERS]);
    softmax_head->set_targets(target);
    for (int r = LAYERS; r >= 0; r--) {
        layer[r]->back_pass();
        check_cuda(layer[r]->get_id() + " back pass");
    }
}

void Neural_net::update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) {
    for (int r = 0; r < LAYERS; r++)
        layer[r]->update_weights(ALPHA, LAMBDA, REG_TYPE);
    check_cuda("Weight update");
}

void Neural_net::reset_dw() {
    for (int r = 0; r < LAYERS; r++)
        layer[r]->reset_dw();
    check_cuda("Weight reset");
}

void Neural_net::process_minibatch_acl (std::vector<GPU_Labeled_shape> shape,
                                        Dataset_type D_TYPE, Ledger* ledger, int t,
                                        float** dev_bmp_batch, Shape_type** dev_type_batch) {
    float* bitmap[MINIBATCH_SIZE];
    Shape_type* actual_type[MINIBATCH_SIZE];
    for (int b = 0; b < MINIBATCH_SIZE; b++) {
        bitmap[b] = shape[b].bitmap;
        actual_type[b] = shape[b].type;
    }
    copy_pointers(bitmap, actual_type, dev_bmp_batch, dev_type_batch, MINIBATCH_SIZE);
    load_input(dev_bmp_batch);
    fw_pass ();
    Neural_net_kernels::update_acl <<<1,1>>> (softmax_head->loss, softmax_head->type,
            layer[LAYERS-1]->output, dev_type_batch,
            ledger->tpc, ledger->tloss, ledger->vpc, ledger->vloss, ledger->pc, ledger->loss,
            D_TYPE, t, MINIBATCH_SIZE);
    cudaDeviceSynchronize();
}

Neural_net::Neural_net (int LAYERS, int MINIBATCH_SIZE, bool dropout) {
    this->LAYERS = LAYERS; this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->dropout = dropout;
    layer = new NN_layer*[LAYERS + 1];
                               // +1 for the classifier head
    cudaMalloc ((void**) &input, MINIBATCH_SIZE * INPUT_DIM * INPUT_DIM * sizeof(float));
}

Neural_net::~Neural_net() {
    cudaFree(input);
    delete[] layer;
}

// ===================================================================
// ===================  Feedforward network  =========================

std::string FF_net::return_info_string() {
    std::string id = "ff-net-gpu---layers-" + std::to_string(LAYERS)
                                +"--nodes-";
    for (int n = 0; n <= (LAYERS/2+1); n++)
        id += std::to_string(NODES[n]) + "-";
    return id;
}

void FF_net::store_weights(std::ofstream& to) {
    FF_layer* ff_ptr;
    to << std::to_string(HIDDEN_BLOCKS+1) << ";";
    if (dropout) {
        Dropout_layer* drop_ptr;
        float p = 0.0;
        for (int b = 0; b <= HIDDEN_BLOCKS; b++) {
            ff_ptr = static_cast<FF_layer*>(layer[3*b]);
            drop_ptr = static_cast<Dropout_layer*>(layer[3*b+1]);
            p = drop_ptr->p;
            ff_ptr->store_weights(to, p);
            to << ";";
        }
    }
    else {
        for (int b = 0; b <= HIDDEN_BLOCKS; b++) {
            ff_ptr = static_cast<FF_layer*>(layer[2*b]);
            ff_ptr->store_weights(to, 0.0);
            to << ";";
        }
    }
    to << "!";
}

// ====================================================================================================================
//                                     FF_net without dropout:
//
//         ( layer[0]   ->   layer[1]  ) * HIDDEN_UNITS ->     ...     -> layer[LAYERS-1] -> layer[LAYERS]
// (input)     FF             ReLU             FF                              FF              Softmax
// NODES[0]       NODES[1]         NODES[1]     NODES[2]    NODES[(LAYERS-1)/2]      NODES[(LAYERS+1)/2]
// 128*128                                                                              = NUM_SHAPES

FF_net::FF_net (int HIDDEN_BLOCKS, int* NODES, int MINIBATCH_SIZE,
                curandGenerator_t& gen, float LEAK)
    : Neural_net (HIDDEN_BLOCKS * 2 + 1, MINIBATCH_SIZE) {

    // Each HIDDEN BLOCK contains two layers (FF + ReLU).
    // LAYERS+1 = 2 * HIDDEN_BLOCKS + FF + Softmax
    // NODES is an array of integers that specifies the number of neurons in the
    // hidden blocks. The index of the array NODES should go up to ((LAYERS+1)/2).
    this->HIDDEN_BLOCKS = HIDDEN_BLOCKS; this->NODES = NODES;
    for (int r = 0; r < HIDDEN_BLOCKS; r++) {
        layer[2*r]   = new FF_layer(NODES[r], NODES[r+1], MINIBATCH_SIZE, gen);
        layer[2*r+1] = new ReLU_layer        (NODES[r+1], MINIBATCH_SIZE, LEAK);
    }   // Defines layer[0] up to layer[LAYERS-2]
    layer[LAYERS-1] = new FF_layer(NODES[LAYERS/2], NODES[LAYERS/2+1], MINIBATCH_SIZE, gen);
    layer[LAYERS]   = new Softmax_layer            (NODES[LAYERS/2+1], NUM_SHAPES, MINIBATCH_SIZE);

    // Link up the layers
    layer[0]->input = this->input;
    layer[0]->delta_in = layer[1]->delta;
    for (int r = 1; r < LAYERS; r++) {
        layer[r]->input = layer[r-1]->output;
        layer[r]->delta_in = layer[r+1]->delta;
    }
    layer[LAYERS]->input = layer[LAYERS-1]->output;
    layer[LAYERS]->delta_in = 0;

    softmax_head = static_cast<Softmax_layer*>(layer[LAYERS]);
}

// ====================================================================================================================
//                                  FF_net with dropout:
//
//         ( layer[0]   ->   layer[1]   ->   layer[2]  ) * HIDDEN_UNITS ->    ...     -> layer[LAYERS-1] -> layer[LAYERS]
// (input)      FF           Dropout           ReLU                                           FF              Softmax
// NODES[0]       NODES[1]         NODES[1]         NODES[1]               NODES[(LAYERS-1)/3]    NODES[(LAYERS-1)/3+1]
// 128*128                                                                                            = NUM_SHAPES

FF_net::FF_net (int HIDDEN_BLOCKS, int* NODES, int MINIBATCH_SIZE,
                curandGenerator_t& gen, bool dropout, float LEAK)
    : Neural_net (HIDDEN_BLOCKS * 3 + 1, MINIBATCH_SIZE, dropout) {
    this->HIDDEN_BLOCKS = HIDDEN_BLOCKS; this->NODES = NODES;
    for (int r = 0; r < HIDDEN_BLOCKS; r++) {
        layer[3*r]   = new FF_layer(NODES[r], NODES[r+1], MINIBATCH_SIZE, gen);
        layer[3*r+1] = new Dropout_layer     (NODES[r+1], MINIBATCH_SIZE, (r == 0 ? 0.05 : 0.5), &gen);
        layer[3*r+2] = new ReLU_layer        (NODES[r+1], MINIBATCH_SIZE, LEAK);
    }   // Defines layer[3] up to layer[LAYERS-2]
    layer[LAYERS-1] = new FF_layer(NODES[(LAYERS-1)/3], NODES[(LAYERS-1)/3+1], MINIBATCH_SIZE, gen);
    layer[LAYERS]   = new Softmax_layer                (NODES[(LAYERS-1)/3+1], NUM_SHAPES, MINIBATCH_SIZE);

    // Link up the layers
    layer[0]->input = this->input;
    layer[0]->delta_in = layer[1]->delta;
    for (int r = 1; r < LAYERS; r++) {
        layer[r]->input = layer[r-1]->output;
        layer[r]->delta_in = layer[r+1]->delta; }
    layer[LAYERS]->input = layer[LAYERS-1]->output;
    layer[LAYERS]->delta_in = 0;

    softmax_head = static_cast<Softmax_layer*>(layer[LAYERS]);
    drp_layer = static_cast<Dropout_layer*>(layer[1]);
}

FF_net::~FF_net () {
    for (int r = 0; r < LAYERS+1; r++)
        delete layer[r];
}

// ===================================================================
// ==================  Convolutonal network  =========================

std::string Conv_net::return_info_string() {
    std::string id = "conv-net-gpu---";
    return id;
}

void Conv_net::store_weights(std::ofstream& to) {
    Conv_layer* conv_ptr;
    FF_layer* ff_ptr;
    if (dropout) {}
    else {
        for (int b = 0; b < CONV_BLOCKS; b++) {
            conv_ptr = static_cast<Conv_layer*>(layer[5*b]);
            conv_ptr->store_weights(to, 0.0);
            to << ";";
            conv_ptr = static_cast<Conv_layer*>(layer[5*b+2]);
            conv_ptr->store_weights(to, 0.0);
            to << ";";
        }
        for (int b = 0; b < FF_BLOCKS; b++) {
            ff_ptr = static_cast<FF_layer*>(layer[5*CONV_BLOCKS+2*b]);
            ff_ptr->store_weights(to, 0.0);
            to << ";";
        }
        ff_ptr = static_cast<FF_layer*>(layer[5*CONV_BLOCKS+2*FF_BLOCKS]);
        ff_ptr->store_weights(to, 0.0);
        to << ";";
    }
    to << "!";
}

// ================================================================================================
//                     Conv net architecture (no dropout):
//        (  layer[0] -> layer[1] -> layer[2] -> layer[3] -> layer[4] ) * CONV_BLOCKS -> ...
//  input      Conv        ReLU        Conv        ReLU        Maxpool
//
// ... -> ( layer[B] -> layer[B+1] ) * FF_BLOCKS -> layer[B+F] -> layer[B+F+1]
//             FF          ReLU                   FF (16 nodes)      Softmax
//
//       Where B = 5*CONV_BLOCKS and F = 2*FF_BLOCKS

Conv_net::Conv_net (int CONV_BLOCKS, int* W_WIDTH, int* STRIDE, int* FILTERS, int* PADDING,
                    int FF_BLOCKS, int* FF_NODES, convdim input_dim, curandGenerator_t& gen,
                    int MINIBATCH_SIZE, float LEAK)
    : Neural_net(5*CONV_BLOCKS + 2*FF_BLOCKS+1, MINIBATCH_SIZE)
{
    this->CONV_BLOCKS = CONV_BLOCKS;
    this->W_WIDTH = W_WIDTH; this->STRIDE = STRIDE;
    this->FILTERS = FILTERS; this->PADDING = PADDING;
    this->FF_BLOCKS = FF_BLOCKS; this->FF_NODES = FF_NODES;
    this->input_dim = input_dim;

    Conv_layer* cnv_ptr; Maxpool_layer* mp_ptr;
    for (int b = 0; b < CONV_BLOCKS; b++) {
        layer[5*b]   = new Conv_layer(W_WIDTH[2*b],   STRIDE[2*b],   FILTERS[2*b],   PADDING[2*b],
                                    (b == 0 ? input_dim : mp_ptr->output_dim), MINIBATCH_SIZE, gen);
                                    cnv_ptr = static_cast<Conv_layer*>(layer[5*b]);
        layer[5*b+1] = new ReLU_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE, LEAK);
        layer[5*b+2] = new Conv_layer(W_WIDTH[2*b+1], STRIDE[2*b+1], FILTERS[2*b+1], PADDING[2*b+1],
                                    cnv_ptr->output_dim, MINIBATCH_SIZE, gen);
                                    cnv_ptr = static_cast<Conv_layer*>(layer[5*b+2]);
        layer[5*b+3] = new ReLU_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE, LEAK);
        layer[5*b+4] = new Maxpool_layer(2, 2, cnv_ptr->output_dim, MINIBATCH_SIZE);
                                    mp_ptr = static_cast<Maxpool_layer*>(layer[5*b+4]); }
    for (int b = 0; b < FF_BLOCKS; b++) {
        layer[5*CONV_BLOCKS+2*b]   = new FF_layer ((b == 0 ? mp_ptr->output_dim.vol() : FF_NODES[b-1]),
                                                    FF_NODES[b], MINIBATCH_SIZE, gen);
        layer[5*CONV_BLOCKS+2*b+1] = new ReLU_layer(FF_NODES[b], MINIBATCH_SIZE, LEAK); }
    layer[5*CONV_BLOCKS+2*FF_BLOCKS]   = new FF_layer( (FF_BLOCKS == 0 ? mp_ptr->output_dim.vol() : FF_NODES[FF_BLOCKS-1]),
                                                            NUM_SHAPES, MINIBATCH_SIZE, gen);
    layer[5*CONV_BLOCKS+2*FF_BLOCKS+1] = new Softmax_layer (NUM_SHAPES, NUM_SHAPES, MINIBATCH_SIZE);

    // Link up the layers
    layer[0]->input = this->input;
    layer[0]->delta_in = layer[1]->delta;
    for (int r = 1; r < LAYERS; r++) {
        layer[r]->input = layer[r-1]->output;
        layer[r]->delta_in = layer[r+1]->delta; }
    layer[LAYERS]->input = layer[LAYERS-1]->output;
    layer[LAYERS]->delta_in = 0;

    softmax_head = static_cast<Softmax_layer*>(layer[LAYERS]);
}

// ================================================================================================
//                     Conv net architecture (with dropout):
//        (  layer[0] -> layer[1] -> layer[2] -> layer[3] -> layer[4] -> layer[5] -> layer[6] ) * CONV_BLOCKS -> ...
//  input      Conv      Dropout       ReLU        Conv       Dropout      ReLU       Maxpool
//
// ... -> ( layer[B] -> layer[B+1] -> layer[B+2] ) * FF_BLOCKS -> layer[B+F] -> layer[B+F+1]
//             FF        Dropout        ReLU                    FF (16 nodes)      Softmax
//
//       Where B = 7*CONV_BLOCKS and F = 3*FF_BLOCKS

Conv_net::Conv_net (int CONV_BLOCKS, int* W_WIDTH, int* STRIDE, int* FILTERS, int* PADDING,
                    int FF_BLOCKS, int* FF_NODES, convdim input_dim, curandGenerator_t& gen,
                    int MINIBATCH_SIZE, bool dropout, float LEAK)
    : Neural_net(7*CONV_BLOCKS + 3*FF_BLOCKS + 1, MINIBATCH_SIZE, dropout)
{
    this->CONV_BLOCKS = CONV_BLOCKS;
    this->W_WIDTH = W_WIDTH; this->STRIDE = STRIDE;
    this->FILTERS = FILTERS; this->PADDING = PADDING;
    this->FF_BLOCKS = FF_BLOCKS; this->FF_NODES = FF_NODES;
    this->input_dim = input_dim;

    Conv_layer* cnv_ptr; Maxpool_layer* mp_ptr;
    for (int b = 0; b < CONV_BLOCKS; b++) {
        layer[7*b]   = new Conv_layer(W_WIDTH[2*b],   STRIDE[2*b],   FILTERS[2*b],   PADDING[2*b],
                                    (b == 0 ? input_dim : mp_ptr->output_dim), MINIBATCH_SIZE, gen);
                                    cnv_ptr = static_cast<Conv_layer*>(layer[7*b]);
        layer[7*b+1] = new Dropout_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE, 0.25, &gen);
        layer[7*b+2] = new ReLU_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE);
        layer[7*b+3] = new Conv_layer(W_WIDTH[2*b+1], STRIDE[2*b+1], FILTERS[2*b+1], PADDING[2*b+1],
                                    cnv_ptr->output_dim, MINIBATCH_SIZE, gen);
                                    cnv_ptr = static_cast<Conv_layer*>(layer[7*b+3]);
        layer[7*b+4] = new Dropout_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE, 0.25, &gen);
        layer[7*b+5] = new ReLU_layer(cnv_ptr->output_dim.vol(), MINIBATCH_SIZE);
        layer[7*b+6] = new Maxpool_layer(2, 2, cnv_ptr->output_dim, MINIBATCH_SIZE);
                                    mp_ptr = static_cast<Maxpool_layer*>(layer[7*b+6]); }
    for (int b = 0; b < FF_BLOCKS; b++) {
        layer[7*CONV_BLOCKS+3*b]   = new FF_layer( (b == 0 ? mp_ptr->output_dim.vol() : FF_NODES[b-1]),
                                                       FF_NODES[b], MINIBATCH_SIZE, gen);
        layer[7*CONV_BLOCKS+3*b+1] = new Dropout_layer(FF_NODES[b], MINIBATCH_SIZE, 0.5, &gen);
        layer[7*CONV_BLOCKS+3*b+2] = new ReLU_layer   (FF_NODES[b], MINIBATCH_SIZE, LEAK); }
    layer[7*CONV_BLOCKS+3*FF_BLOCKS]   = new FF_layer( (FF_BLOCKS == 0 ? mp_ptr->output_dim.vol() : FF_NODES[FF_BLOCKS-1]),
                                                            NUM_SHAPES, MINIBATCH_SIZE, gen);
    layer[7*CONV_BLOCKS+3*FF_BLOCKS+1] = new Softmax_layer (NUM_SHAPES, NUM_SHAPES, MINIBATCH_SIZE);

    // Link up the layers
    layer[0]->input = this->input;
    layer[0]->delta_in = layer[1]->delta;
    for (int r = 1; r < LAYERS; r++) {
        layer[r]->input = layer[r-1]->output;
        layer[r]->delta_in = layer[r+1]->delta; }
    layer[LAYERS]->input = layer[LAYERS-1]->output;
    layer[LAYERS]->delta_in = 0;

    softmax_head = static_cast<Softmax_layer*>(layer[LAYERS]);
    drp_layer = static_cast<Dropout_layer*>(layer[1]);
}

Conv_net::~Conv_net () {
    for (int r = 0; r < LAYERS+1; r++)
        delete layer[r];
}


// ==================================================================
// =====================  Trainer  ==================================

void NN_trainer::find_acl (Dataset_type D_TYPE, Neural_net* net) {
    std::vector<GPU_Labeled_shape> data_slice(MINIBATCH_SIZE);
    GPU_Data_set* data;

    switch(D_TYPE) {
        case TRAIN    :    data = train_set; break;
        case VALID    :    data = valid_set; break;
        case TEST     :    data = test_set;  break;
        default       :    data = 0;         break;   // Error
    }

    for (std::vector<GPU_Labeled_shape>::iterator it = data->data.begin();
                                                  it != data->data.end();
                                                  it += MINIBATCH_SIZE) {
        std::copy (it, it+MINIBATCH_SIZE, data_slice.begin());
        net->process_minibatch_acl (data_slice, D_TYPE, ledger, t, dev_bmp_batch, dev_type_batch);
    }
}

void NN_trainer::train (const int epochs, std::default_random_engine& gen, bool dropout) {
    this->dropout = dropout;
    std::vector<GPU_Labeled_shape> data_slice(MINIBATCH_SIZE);
    float* bmp_batch[MINIBATCH_SIZE]; Shape_type* type_batch[MINIBATCH_SIZE];

    for (int ep = 0; ep < epochs; ep++) {
        if (dropout) net->drp_layer->training = true;
        std::shuffle(train_set->data.begin(), train_set->data.end(), gen);
        for (int p = 0; p < train_set->data.size(); p+=MINIBATCH_SIZE) {

            for (int b = 0; b < MINIBATCH_SIZE; b++) {
                bmp_batch[b] = train_set->data[p+b].bitmap;
                type_batch[b] = train_set->data[p+b].type;
            }
            copy_pointers(bmp_batch, type_batch, dev_bmp_batch, dev_type_batch, MINIBATCH_SIZE);
            net->load_input (dev_bmp_batch);
            net->fw_pass ();
            net->back_pass  (dev_type_batch);
            net->update_weights (ALPHA, LAMBDA, REG_TYPE);
            net->reset_dw();
        }
        if (dropout) net->drp_layer->training = false;
        find_acl(TRAIN, net); find_acl(VALID, net);
        cudaDeviceSynchronize();
        ledger->end_cur_epoch(t);
        t++;
    }
}

void NN_trainer::reset() {
    t = 1; ledger->reset();
}

void NN_trainer::output_to_file (std::string file_name) {
    if (file_name == "") {
        file_name = net->return_info_string();
        file_name += "-minibatch-" + std::to_string(MINIBATCH_SIZE)
                   + "-alpha=" + std::to_string(ALPHA)
                   + "-lambda=" + std::to_string(LAMBDA)
                   + "-trainsize=" + std::to_string(TRAIN_SET_SIZE)
                   + "-validsize=" + std::to_string(VALID_SET_SIZE)
                   + "-def=" + std::to_string(DEF_LVL)
                   + "-reg=" + std::to_string(REG_TYPE)
                   + "-dropout=" + std::to_string(dropout) + ".txt";
    }
    std::ofstream to (file_name);
    to << ledger->print(1,t-1);
    to.close();
}

float NN_trainer::get_best(Ledger_entry ent) {
    return ledger->get_best(ent);
}


NN_trainer::NN_trainer ( const int TRAIN, const int VALID, const int TEST,
                         const Deformation_level DEF_LVL, const int MINIBATCH_SIZE,
                         std::default_random_engine& gen ) {
    this->TRAIN_SET_SIZE = TRAIN;
    this->VALID_SET_SIZE = VALID;
    this->TEST_SET_SIZE = TEST;
    this->MINIBATCH_SIZE = MINIBATCH_SIZE;
    this->DEF_LVL = DEF_LVL;

    train_set = new GPU_Given_set(TRAIN_SET_SIZE, DEF_LVL, gen);
//    train_set = new GPU_Given_set_with_rounding(TRAIN_SET_SIZE, DEF_LVL, gen);
//    train_set = new GPU_Given_set_expanded(TRAIN_SET_SIZE, gen);
    valid_set = new GPU_Testing_set(VALID_SET_SIZE, DEF_LVL, gen);
    test_set  = new GPU_Testing_set(TEST_SET_SIZE, DEF_LVL, gen);

    ledger = new Ledger(TRAIN, VALID, TEST);
    cudaMalloc ((void**) &dev_bmp_batch,  MINIBATCH_SIZE * sizeof(float*));
    cudaMalloc ((void**) &dev_type_batch, MINIBATCH_SIZE * sizeof(Shape_type*));
}

NN_trainer::~NN_trainer () {
    delete train_set; delete valid_set; delete test_set;
    delete ledger; cudaFree(dev_bmp_batch); cudaFree(dev_type_batch);
}

void copy_pointers (float** bmp_batch, Shape_type** type_batch,
                    float** dev_bmp_batch, Shape_type** dev_type_batch,
                    int MINIBATCH_SIZE) {
    cudaMemcpy(dev_bmp_batch,  bmp_batch,  MINIBATCH_SIZE * sizeof(float*),      cudaMemcpyHostToDevice);
    cudaMemcpy(dev_type_batch, type_batch, MINIBATCH_SIZE * sizeof(Shape_type*), cudaMemcpyHostToDevice);
}

/* void FF_net_trainer::train_conc (int num,  int* epochs,  int* MINIBATCH_SIZES,  bool output_to_file ) {
    std::vector<std::thread> threads;
    FF_net* net_ptr;
    std::vector<FF_net*> nets;
    for (int n = 0; n < num; n++) {
        net_ptr = new FF_net( LAYERS, NODES, gen);
        nets.push_back(net_ptr);
        threads.push_back(std::thread(&FF_net_trainer::train, this, epochs[n],
                     MINIBATCH_SIZES[n], nets[n], L2, true));
    }
    for (std::vector<std::thread>::iterator it = threads.begin(); it != threads.end(); it++)
        (*it).join();
    for (std::vector<FF_net*>::iterator it = nets.begin(); it != nets.end(); it++)
        delete *it;

} */
