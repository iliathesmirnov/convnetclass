
#include "data.hpp"
#include <fstream>
#include <random>
#include <curand.h>

enum Net_type {FF_NET, CONV_NET};
enum Reg_type {L1, L2, L1_AND_L2, NO_REG};
enum Ledger_entry {TPC, TLOSS, VPC, VLOSS};

void copy_pointers (float**, Shape_type**, float**, Shape_type**, int);

struct convdim {
   int w, h, d;
   int vol();
   convdim(int w = 0, int h = 0, int d = 0);
};


// =============================================================================
// =========================== Ledger ==========================================
// =============================================================================

class Ledger {
private:
    int TRAIN_SET_SIZE, VALID_SET_SIZE, TEST_SET_SIZE;
    int cur_item, max_size;
    void expand(int new_max);
public:
    float *tpc, *tloss,   // Training set proportion correct and loss
          *vpc, *vloss,   // Validation set
           *pc,  *loss;   // Testing set
    float *best_tpc, *best_tloss,
          *best_vpc, *best_vloss;
    std::string print(int from, int to);
    float get_best(Ledger_entry ent);
    void end_cur_epoch(int t); void reset();
    Ledger(int TRAIN, int VALID, int TEST, int max_size = 128);
    ~Ledger();
};

// =============================================================================
// =========================== Layers ==========================================
// =============================================================================

class NN_layer {
friend class Neural_net;
friend class FF_net; friend class Conv_net;
protected:
    const float *input, *delta_in;
    float *output, *delta;
    int MINIBATCH_SIZE;
public:
    virtual std::string get_id() = 0;
    virtual void fw_pass() = 0;
    virtual void back_pass() = 0;
    virtual void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) = 0;
    virtual void reset_dw() = 0;
    virtual void set_msize(int MINIBATCH_SIZE) = 0;
    virtual ~NN_layer() {}
};

class FF_layer : public NN_layer {
private:
    static const std::string layer_id;
    int PREV, CUR;
    float *weight, *dw;

    // CUDA data
    int NODES_PER_BLOCK_delta, NODES_PER_BLOCK_dw;
    dim3 dim_grid_delta, dim_block_delta;
    dim3 dim_grid_dw, dim_block_dw;
public:
    std::string get_id() { return FF_layer::layer_id; }
    void fw_pass() override;
    void back_pass() override;
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    void store_weights(std::ofstream& to, float p = 0.0);
    FF_layer (int PREV, int CUR, int MINIBATCH_SIZE,
              curandGenerator_t& gen);
    ~FF_layer ();
};

class ReLU_layer : public NN_layer {
private:
    static const std::string layer_id;
    int CUR; float LEAK;
public:
    std::string get_id() { return ReLU_layer::layer_id; }
    void fw_pass() override;
    void back_pass() override;
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    ReLU_layer(int CUR, int MINIBATCH_SIZE, float LEAK = 0.01);
    ~ReLU_layer();
};

class Softmax_layer : public NN_layer {
private:
    static const std::string layer_id;
    int PREV, CUR, MINIBATCH_SIZE;
    Shape_type** target; int* target_int;
public:
    std::string get_id() { return Softmax_layer::layer_id; }
    float* loss; Shape_type* type;
    void set_targets(Shape_type** target);

    void fw_pass() override;
    void back_pass() override;
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    Softmax_layer(int PREV, int CUR, int MINIBATCH_SIZE);
    ~Softmax_layer();
};

class Maxpool_layer : public NN_layer {
private:
    static const std::string layer_id;
    int window_width, window_stride;
    dim3 grid_dim_in, grid_dim_out;
    int* max_index_loc;
public:
    std::string get_id() { return Maxpool_layer::layer_id; }
    convdim input_dim, output_dim;
    void fw_pass() override;
    void back_pass() override;
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    Maxpool_layer(int width, int stride, convdim input_dim, int MINIBATCH_SIZE);
    ~Maxpool_layer();
};

class Conv_layer : public NN_layer {
private:
    static const std::string layer_id;
    float *weight, *dw;
    int window_width, window_stride;
    int num_filters, pad;
    dim3 grid_dim_in, grid_dim_pad, grid_dim_out,
         grid_dim_weight;
public:
    float *padded_input; // Fix
    std::string get_id() { return Conv_layer::layer_id; }
    convdim input_dim, padded_dim, output_dim;
    void fw_pass() override;
    void back_pass() override;
    void update_weights (const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    void store_weights(std::ofstream& to, float p = 0.0);
    Conv_layer(int width, int stride, int n, int pad,
                convdim input_dim, int MINIBATCH_SIZE, curandGenerator_t& gen);
    ~Conv_layer();
};

class Dropout_layer : public NN_layer {
private:
    static const std::string layer_id;
    int CUR; int MINIBATCH_SIZE;
    float* mask;
    curandGenerator_t* gen;
public:
    float p;
    std::string get_id() { return Dropout_layer::layer_id; }
    static bool training;
    void fw_pass() override;
    void back_pass() override;
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE) override;
    void reset_dw() override;
    void set_msize(int MINIBATCH_SIZE) override;
    Dropout_layer(int CUR, int MINIBATCH_SIZE, float p, curandGenerator_t* gen);
    ~Dropout_layer();
};

// =============================================================================
// ========================== Networks =========================================
// =============================================================================

class Neural_net {
protected:
    int LAYERS, MINIBATCH_SIZE;
    float* input;
    bool dropout;
    NN_layer** layer;
public:
    void load_input (float* input); void load_input (float** bitmaps);
    void fw_pass();
    void back_pass(Shape_type** target);
    void update_weights(const float ALPHA, const float LAMBDA, const Reg_type REG_TYPE);
    void reset_dw();
    void classify(float* in);
    void process_minibatch_acl(std::vector<GPU_Labeled_shape> shape,
                               Dataset_type D_TYPE, Ledger* ledger, int t,
                               float** dev_bmp_batch, Shape_type** dev_type_batch);
    virtual std::string return_info_string() = 0;
    virtual void store_weights(std::ofstream& to) = 0;
    Dropout_layer* drp_layer = 0; Softmax_layer* softmax_head;
    Neural_net(int LAYERS, int MINIBATCH_SIZE, bool dropout = false);
    virtual ~Neural_net();
};

class FF_net : public Neural_net {
private:
    int HIDDEN_BLOCKS; int* NODES;
public:
    std::string return_info_string() override;
    void store_weights(std::ofstream& to) override;
    FF_net (int HIDDEN_BLOCKS, int* NODES, int MINIBATCH_SIZE, curandGenerator_t& gen,
            float LEAK = 0.01);
    FF_net (int HIDDEN_BLOCKS, int* NODES, int MINIBATCH_SIZE,
            curandGenerator_t& gen, bool dropout, float LEAK = 0.01);
    ~FF_net ();
};

class Conv_net : public Neural_net {
private:
    convdim input_dim;
    int CONV_BLOCKS; int *W_WIDTH, *STRIDE, *FILTERS, *PADDING;
    int FF_BLOCKS; int* FF_NODES;
public:
    std::string return_info_string() override;
    void store_weights(std::ofstream& to) override;
    Conv_net (int CONV_BLOCKS, int* W_WIDTH, int* STRIDE, int* FILTERS, int* PADDING,
              int FF_BLOCKS, int* FF_NODES, convdim input_dim, curandGenerator_t& gen,
              int MINIBATCH_SIZE, float LEAK = 0.01);
    Conv_net (int CONV_BLOCKS, int* W_WIDTH, int* STRIDE, int* FILTERS, int* PADDING,
              int FF_BLOCKS, int* FF_NODES, convdim input_dim, curandGenerator_t& gen,
              int MINIBATCH_SIZE, bool dropout, float LEAK = 0.01);
    ~Conv_net ();
};


// =============================================================================
// ========================== Trainer ==========================================
// =============================================================================

class NN_trainer {
private:
    int TRAIN_SET_SIZE, VALID_SET_SIZE, TEST_SET_SIZE;
    int MINIBATCH_SIZE; Deformation_level DEF_LVL;
    GPU_Given_set* train_set;
//    GPU_Given_set_with_rounding* train_set;
//    GPU_Given_set_expanded* train_set;
    GPU_Testing_set* valid_set;
    GPU_Testing_set* test_set;

    int t = 1; Ledger* ledger;
    void find_acl (Dataset_type D_TYPE, Neural_net* net);
    bool dropout;
    // CUDA variables
    float** dev_bmp_batch; Shape_type** dev_type_batch;
public:
    float ALPHA = 0.01, LAMBDA = 0.00;
    Reg_type REG_TYPE = NO_REG;
    Neural_net* net = 0;

    void train (const int epochs, std::default_random_engine& gen, bool dropout = false);
    void output_to_file(std::string file_name = "");
    void reset();
    float get_best(Ledger_entry ent);
    NN_trainer( const int TRAIN, const int VALID, const int TEST,
                const Deformation_level DEF_LVL, const int MINIBATCH_SIZE,
                std::default_random_engine& gen );
    ~NN_trainer();
};
