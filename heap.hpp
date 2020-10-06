struct Labeled_value {
    int label;
    float value;
    Labeled_value(int label = 0, float value = 0);
};

class Labeled_heap {
private:
    int size = 0;
    Labeled_value* entry;

    int parent (int c);
    int lchild (int p);
    int rchild (int p);
    void update (int n);
    void insert (Labeled_value e);
public:
    bool is_empty();
    void print();
    Labeled_value extract_min();
    Labeled_heap (Labeled_value* entry = 0, int size = 0, bool topdown = true);
    Labeled_heap& operator= (const Labeled_heap& h2);
    Labeled_heap (const Labeled_heap& h2);
    ~Labeled_heap();
};
