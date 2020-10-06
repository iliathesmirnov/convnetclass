#include "heap.hpp"
#include<iostream>

//   An implementation of a min-heap with labels

Labeled_value::Labeled_value(int label, float value) {
    this->label = label;
    this->value = value;
}

void swap (Labeled_value& a, Labeled_value& b) {
    Labeled_value temp;
    temp = a;    a = b;        b = temp;
}

int Labeled_heap::parent (int c) {
    if ((c > 0) && (c < size)) return ((c-1)/2);
    else return -1;
}

int Labeled_heap::lchild (int p) {
    int t = 2*p + 1;
    if (t < size) return t;
    else return -1;
}

int Labeled_heap::rchild (int p) {
    int t = 2*p + 2;
    if (t < size) return t;
    else return -1;
}

void Labeled_heap::update (int n) {
    if (n < size) {
        int smallest = n;
        int l = lchild(n),   r = rchild(n);
        if ( (l != -1) && (entry[l].value < entry[n].value)) smallest = l;
        if ( (r != -1) && (entry[r].value < entry[smallest].value)) smallest = r;
        if ( smallest != n) {
            swap (entry[n], entry[smallest]);
            update(smallest);
        }
    }
}

void Labeled_heap::insert (Labeled_value e) {
    int c = size;
    entry[c] = e;
    size++;

    while ( (parent(c) != -1) && (entry[parent(c)].value > entry[c].value) ) {
        swap(entry[parent(c)], entry[c]);
        c = parent(c);
    }
}

bool Labeled_heap::is_empty() {
    if (size == 0) return true;
    else return false;
}

void Labeled_heap::print() {
    for (int i = 0; i < size; i++) std::cout << "label: " << entry[i].label << ", value: " << entry[i].value << ";   ";
    std::cout << std::endl << "========" << std::endl;
}

Labeled_value Labeled_heap::extract_min() {
    Labeled_value min = entry[0];
    swap(entry[0], entry[size-1]);
    size--;
    update(0);
    return min;
}

Labeled_heap::Labeled_heap (Labeled_value* input, int s, bool topdown) {
    entry = new Labeled_value[s];

    if (topdown) {
        size = s;
        for (int i = 0; i < size; i++) entry[i] = input[i];
        for (int i = size/2; i >= 0; i--) update(i);
    }
    else for (int i = 0; i < s; i++) insert(input[i]);
}

Labeled_heap& Labeled_heap::operator= (const Labeled_heap& h2) {
    if (size != h2.size) {
        delete[] entry;
        size = h2.size;
        entry = new Labeled_value[size];
    }
    for (int i = 0; i < size; i++) entry[i] = h2.entry[i];
    return *this;
}

Labeled_heap::Labeled_heap (const Labeled_heap& h2) {
    if (size != h2.size) {
        delete[] entry;
        size = h2.size;
        entry = new Labeled_value[size];
    }
    for (int i = 0; i < size; i++) entry[i] = h2.entry[i];
}

Labeled_heap::~Labeled_heap() {
    delete[] entry;
}

/*
int main() {
    clock_t start = clock();
    double dur;

    const int num = 5;

    Labeled_value input[num];
    for (int i = 0; i < num; i++) {
        input[i].label = i;
        input[i].value = rand() % 3000;
    }


    start = clock();
    Labeled_heap heap(input, num, true);
    heap.print();

    while (!heap.is_empty()) {
        heap.extract_min();
        heap.print();
    } std::cout << std::endl;
    dur = (double) (clock() - start) / CLOCKS_PER_SEC;
    std::cout << dur << std::endl;
    std::cout << "=====" << std::endl;

    start = clock();
    
    heap = Labeled_heap(input, num, false);
    while (!heap.is_empty()) {
        heap.extract_min();
    } std::cout << std::endl;
    dur = (double) (clock() - start) / CLOCKS_PER_SEC;
    std::cout << dur << std::endl;

    return 0;
}*/

