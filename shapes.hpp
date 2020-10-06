#include <random>
#include <vector>
#include <tuple>

#ifndef SHAPES_H
#define SHAPES_H

const int INFTY = 65535;
typedef std::tuple<int,int> coords;

struct Affine_map { // Ax + b, where A is a 2x2 matrix and b is a 2x1 vector
    float a11, a12, a21, a22, b1, b2;
    float det;
    Affine_map (float a11 = 0, float a12 = 0,
            float a21 = 0, float a22 = 0,
            float b1 = 0, float b2 = 0);
};

class Bounding_box {
private:
    float x[4], y[4];
    float xmin = INFTY, xmax = -INFTY, ymin = INFTY, ymax = -INFTY;
    float xl, xh, yl, yh;
    float x_bound = 128, y_bound = 128;

    static std::random_device rdev;
    static std::mt19937 gen;

public:
    void transform(Affine_map T);
    void print_to_file(std::ofstream& to);
    Affine_map gen_translation();

    Bounding_box (float x_0 = 0, float y0 = 0,
              float x1 = 0, float y1 = 0,
              float x2 = 0, float y2 = 0,
              float x3 = 0, float y3 = 0,
              float x_bound = 128, float y_bound = 128);
};

struct Generr{ Generr(); };


class Eq {
public:
    virtual void transform(Affine_map T) = 0;
};

class Linear_Eq : public Eq {     // Ax + By + C
private:
    float A, B, C;
    bool geq;
public:
    void transform (Affine_map T);
    bool in_halfspace(int x, int y);
    Linear_Eq(float A = 0, float B = 0, float C = 0, bool geq = true);
};

class Quadratic_Eq : public Eq { // Ax^2 + Bxy + Cy^2 + Dx + Ey + F
private:
    float A, B, C, D, E, F;
    bool geq;
public:
    void transform (Affine_map T);
    bool in_halfspace(int x, int y);
    Quadratic_Eq (float A = 0, float B = 0, float C = 0, float D = 0,
              float E = 0, float F = 0, bool geq = true);
};

class Cubic_Eq : public Eq {     // Ax^3 + Bx^2y + Cxy^2 + Dy^3 + Ex^2 + Fxy + Gy^2 + Hx + Iy + J
private:
    double A, B, C, D, E, F, G, H, I, J;
    bool geq;
public:
    void transform (Affine_map T);
    bool in_halfspace(int x, int y);
    Cubic_Eq(float A = 0, float B = 0, float C = 0, float D = 0, float E = 0,
          float F = 0, float G = 0, float H = 0, float I = 0, float J = 0,
           bool geq = true);
};


enum Shape_type {TRIANGLE, SQUARE, PENTAGON, HEXAGON,
                 CIRCLE, CIRCLE_II, CIRCLE_III, CIRCLE_IV,
                 RHOMBUS, RHOMBUS_II, RHOMBUS_III, RHOMBUS_IV,
                 HEART, DIAMOND, CLUB, SPADE};
const int NUM_SHAPES = 16;
const int INPUT_DIM = 128;

// Circle:          Disc (filled-in circle)
// Circle_II:       Annular ring
// Circle_III:      Annular ring with an interior circle
// Circle_IV:       Annular ring with an interior annular ring
// Rhombus:         Rhombus with another rhombus cut out from interior
// Rhombus_II:      Rhombus with two triangles cut out from interior
// Rhombus_III:     Rhombus with four triangles cut out from interior
// Rhombus_IV:      Hourglass figure

class Shape {
protected:
    int xmax, ymax; float xmid, ymid;
    Bounding_box* box;
    bool bitmap_generated = false;
public:
    float* bitmap = 0;
    virtual Affine_map gen_translation() = 0;
    virtual void rounden (int r) = 0;
    virtual void transform (Affine_map T) = 0;
    virtual void copy (Shape* s) = 0;
    virtual void gen_bitmap() = 0;
    virtual float find_L2_distance_from (Shape* s2) = 0;
    virtual void print_to_file (std::ofstream& to, bool bbox = true) = 0;
    Shape (int xmax = INPUT_DIM, int ymax = INPUT_DIM);
    virtual ~Shape() {}
};

class CPU_Shape : public Shape {
public:
    Affine_map gen_translation();
    void rounden (int r);
    virtual void transform (Affine_map T) = 0;
    virtual void copy (Shape* s) = 0;
    virtual void gen_bitmap();
    float find_L2_distance_from (Shape* s2);
    void print_to_file (std::ofstream& to, bool bbox = true);
    virtual ~CPU_Shape();
};

std::string name_by_shape_type (Shape_type type);
std::string file_name_by_shape_type (Shape_type type);
CPU_Shape* allocate_mem_by_shape_type (Shape_type type);

class GPU_Shape : public Shape {
public:
    Affine_map gen_translation();
    virtual void transform (Affine_map T) = 0;
    virtual void copy (Shape* s) = 0;
    virtual void gen_bitmap();
    float find_L2_distance_from (Shape* s2);
    void print_to_file (std::ofstream& to, bool bbox = true);

    GPU_Shape ();
    virtual ~GPU_Shape();
};

class Triangle : public CPU_Shape  {
private:
    Linear_Eq L1, L2, L3;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Triangle();
};

class GPU_Triangle : public GPU_Shape {
private:
    Linear_Eq *L1, *L2, *L3;
public:
    void transform (Affine_map T) override;
    void gen_bitmap() override;
    void copy (Shape* s) override;

    GPU_Triangle();
};

class Square : public CPU_Shape  {
private:
    Linear_Eq L1, L2, L3, L4;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy (Shape* s) override;

    Square();
};

class Pentagon : public CPU_Shape  {
private:
    Linear_Eq L1, L2, L3, L4, L5;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Pentagon();
};

class Hexagon : public CPU_Shape  {
private:
    Linear_Eq L1, L2, L3, L4, L5, L6;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Hexagon();
};

class Circle : public CPU_Shape {
private:
    Quadratic_Eq Q1;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Circle();
};

class Circle_II : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Circle_II();
};

class Circle_III : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2, Q3;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Circle_III();
};

class Circle_IV : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2, Q3, Q4;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Circle_IV();
};

class Rhombus : public CPU_Shape {
private:
    Linear_Eq L1, L2, L3, L4, L5, L6, L7, L8;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Rhombus();
};

class Rhombus_II : public CPU_Shape {
private:
    Linear_Eq L1, L2, L3, L4, L5, L6, L7;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Rhombus_II();
};

class Rhombus_III : public CPU_Shape {
private:
    Linear_Eq L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Rhombus_III();
};

class Rhombus_IV : public CPU_Shape {
private:
    Linear_Eq L1, L2, L3, L4, L5, L6;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Rhombus_IV();
};

class Diamond : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2, Q3, Q4;
    Linear_Eq L1, L2, L3, L4;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Diamond();
};

class Club : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2, Q3, Q4, Q5, Q6;
    Linear_Eq L1, L2, L3, L4, L5, L6, L7;
public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Club();
};

class Heart : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2;
    Cubic_Eq C3, C4;
    Linear_Eq L11, L12, L21, L22, L31, L32, L41, L42;

public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Heart();
};

class Spade : public CPU_Shape {
private:
    Quadratic_Eq Q1, Q2, Q5, Q6;
    Cubic_Eq C3, C4;
    Linear_Eq L11, L12, L21, L22, L31, L32, L41, L42;
    Linear_Eq L1, L2, L3;

public:
    void transform(Affine_map T) override;
    void gen_bitmap() override;
    void copy(Shape* s) override;

    Spade();
};

#endif /* SHAPES_H */
