#include "shapes.hpp"
#include <iostream>
#include <fstream>

using namespace std;

Affine_map::Affine_map( float a11, float a12,
            float a21, float a22,
            float b1, float b2) {
    this->a11 = a11;  this->a12 = a12;
    this->a21 = a21;  this->a22 = a22;
    this->b1 = b1;  this->b2 = b2;
    this->det = a11 * a22 - a12 * a21;
}

// Generating error (comes up in the gen_translation function of Bounding_box)
Generr::Generr() {};

// Initializing the static members of Bounding_box
std::random_device Bounding_box::rdev;
std::mt19937 Bounding_box::gen(rdev());

void Bounding_box::transform (Affine_map T) {
    float new_x, new_y;

    for (int i = 0; i < 4; i++) {
        new_x = T.a11*x[i] + T.a12*y[i] + T.b1;
        new_y = T.a21*x[i] + T.a22*y[i] + T.b2;
        x[i] = new_x;
        y[i] = new_y;

        xmax = std::max(xmax, new_x);  xmin = std::min(xmin, new_x);
        ymax = std::max(ymax, new_y);  ymin = std::min(ymin, new_y);
    }

    xh = x_bound - xmax;
    xl = -xmin;
    yh = y_bound - ymax;
    yl = -ymin;
}

void Bounding_box::print_to_file (ofstream& to) {
    for (int i = 0; i < 4; i++)
        to << x[i] << " " << y[i] << ",";
    to << "!";
}

Affine_map Bounding_box::gen_translation() {
    if ( (xl >= xh) || (yl >= yh) ) throw Generr();
        // A translation that moves the shape inside the canvas can't
        // be generated

    uniform_int_distribution<int> xdistr (xl, xh);
    uniform_int_distribution<int> ydistr (yl, yh);
    Affine_map T = Affine_map(1, 0, 0, 1, xdistr(gen), ydistr(gen));
    return T;
}


Bounding_box::Bounding_box (float x0, float y0,
              float x1, float y1,
              float x2, float y2,
              float x3, float y3,
              float x_bound, float y_bound) {
    x[0] = x0;  y[0] = y0;
    x[1] = x1;  y[1] = y1;
    x[2] = x2;  y[2] = y2;
    x[3] = x3;  y[3] = y3;
}


// =====================================================================
// ===================  Equations   ====================================
// =====================================================================


void Linear_Eq::transform (Affine_map T) {
    float new_A = (1/T.det) * ( A * T.a22 - B * T.a21),
          new_B = (1/T.det) * ( -A * T.a12 + B * T.a11),
          new_C = (1/T.det) * ( A*(-T.a22*T.b1 + T.a12*T.b2) + B*(T.a21*T.b1 - T.a11*T.b2)) + C;
    A = new_A;  B = new_B;  C = new_C;
}

bool Linear_Eq::in_halfspace(int x, int y) {
    bool c = ((A*x + B*y + C) >= 0);
    if ((geq && c) || (!geq && !c)) return true;
    else return false;
}

Linear_Eq::Linear_Eq (float A, float B, float C, bool geq) {
    this->A = A;  this->B = B;  this->C = C;
    this->geq = geq;
}

// ===================================================================
void Quadratic_Eq::transform (Affine_map T) {

    float new_A = (1/T.det) * (1/T.det) * (A*T.a22*T.a22
                         - B*T.a21*T.a22
                         + C*T.a21*T.a21),
          new_B = (1/T.det) * (1/T.det) * (-2*A*T.a12*T.a22
                            + B*(T.a11*T.a22 + T.a12*T.a21)
                        - 2*C*(T.a11*T.a21)),
          new_C = (1/T.det) * (1/T.det) * (A*T.a12*T.a12
                         - B*T.a11*T.a12
                         + C*T.a11*T.a11),
          new_D = (1/T.det) * ( (1/T.det) * (2*A*(T.a12*T.a22*T.b2 - T.a22*T.a22*T.b1)
                         + B*(2*T.a21*T.a22*T.b1 - T.a11*T.a22*T.b2 - T.a21*T.a12*T.b2)
                         + 2*C*(T.a11*T.a21*T.b2 - T.a21*T.a21*T.b1))
                         + D*T.a22 - E*T.a21),
          new_E = (1/T.det) * ( (1/T.det) * (2*A*(T.a12*T.a22*T.b1 - T.a12*T.a12*T.b2)
                         + B*(2*T.a11*T.a12*T.b2 - T.a12*T.a21*T.b1 - T.a11*T.a22*T.b1)
                         + 2*C*(T.a11*T.a21*T.b1 - T.a11*T.a11*T.b2))
                         - D*T.a12 + E*T.a11),
          new_F = (1/T.det) * ( (1/T.det) * (A*(T.a22*T.a22*T.b1*T.b1 + T.a12*T.a12*T.b2*T.b2 - 2*T.a12*T.a22*T.b1*T.b2)
                           + B*((T.a11*T.a22 + T.a12*T.a21)*T.b1*T.b2 - T.a21*T.a22*T.b1*T.b1 - T.a11*T.a12*T.b2*T.b2)
                           + C*(T.a21*T.a21*T.b1*T.b1 + T.a11*T.a11*T.b2*T.b2 - 2*T.a11*T.a21*T.b1*T.b2))
                           + D*(-T.a22*T.b1 + T.a12*T.b2) + E*(T.a21*T.b1 - T.a11*T.b2)) + F;

    A = new_A; B = new_B; C = new_C;
    D = new_D; E = new_E; F = new_F;
}

bool Quadratic_Eq::in_halfspace (int x, int y) {
    bool c = ( (A*x*x + B*x*y + C*y*y + D*x + E*y + F) >= 0);
    if ((geq && c) || (!geq && !c)) return true;
    else return false;
}

Quadratic_Eq::Quadratic_Eq (float A, float B, float C, float D, float E, float F, bool geq) {
    this->A = A;  this->B = B;  this->C = C;
    this->D = D;  this->E = E;  this->F = F;
    this->geq = geq;
}

// ===================================================================
void Cubic_Eq::transform (Affine_map T) {
    float new_A = (1/T.det) * (1/T.det) * (1/T.det) * (A*T.a22*T.a22*T.a22                                            // u^3
                              -B*T.a21*T.a22*T.a22
                              +C*T.a21*T.a21*T.a22
                              -D*T.a21*T.a21*T.a21),
          new_B = (1/T.det) * (1/T.det) * (1/T.det) * ((-1)*3*A*T.a12*T.a22*T.a22                                        // u^2 v
                               +B*T.a22*(T.a11*T.a22 + 2*T.a12*T.a21)
                               -C*T.a21*(2*T.a11*T.a22 + T.a12*T.a21)
                               +3*D*T.a11*T.a21*T.a21),
          new_C = (1/T.det) * (1/T.det) * (1/T.det) * (3*A*T.a12*T.a12*T.a22                                         // uv^2
                               -B*T.a12*(2*T.a11*T.a22 + T.a12*T.a21)
                               +C*T.a11*(T.a11*T.a22 + 2*T.a12*T.a21)
                               -3*D*T.a11*T.a11*T.a21),
          new_D = (1/T.det) * (1/T.det) * (1/T.det) * ((-1)*A*T.a12*T.a12*T.a12                                        // v^3
                               +B*T.a11*T.a12*T.a12
                               +(-1)*C*T.a11*T.a11*T.a12
                               +D*T.a11*T.a11*T.a11),
          new_E = (1/T.det) * (1/T.det) * ( (1/T.det) * (3*A*T.a22*T.a22*(T.a12*T.b2 - T.a22*T.b1)                                // u^2
                                +B*T.a22*(3*T.a21*T.a22*T.b1 - T.a11*T.a22*T.b2 - 2*T.a12*T.a21*T.b2)
                                +C*T.a21*(-3*T.a21*T.a22*T.b1 + 2*T.a11*T.a22*T.b2 + T.a12*T.a21*T.b2)
                                +3*D*T.a21*T.a21*(T.a21*T.b1 - T.a11*T.b2))
                         +E*T.a22*T.a22
                         -F*T.a21*T.a22
                         +G*T.a21*T.a21),
          new_F = (1/T.det) * (1/T.det) * ( (1/T.det) * (6*A*T.a12*T.a22*(T.a22*T.b1 - T.a12*T.b2)                                // uv
                                +B*(4*T.a12*T.a22*(T.a11*T.b2 - T.a21*T.b1) + 2*((-1)*T.a11*T.a22*T.a22*T.b1 + T.a12*T.a12*T.a21*T.b2))
                                +C*( 4*T.a11*T.a21*(T.a22*T.b1 - T.a12*T.b2) + 2*(T.a12*T.a21*T.a21*T.b1 - T.a11*T.a11*T.a22*T.b2) )
                                +6*D*T.a11*T.a21*(T.a11*T.b2 - T.a21*T.b1) )
                        -2*E*T.a12*T.a22
                            + F*(T.a11*T.a22 + T.a12*T.a21)
                        - 2*G*(T.a11*T.a21)),
          new_G = (1/T.det) * (1/T.det) * ( (1/T.det) * (3*A*T.a12*T.a12*(T.a12*T.b2 - T.a22*T.b1)                                // v^2
                                 +B*T.a12*(2*T.a11*T.a22*T.b1 + T.a12*T.a21*T.b1 - 3*T.a11*T.a12*T.b2)
                                 +C*T.a11*((-1)*T.a11*T.a22*T.b1 - 2*T.a12*T.a21*T.b1 + 3*T.a11*T.a12*T.b2)
                                 +3*D*T.a11*T.a11*(T.a21*T.b1 - T.a11*T.b2) )
                         +E*T.a12*T.a12
                         -F*T.a11*T.a12
                         +G*T.a11*T.a11),
          new_H = (1/T.det) * ( (1/T.det) * ((1/T.det) * (A*3*T.a22*(T.a22*T.b1 - T.a12*T.b2)*(T.a22*T.b1 - T.a12*T.b2)                    // u
                                 +B*(2*T.a11*T.a22*T.b2*(T.a22*T.b1 - T.a12*T.b2)
                                    + T.a21*(3*T.a22*T.a22*T.b1*T.b1 - T.a12*T.a12*T.b2*T.b2 + 4*T.a12*T.a22*T.b1*T.b2))
                                 +C*(2*T.a12*T.a21*T.b2*(T.a11*T.b2 - T.a21*T.b1)
                                    + T.a22*(3*T.a21*T.a21*T.b1*T.b1 + T.a11*T.a11*T.b2*T.b2 - 4*T.a11*T.a21*T.b1*T.b2))
                                 -D*3*T.a21*(T.a21*T.b1 - T.a11*T.b2)*(T.a21*T.b1 - T.a11*T.b2))
                         + 2*E*(T.a12*T.a22*T.b2 - T.a22*T.a22*T.b1)
                         + F*(2*T.a21*T.a22*T.b1 - T.a11*T.a22*T.b2 - T.a21*T.a12*T.b2)
                         + 2*G*(T.a11*T.a21*T.b2 - T.a21*T.a21*T.b1))
                         + H*T.a22 - I*T.a21),
          new_I = (1/T.det) * ( (1/T.det) * ((1/T.det) * ((-1)*A*3*T.a12*(T.a22*T.b1 - T.a12*T.b2)*(T.a22*T.b1 - T.a12*T.b2)                     // v
                                  +B*(2*T.a12*T.a21*T.b1*(T.a22*T.b1 - T.a12*T.b2)
                                    + T.a11*(T.a22*T.a22*T.b1*T.b1 + 3*T.a12*T.a12*T.b2*T.b2 - 4*T.a12*T.a22*T.b1*T.b2))
                                 +C*(2*T.a11*T.a22*T.b1*(T.a11*T.b2 - T.a21*T.b1)
                                    + T.a12*(T.a21*T.a21*T.b1*T.b1 - 3*T.a11*T.a11*T.b2*T.b2 + 4*T.a11*T.a21*T.b1*T.b2))
                                 +D*3*T.a11*(T.a11*T.b2 - T.a21*T.b1)*(T.a11*T.b2 - T.a21*T.b1))
                         + 2*E*(T.a12*T.a22*T.b1 - T.a12*T.a12*T.b2)
                         + F*(2*T.a11*T.a12*T.b2 - T.a12*T.a21*T.b1 - T.a11*T.a22*T.b1)
                         + 2*G*(T.a11*T.a21*T.b1 - T.a11*T.a11*T.b2))
                         - H*T.a12 + I*T.a11),
          new_J = (1/T.det) * (
                    (1/T.det) * (
                            (1/T.det) * (A*(T.a12*T.b2 - T.a22*T.b1)*(T.a12*T.b2 - T.a22*T.b1)*(T.a12*T.b2 - T.a22*T.b1)        // 1
                                        +B*(T.a22*T.b1 - T.a12*T.b2)*(T.a22*T.b1 - T.a12*T.b2)*(T.a21*T.b1 - T.a11*T.b2)
                                        +C*(T.a11*T.b2 - T.a21*T.b1)*(T.a11*T.b2 - T.a21*T.b1)*(T.a12*T.b2 - T.a22*T.b1)
                                    +D*(T.a21*T.b1 - T.a11*T.b2)*(T.a21*T.b1 - T.a11*T.b2)*(T.a21*T.b1 - T.a11*T.b2)
                                     )
                                   + E*(T.a22*T.b1 - T.a12*T.b2)*(T.a22*T.b1 - T.a12*T.b2)
                                   + F*((T.a11*T.a22 + T.a12*T.a21)*T.b1*T.b2 - T.a21*T.a22*T.b1*T.b1 - T.a11*T.a12*T.b2*T.b2)
                                   + G*(T.a21*T.b1 - T.a11*T.b2)*(T.a21*T.b1 - T.a11*T.b2)
                             )
                    + H*(-T.a22*T.b1 + T.a12*T.b2) + I*(T.a21*T.b1 - T.a11*T.b2)
                   ) + J;

    A = new_A;  B = new_B;  C = new_C;  D = new_D;  E = new_E;
    F = new_F;  G = new_G;  H = new_H;  I = new_I;  J = new_J;
}

bool Cubic_Eq::in_halfspace(int x, int y) {
    bool c = ((A*x*x*x + B*x*x*y + C*x*y*y + D*y*y*y + E*x*x + F*x*y + G*y*y + H*x + I*y + J) >= 0);
    if ((geq && c) || (!geq && !c)) return true;
    else return false;
}

Cubic_Eq::Cubic_Eq (float A, float B, float C, float D, float E,
            float F, float G, float H, float I, float J, bool geq) {
    this->A = A;  this->B = B;  this->C = C;  this->D = D;  this->E = E;
    this->F = F;  this->G = G;  this->H = H;  this->I = I;  this->J = J;
    this->geq = geq;
}

// =====================================================================
// ===================  Shapes   =======================================
// =====================================================================

Shape::Shape (int xmax, int ymax) {
    this->xmax = xmax; this->ymax = ymax;
    this->xmid = xmax / 2; this->ymid = ymax / 2;
}

// =================== CPU Shape =======================================
Affine_map CPU_Shape::gen_translation() {
    return box->gen_translation();
}

void CPU_Shape::rounden (int r) {
    int i, j, a, b, c;
    std::vector<coords> coord_stack;
    for (i = r; i < INPUT_DIM-r; i++) {
        for (j = r; j < INPUT_DIM-r; j++) {
            c = i + j*INPUT_DIM;
            if (bitmap[c] == 1)
                coord_stack.push_back( std::make_tuple(i,j) );
        }
    }

    for (std::vector<coords>::iterator it = coord_stack.begin(); it != coord_stack.end(); it++) {
        i = std::get<0>(*it);
        j = std::get<1>(*it);

        for (a = -r; a <= r; a++) {
            for (b = -r; b <= r; b++) {
                if (a*a + b*b <= r*r) {
                    c = (i+a) + (j+b)*INPUT_DIM;
                    bitmap[c] = 1;
                }
            }
        }
    }
}

void CPU_Shape::print_to_file(ofstream& to, bool bbox) {
    for (int i = 0; i < xmax*ymax; i++) {
        to << bitmap[i];
    } to << ",";
    if (bbox) box->print_to_file(to);
}

float CPU_Shape::find_L2_distance_from (Shape* s2_in) {
    CPU_Shape* s2 = static_cast<CPU_Shape*>(s2_in);
    float dist = 0.0;
    if ( (xmax != s2->xmax) || (ymax != s2->ymax) ) {
        cout << "The shapes have different dimensions" << endl;
    }
    else {
        for (int i = 0; i < xmax * ymax; i++) {
            dist += (float) ( (bitmap[i] - s2->bitmap[i])*(bitmap[i] - s2->bitmap[i]) );
        }
    }
    return dist;
}

void CPU_Shape::gen_bitmap() {
    if (!bitmap_generated) {
        bitmap = new float[xmax * ymax];
        bitmap_generated = true;
    }
}

CPU_Shape::~CPU_Shape() {
    if (bitmap_generated) delete[] bitmap;
    delete box;
}

// =================== GPU Shape =======================================

Affine_map GPU_Shape::gen_translation () { Affine_map T; return T; }
float GPU_Shape::find_L2_distance_from (Shape* s2) { return 0.0; }
void GPU_Shape::print_to_file (ofstream& to, bool bbox) {}

void GPU_Shape::gen_bitmap() {
    if (!bitmap_generated) {
        cudaMalloc( (void**) &bitmap, xmax * ymax * sizeof(float));
        bitmap_generated = true;
    }
}

GPU_Shape::GPU_Shape () {
    cudaMalloc ((void**) &box, sizeof(Bounding_box));
}

GPU_Shape::~GPU_Shape () {
    cudaFree(bitmap); cudaFree(box);
}

// ========================= Triangle =============================
void Triangle::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    box->transform(T);
}

void Triangle::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Triangle::copy (Shape* s) {
    Triangle* t = dynamic_cast<Triangle*>(s);
    this->L1 = t->L1;  this->L2 = t->L2;  this->L3 = t->L3;
    *this->box = *t->box;
}

Triangle::Triangle() {
    L1 = Linear_Eq(1, 0.577 * xmid / ymid, -1.289 * xmid, true);
    L2 = Linear_Eq(1, -0.577 * xmid / ymid, -0.711 * xmid, false);
    L3 = Linear_Eq(0, 1, -3*ymid/2, false);
    box = new Bounding_box( xmid - 0.577*ymid-1,  ymid/2-1,
                xmid + 0.577*ymid+1,  ymid/2-1,
                xmid + 0.577*ymid-1,  3*ymid/2+1,
                xmid - 0.577*ymid+1,  3*ymid/2+1);
}


void GPU_Triangle::transform (Affine_map T) {
    L1->transform(T); L2->transform(T); L3->transform(T);
    box->transform(T);
}
void GPU_Triangle::gen_bitmap () {}
void GPU_Triangle::copy (Shape* s) {}

GPU_Triangle::GPU_Triangle() {
    Linear_Eq *L1_host = new Linear_Eq(1, 0.577 * xmid / ymid, -1.289 * xmid, true),
              *L2_host = new Linear_Eq(1, -0.577 * xmid / ymid, -0.711 * xmid, false),
              *L3_host = new Linear_Eq(0, 1, -3*ymid/2, false);
    Bounding_box* box_host = new Bounding_box( xmid - 0.577*ymid-1,  ymid/2-1,
                         xmid + 0.577*ymid+1,  ymid/2-1,
                         xmid + 0.577*ymid-1,  3*ymid/2+1,
                         xmid - 0.577*ymid+1,  3*ymid/2+1);
    cudaMemcpy (L1, L1_host, sizeof(Linear_Eq), cudaMemcpyHostToDevice);
    cudaMemcpy (L2, L2_host, sizeof(Linear_Eq), cudaMemcpyHostToDevice);
    cudaMemcpy (L3, L3_host, sizeof(Linear_Eq), cudaMemcpyHostToDevice);
    cudaMemcpy (box, box_host, sizeof(Bounding_box), cudaMemcpyHostToDevice);
    delete L1_host; delete L2_host; delete L3_host; delete box_host;
}

// ========================= Square =============================
void Square::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);  L4.transform(T);
    box->transform(T);
}

void Square::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) &&
             L4.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Square::copy (Shape* s) {
    Square* sq = dynamic_cast<Square*>(s);
    this->L1 = sq->L1;  this->L2 = sq->L2;  this->L3 = sq->L3;  this->L4 = sq->L4;
    *this->box = *sq->box;
}

Square::Square() {
    L1 = Linear_Eq(1, 0, -xmid/2, true);
    L2 = Linear_Eq(1, 0, -3*xmid/2, false);
    L3 = Linear_Eq(0, 1, -ymid/2, true);
    L4 = Linear_Eq(0, 1, -3*ymid/2, false);
    box = new Bounding_box( xmid/2-1,   ymid/2-1,
                3*xmid/2+1, ymid/2-1,
                3*xmid/2+1, 3*ymid/2+1,
                xmid/2-1,   3*ymid/2+1);
}

// ========================= Pentagon =============================
void Pentagon::transform (Affine_map T) {
    box->transform(T);
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    L4.transform(T);  L5.transform(T);
}

void Pentagon::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) &&
             L4.in_halfspace(x,y) &&
             L5.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Pentagon::copy (Shape* s) {
    Pentagon* p = dynamic_cast<Pentagon*>(s);
    this->L1 = p->L1;  this->L2 = p->L2;  this->L3 = p->L3;  this->L4 = p->L4;
    this->L5 = p->L5;
    *this->box = *p->box;
}

Pentagon::Pentagon() {
    L1 = Linear_Eq(1.376, -1, -2.065 * xmid + ymid, false);
    L2 = Linear_Eq(-0.325, -1, ymid - 0.1 * xmid, false);
    L3 = Linear_Eq(1, 0, -0.595*xmid, true );
    L4 = Linear_Eq(0.325, -1, ymid + 0.1 * xmid, true);
    L5 = Linear_Eq(-1.376, -1, 2.065 * xmid + ymid, true);
    box = new Bounding_box( 0.595*xmid,   ymid/2,
                3*xmid/2, ymid/2,
                3*xmid/2, 3*ymid/2,
                0.595*xmid,   3*ymid/2);

    // The default pentagon defined as above is slightly too small
    Affine_map T = Affine_map(1.3, 0, 0, 1.3, -20, -20);
    transform(T);
}

// ========================= Hexagon =============================
void Hexagon::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    L4.transform(T);  L5.transform(T);  L6.transform(T);
    box->transform(T);
}

void Hexagon::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) &&
             L4.in_halfspace(x,y) &&
             L5.in_halfspace(x,y) &&
             L6.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Hexagon::copy (Shape* s) {
    Hexagon* h = dynamic_cast<Hexagon*>(s);
    this->L1 = h->L1;  this->L2 = h->L2;  this->L3 = h->L3;  this->L4 = h->L4;
    this->L5 = h->L5;  this->L6 = h->L6;
    *this->box = *h->box;
}

Hexagon::Hexagon() {
    L1 = Linear_Eq(1, 0, -1.577*xmid, false);
    L2 = Linear_Eq(0.577*ymid/xmid, -1, -0.244*ymid, false);
    L3 = Linear_Eq(-0.577*ymid/xmid, -1, 0.911 * ymid, false);
    L4 = Linear_Eq(1, 0, -0.423*xmid, true);
    L5 = Linear_Eq(0.577*ymid/xmid, -1, 1.089 * ymid, true);
    L6 = Linear_Eq(-0.577*ymid/xmid, -1, 2.244 * ymid, true);
    box = new Bounding_box( 0.423*xmid,   ymid/3,
                1.577*xmid+1,  ymid/3,
                1.577*xmid+1, 5*ymid/3+1,
                0.423*xmid,   5*ymid/3+1);
}

// ========================= Circle =============================
void Circle::transform (Affine_map T) {
    Q1.transform(T);
    box->transform(T);
}

void Circle::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( Q1.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Circle::copy (Shape* s) {
    Circle* c = dynamic_cast<Circle*>(s);
    this->Q1 = c->Q1;
    *this->box = *c->box;
}

Circle::Circle () {
    Q1 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.75*ymid*ymid, false);
    box = new Bounding_box( 0.5*xmid,   0.5*ymid,
                3*xmid/2+1,  0.5*ymid,
                3*xmid/2+1, 3*ymid/2+1,
                0.5*xmid,   3*ymid/2+1);
}

// ========================= Circle_II =============================
void Circle_II::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);
    box->transform(T);
}

void Circle_II::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( Q1.in_halfspace(x,y) &&
            !Q2.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Circle_II::copy (Shape* s) {
    Circle_II* c = dynamic_cast<Circle_II*>(s);
    this->Q1 = c->Q1;  this->Q2 = c->Q2;
    *this->box = *c->box;
}

Circle_II::Circle_II () {
    Q1 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.75*ymid*ymid, false);
    Q2 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.9*ymid*ymid, false);
    box = new Bounding_box( 0.5*xmid,   0.5*ymid,
                3*xmid/2+1,  0.5*ymid,
                3*xmid/2+1, 3*ymid/2+1,
                0.5*xmid,   3*ymid/2+1);
}

// ========================= Circle_III =============================
void Circle_III::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);  Q3.transform(T);
    box->transform(T);
}

void Circle_III::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( Q1.in_halfspace(x,y) &&
            (!Q2.in_halfspace(x,y) || Q3.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Circle_III::copy (Shape* s) {
    Circle_III* c = dynamic_cast<Circle_III*>(s);
    this->Q1 = c->Q1;  this->Q2 = c->Q2;  this->Q3 = c->Q3;
    *this->box = *c->box;
}

Circle_III::Circle_III () {
    Q1 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.75*ymid*ymid, false);
    Q2 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.9*ymid*ymid, false);
    Q3 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.95*ymid*ymid, false);
    box = new Bounding_box( 0.5*xmid,   0.5*ymid,
                3*xmid/2+1,  0.5*ymid,
                3*xmid/2+1, 3*ymid/2+1,
                0.5*xmid,   3*ymid/2+1);
}

// ========================= Circle_IV =============================
void Circle_IV::transform (Affine_map T) {
    box->transform(T);
    Q1.transform(T);  Q2.transform(T);  Q3.transform(T);  Q4.transform(T);
}

void Circle_IV::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( Q1.in_halfspace(x,y) &&
            (!Q2.in_halfspace(x,y) || ( Q3.in_halfspace(x,y) && !Q4.in_halfspace(x,y) )) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Circle_IV::copy (Shape* s) {
    Circle_IV* c = dynamic_cast<Circle_IV*>(s);
    this->Q1 = c->Q1;  this->Q2 = c->Q2;  this->Q3 = c->Q3;  this->Q4 = c->Q4;
    *this->box = *c->box;
}

Circle_IV::Circle_IV () {
    Q1 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.75*ymid*ymid, false);
    Q2 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.9*ymid*ymid, false);
    Q3 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.95*ymid*ymid, false);
    Q4 = Quadratic_Eq(1, 0, 1, -2*xmid, -2*ymid, xmid*xmid + 0.995*ymid*ymid, false);
    box = new Bounding_box( 0.5*xmid,   0.5*ymid,
                3*xmid/2+1,  0.5*ymid,
                3*xmid/2+1, 3*ymid/2+1,
                0.5*xmid,   3*ymid/2+1);
}

// ========================= Rhombus =============================
void Rhombus::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);  L4.transform(T);
    L5.transform(T);  L6.transform(T);  L7.transform(T);  L8.transform(T);
    box->transform(T);
}

void Rhombus::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) &&
             L4.in_halfspace(x,y) &&
           !(L5.in_halfspace(x,y) &&
             L6.in_halfspace(x,y) &&
             L7.in_halfspace(x,y) &&
             L8.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Rhombus::copy (Shape* s) {
    Rhombus* r = dynamic_cast<Rhombus*>(s);
    this->L1 = r->L1;  this->L2 = r->L2;  this->L3 = r->L3;  this->L4 = r->L4;
    this->L5 = r->L5;  this->L6 = r->L6;  this->L7 = r->L7;  this->L8 = r->L8;
    *this->box = *r->box;
}

Rhombus::Rhombus () {
    L1 = Linear_Eq(1, 1, -xmid - ymid/3, true);
    L2 = Linear_Eq(1, -1, -xmid+ymid/3, false);
    L3 = Linear_Eq(1, 1, -xmid-5*ymid/3, false);
    L4 = Linear_Eq(1,-1,-xmid+5*ymid/3, true);
    L5 = Linear_Eq(1, 1, -xmid - 2*ymid/3, true);
    L6 = Linear_Eq(1, -1, -xmid+2*ymid/3 , false);
    L7 = Linear_Eq(1, 1, -xmid-4*ymid/3, false);
    L8 = Linear_Eq(1, -1, -xmid + 4*ymid/3, true);
    box = new Bounding_box( xmid/3,   ymid/3,
                5*xmid/3+1,  ymid/3,
                5*xmid/3+1, 5*ymid/3+1,
                xmid/3,   5*ymid/3+1);
}

// ========================= Rhombus_II =============================
void Rhombus_II::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);  L4.transform(T);
    L5.transform(T);  L6.transform(T);  L7.transform(T);
    box->transform(T);
}

void Rhombus_II::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) &&
             L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) &&
             L4.in_halfspace(x,y) &&
           !(L5.in_halfspace(x,y) &&
             L6.in_halfspace(x,y) &&
             L7.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Rhombus_II::copy (Shape* s) {
    Rhombus_II* r = dynamic_cast<Rhombus_II*>(s);
    this->L1 = r->L1;  this->L2 = r->L2;  this->L3 = r->L3;  this->L4 = r->L4;
    this->L5 = r->L5;  this->L6 = r->L6;  this->L7 = r->L7;
    *this->box = *r->box;
}

Rhombus_II::Rhombus_II () {
    L1 = Linear_Eq(1, 1, -xmid - ymid/3, true);
    L2 = Linear_Eq(1, -1, -xmid+ymid/3, false);
    L3 = Linear_Eq(1, 1, -xmid-5*ymid/3, false);
    L4 = Linear_Eq(1,-1,-xmid+5*ymid/3, true);
    L5 = Linear_Eq(1, 1, -xmid - ymid/2, true);
    L6 = Linear_Eq(1, -1, -xmid + 3*ymid/2, true);
    L7 = Linear_Eq(1, 0, -xmid, false);

    box = new Bounding_box( xmid/3,   ymid/3,
                5*xmid/3+1,  ymid/3,
                5*xmid/3+1, 5*ymid/3+1,
                xmid/3,   5*ymid/3+1);
}

// ========================= Rhombus_III =============================
void Rhombus_III::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    L4.transform(T);  L5.transform(T);  L6.transform(T);
    L7.transform(T);  L8.transform(T);  L9.transform(T);
    L10.transform(T);  L11.transform(T);  L12.transform(T);
    box->transform(T);
}

void Rhombus_III::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( L1.in_halfspace(x,y) && L2.in_halfspace(x,y) && L3.in_halfspace(x,y) && L4.in_halfspace(x,y) &&
           !(L5.in_halfspace(x,y) && L6.in_halfspace(x,y) && L9.in_halfspace(x,y)) &&
           !(L5.in_halfspace(x,y) && L8.in_halfspace(x,y) && L10.in_halfspace(x,y)) &&
           !(L7.in_halfspace(x,y) && L6.in_halfspace(x,y) && L11.in_halfspace(x,y)) &&
           !(L7.in_halfspace(x,y) && L8.in_halfspace(x,y) && L12.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Rhombus_III::copy (Shape* s) {
    Rhombus_III* r = dynamic_cast<Rhombus_III*>(s);
    this->L1 = r->L1;  this->L2 = r->L2;  this->L3 = r->L3;  this->L4 = r->L4;
    this->L5 = r->L5;  this->L6 = r->L6;  this->L7 = r->L7;  this->L8 = r->L8;
    this->L9 = r->L9;  this->L10 = r->L10;  this->L11 = r->L11;  this->L12 = r->L12;
    *this->box = *r->box;
}

Rhombus_III::Rhombus_III () {
    L1 = Linear_Eq(1, 1, -xmid - ymid/3, true);
    L2 = Linear_Eq(1, -1, -xmid+ymid/3, false);
    L3 = Linear_Eq(1, 1, -xmid-5*ymid/3, false);
    L4 = Linear_Eq(1,-1,-xmid+5*ymid/3, true);

    L5 = Linear_Eq(1, 0, -15*xmid/16, false);
    L6 = Linear_Eq(0, 1, -15*ymid/16, false);
    L7 = Linear_Eq(1, 0, -17*xmid/16, true);
    L8 = Linear_Eq(0, 1, -17*ymid/16, true);

    L9 = Linear_Eq(1, 1, -xmid - ymid/2, true);
    L10 = Linear_Eq(1, -1, -xmid + 3*ymid/2, true);
    L11 = Linear_Eq(1, -1, -xmid+ymid/2 , false);
    L12 = Linear_Eq(1, 1, -xmid-3*ymid/2, false);

    box = new Bounding_box( xmid/3,   ymid/3,
                5*xmid/3+1,  ymid/3,
                5*xmid/3+1, 5*ymid/3+1,
                xmid/3,   5*ymid/3+1);
}

// ========================= Rhombus_IV =============================
void Rhombus_IV::transform (Affine_map T) {
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    L4.transform(T);  L5.transform(T);  L6.transform(T);
    box->transform(T);
}

void Rhombus_IV::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( (L1.in_halfspace(x,y) &&
              L2.in_halfspace(x,y) &&
              L3.in_halfspace(x,y)) ||
             (L4.in_halfspace(x,y) &&
              L5.in_halfspace(x,y) &&
              L6.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Rhombus_IV::copy (Shape* s) {
    Rhombus_IV* r = dynamic_cast<Rhombus_IV*>(s);
    this->L1 = r->L1;  this->L2 = r->L2;  this->L3 = r->L3;  this->L4 = r->L4;
    this->L5 = r->L5;  this->L6 = r->L6;
    *this->box = *r->box;
}

Rhombus_IV::Rhombus_IV () {
    L1 = Linear_Eq(1, -1, ymid - xmid, true);
    L2 = Linear_Eq(1, 1, -ymid - xmid, false);
    L3 = Linear_Eq(0, 1, -ymid/2, true);
    L4 = Linear_Eq(1, -1, ymid-xmid, false);
    L5 = Linear_Eq(1, 1, -ymid - xmid, true);
    L6 = Linear_Eq(0, 1, -3*ymid/2, false);

    box = new Bounding_box( xmid/2,   ymid/2,
                3*xmid/2+1,  ymid/2,
                3*xmid/2+1, 3*ymid/2+1,
                xmid/2,   3*ymid/2+1);
}

// ========================= Diamond =============================
void Diamond::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);  Q3.transform(T);  Q4.transform(T);
    L1.transform(T);  L2.transform(T);  L3.transform(T);  L4.transform(T);
    box->transform(T);
}

void Diamond::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( Q1.in_halfspace(x,y) && Q2.in_halfspace(x,y) &&
             Q3.in_halfspace(x,y) && Q4.in_halfspace(x,y) &&
             L1.in_halfspace(x,y) && L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) && L4.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Diamond::copy (Shape* s) {
    Diamond* d = dynamic_cast<Diamond*>(s);
    this->Q1 = d->Q1;  this->Q2 = d->Q2;  this->Q3 = d->Q3;  this->Q4 = d->Q4;
    this->L1 = d->L1;  this->L2 = d->L2;  this->L3 = d->L3;  this->L4 = d->L4;
    *this->box = *d->box;
}

Diamond::Diamond () {
    float x1 = xmid/4,   A1 = -8*ymid/(3*xmid*(3*xmid - 4*x1)), B1 = ymid/3 - A1 * (xmid-x1)*(xmid-x1),
          x2 = 7*xmid/4, A2 = 8*ymid/(3*xmid*(5*xmid - 4*x2)),  B2 = ymid/3 - A2 * (xmid-x2)*(xmid-x2),
          x3 = x1,       A3 = -A1,                              B3 = 5*ymid/3 - A3 * (xmid-x3)*(xmid-x3),
          x4 = x2,       A4 = -A2,                              B4 = 5*ymid/3 - A4 * (xmid-x4)*(xmid-x4);

    Q1 = Quadratic_Eq(A1, 0, 0, -2*A1*x1, -1, A1*x1*x1 + B1, false);
    Q2 = Quadratic_Eq(A2, 0, 0, -2*A2*x2, -1, A2*x2*x2 + B2, false);
    Q3 = Quadratic_Eq(A3, 0, 0, -2*A3*x3, -1, A3*x3*x3 + B3, true);
    Q4 = Quadratic_Eq(A4, 0, 0, -2*A4*x4, -1, A4*x4*x4 + B4, true);

    L1 = Linear_Eq(1, 0, -xmid/2, true);
    L2 = Linear_Eq(1, 0, -3*xmid/2, false);
    L3 = Linear_Eq(0, 1, -ymid/3, true);
    L4 = Linear_Eq(0, 1, -5*ymid/3, false);

    box = new Bounding_box( xmid/2,   ymid/3,
                3*xmid/2+1,  ymid/3,
                3*xmid/2+1, 5*ymid/3+1,
                xmid/2,   5*ymid/3+1);

}

// ========================= Club =============================
void Club::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);  Q3.transform(T);  Q4.transform(T);  Q5.transform(T);
    L1.transform(T);  L2.transform(T);  L3.transform(T);  L4.transform(T);
    L5.transform(T);  L6.transform(T);  L7.transform(T);
    box->transform(T);
}

void Club::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( (Q1.in_halfspace(x,y) || Q2.in_halfspace(x,y) || Q3.in_halfspace(x,y) ||
           (Q4.in_halfspace(x,y) && Q5.in_halfspace(x,y) && L5.in_halfspace(x,y) && L6.in_halfspace(x,y) && L7.in_halfspace(x,y))) &&
             L1.in_halfspace(x,y) && L2.in_halfspace(x,y) &&
             L3.in_halfspace(x,y) && L4.in_halfspace(x,y) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Club::copy (Shape* s) {
    Club* c = dynamic_cast<Club*>(s);
    this->Q1 = c->Q1;  this->Q2 = c->Q2;  this->Q3 = c->Q3;  this->Q4 = c->Q4;  this->Q5 = c->Q5;
    this->L1 = c->L1;  this->L2 = c->L2;  this->L3 = c->L3;  this->L4 = c->L4;
    this->L5 = c->L5;  this->L6 = c->L6;  this->L7 = c->L7;
    *this->box = *c->box;
}

Club::Club () {
    float  A = - 9*ymid/(2*xmid*xmid), B = 7*ymid/4,  x0 = 2*xmid/3,  x1 = 4*xmid/3;

    // Three circles
    Q1 = Quadratic_Eq(1, 0, 1, -2*xmid, -5*ymid/4, xmid*xmid + 161*ymid*ymid/576, false);
    Q2 = Quadratic_Eq(1, 0, 1, -3*xmid/2, -9*ymid/4, 9*xmid*xmid/16 + 665*ymid*ymid/576, false);
    Q3 = Quadratic_Eq(1, 0, 1, -5*xmid/2, -9*ymid/4, 25*xmid*xmid/16 + 665*ymid*ymid/576, false);

    // The stem
    Q4 = Quadratic_Eq(A, 0, 0, -2*A*x0, -1, A*x0*x0 + B, false);
    Q5 = Quadratic_Eq(A, 0, 0, -2*A*x1, -1, A*x1*x1 + B, false);
    L5 = Linear_Eq(1, 0, -2*xmid/3, true);
    L6 = Linear_Eq(1, 0, -4*xmid/3, false);
    L7 = Linear_Eq(0, 1, -5*ymid/3, false);

    // Boundaries
    L1 = Linear_Eq(1, 0, -5*xmid/12, true);
    L2 = Linear_Eq(1, 0, -19*xmid/12, false);
    L3 = Linear_Eq(0, 1, -7*ymid/24, true);
    L4 = Linear_Eq(0, 1, -5*ymid/3, false);

    box = new Bounding_box( 5*xmid/12,   7*ymid/24,
                19*xmid/12+1,  7*ymid/24,
                19*xmid/12+1, 5*ymid/3+1,
                5*xmid/12,   5*ymid/3+1);
}

// ========================= Heart =============================
void Heart::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);  C3.transform(T);  C4.transform(T);
    L11.transform(T);  L12.transform(T);  L21.transform(T);  L22.transform(T);
    L31.transform(T);  L32.transform(T);  L41.transform(T);  L42.transform(T);
    box->transform(T);
}

void Heart::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( (Q1.in_halfspace(x,y) && L11.in_halfspace(x,y) && L12.in_halfspace(x,y)) ||
             (Q2.in_halfspace(x,y) && L21.in_halfspace(x,y) && L22.in_halfspace(x,y)) ||
             (C3.in_halfspace(x,y) && L31.in_halfspace(x,y) && L32.in_halfspace(x,y)) ||
             (C4.in_halfspace(x,y) && L41.in_halfspace(x,y) && L42.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Heart::copy (Shape* s) {
    Heart* h = dynamic_cast<Heart*>(s);
    this->Q1 = h->Q1;  this->Q2 = h->Q2;  this->C3 = h->C3;  this->C4 = h->C4;
    this->L11 = h->L11;  this->L12 = h->L12;  this->L21 = h->L21;  this->L22 = h->L22;
    this->L31 = h->L31;  this->L32 = h->L32;  this->L41 = h->L41;  this->L42 = h->L42;
    *this->box = *h->box;
}

Heart::Heart () {
    float A = xmid/3 + xmid/25,
          B = ymid/2,
          C = (5*ymid) / (4*xmid*((2*xmid/5)*(2*xmid/5) + 999)),
          D = 11*ymid/8,
          x0 = 2*xmid/3 + xmid/25,   y0 = 7*ymid/8,
          x1 = 4*xmid/3 - xmid/25,   y1 = 7*ymid/8;

    Q1 = Quadratic_Eq(B*B, 0, A*A, -B*B*2*x0, -A*A*2*y0, B*B*x0*x0 + A*A*y0*y0 - A*A*B*B, false);
    Q2 = Quadratic_Eq(B*B, 0, A*A, -B*B*2*x1, -A*A*2*y1, B*B*x1*x1 + A*A*y1*y1 - A*A*B*B, false);

        // C3, non-expanded: ( ( C * (x - 2*xm/3) * 0.001 * ( (x-2*xm/3)*(x-2*xm/3) + 999) + D - y ) >= 0 );

    C3 = Cubic_Eq(C, 0, 0, 0, -2*C*xmid, 0, 0, C*(1.333*xmid*xmid+999), -1, D - C*(8*xmid*xmid*xmid/27 + 666*xmid), true);
        //    x^3           x^2      xy y^2     x                    y                    1
    C4 = Cubic_Eq(-C, 0, 0, 0, 4*C*xmid, 0, 0, -C*(16*xmid*xmid/3 + 999), -1, D + C*(64*xmid*xmid*xmid/27 + 1332*xmid), true);


    L11 = Linear_Eq(0, 1, -ymid, false);
    L12 = Linear_Eq(1, 0, -xmid, false);
    L21 = L11;
    L22 = Linear_Eq(1, 0, -xmid, true);
    L31 = Linear_Eq(0, 1, -ymid, true);
    L32 = L12;
    L41 = L31;
    L42 = L22;


    float ybot = C*xmid/3*( xmid*xmid/9 + 999 ) + D;

    box = new Bounding_box( x0-A,   y0-B,
                x1+A+1, y0-B,
                x1+A+1, ybot+1,
                x0-A,   ybot+1);

}

// ========================= Spade =============================
void Spade::transform (Affine_map T) {
    Q1.transform(T);  Q2.transform(T);  C3.transform(T);  C4.transform(T);
    L11.transform(T);  L12.transform(T);  L21.transform(T);  L22.transform(T);
    L31.transform(T);  L32.transform(T);  L41.transform(T);  L42.transform(T);
    Q5.transform(T);  Q6.transform(T);
    L1.transform(T);  L2.transform(T);  L3.transform(T);
    box->transform(T);
}

void Spade::gen_bitmap() {
    CPU_Shape::gen_bitmap();
    int x, y;

    for (int i = 0; i < xmax*ymax; i++) {
        x = (i % xmax) * 1 ;
        y = (i / xmax) * 1 ;
        if ( (Q1.in_halfspace(x,y) && L11.in_halfspace(x,y) && L12.in_halfspace(x,y)) ||
             (Q2.in_halfspace(x,y) && L21.in_halfspace(x,y) && L22.in_halfspace(x,y)) ||
             (C3.in_halfspace(x,y) && L31.in_halfspace(x,y) && L32.in_halfspace(x,y)) ||
             (C4.in_halfspace(x,y) && L41.in_halfspace(x,y) && L42.in_halfspace(x,y)) ||
             (Q5.in_halfspace(x,y) && Q6.in_halfspace(x,y) && L1.in_halfspace(x,y) && L2.in_halfspace(x,y) && L3.in_halfspace(x,y)) )
            bitmap[i] = 1;
        else bitmap[i] = 0;
    }
}

void Spade::copy (Shape* s) {
    Spade* h = dynamic_cast<Spade*>(s);
    this->Q1 = h->Q1;  this->Q2 = h->Q2;  this->C3 = h->C3;  this->C4 = h->C4;
    this->Q5 = h->Q5;  this->Q6 = h->Q6;
    this->L11 = h->L11;  this->L12 = h->L12;  this->L21 = h->L21;  this->L22 = h->L22;
    this->L31 = h->L31;  this->L32 = h->L32;  this->L41 = h->L41;  this->L42 = h->L42;
    this->L1 = h->L1;  this->L2 = h->L2;  this->L3 = h->L3;
    *this->box = *h->box;
}

Spade::Spade () {
    // The upside-down heart
    float A = xmid/3 + xmid/25,
          B = 2*ymid/5,
          C = (5*ymid) / (4*xmid*((2*xmid/5)*(2*xmid/5) + 999)),
          D = 11*ymid/8 - 2*ymid,
          x0 = 2*xmid/3 + xmid/25,   y0 = 17*ymid/16,        // Centers of the two ellipses
          x1 = 4*xmid/3 - xmid/25,   y1 = 17*ymid/16;

    Q1 = Quadratic_Eq(B*B, 0, A*A, -B*B*2*x0, -A*A*2*y0, B*B*x0*x0 + A*A*y0*y0 - A*A*B*B, false);
    Q2 = Quadratic_Eq(B*B, 0, A*A, -B*B*2*x1, -A*A*2*y1, B*B*x1*x1 + A*A*y1*y1 - A*A*B*B, false);
    C3 = Cubic_Eq(C, 0, 0, 0, -2*C*xmid, 0, 0, C*(1.333*xmid*xmid+999), 1, D - C*(8*xmid*xmid*xmid/27 + 666*xmid), true);
        //    x^3           x^2      xy y^2     x                    y                    1
    C4 = Cubic_Eq(-C, 0, 0, 0, 4*C*xmid, 0, 0, -C*(16*xmid*xmid/3 + 999), 1, D + C*(64*xmid*xmid*xmid/27 + 1332*xmid), true);

    L11 = Linear_Eq(0, 1, -ymid, true);
    L12 = Linear_Eq(1, 0, -xmid, false);
    L21 = L11;
    L22 = Linear_Eq(1, 0, -xmid, true);
    L31 = Linear_Eq(0, 1, -ymid, false);
    L32 = L12;
    L41 = L31;
    L42 = L22;


    // The stem
    float  E = - 9*ymid/(2*xmid*xmid), F = 7*ymid/4,  x2 = 2*xmid/3,  x3 = 4*xmid/3;

    Q5 = Quadratic_Eq(E, 0, 0, -2*E*x2, -1, E*x2*x2 + F, false);
    Q6 = Quadratic_Eq(E, 0, 0, -2*E*x3, -1, E*x3*x3 + F, false);
    L1 = Linear_Eq(1, 0, -2*xmid/3, true);
    L2 = Linear_Eq(1, 0, -4*xmid/3, false);
    L3 = Linear_Eq(0, 1, -5*ymid/3, false);


    float ytop = -C*xmid/3*( xmid*xmid/9 + 999 ) - D;

    box = new Bounding_box( x0-A,   ytop,
                x1+A+1, ytop,
                x1+A+1, 5*ymid/3+1,
                x0-A,   5*ymid/3+1);
}

// =====================================================================
// =====================================================================
// =====================================================================


CPU_Shape* allocate_mem_by_shape_type(Shape_type type) {
    CPU_Shape* shape_link = 0;
    switch (type) {
        case TRIANGLE:         shape_link = new Triangle();
                    break;
        case SQUARE:           shape_link = new Square();
                    break;
        case PENTAGON:         shape_link = new Pentagon();
                    break;
        case HEXAGON:          shape_link = new Hexagon();
                    break;
        case CIRCLE:           shape_link = new Circle();
                    break;
        case CIRCLE_II:        shape_link = new Circle_II();
                    break;
        case CIRCLE_III:       shape_link = new Circle_III();
                    break;
        case CIRCLE_IV:        shape_link = new Circle_IV();
                    break;
        case RHOMBUS:          shape_link = new Rhombus();
                    break;
        case RHOMBUS_II:       shape_link = new Rhombus_II();
                    break;
        case RHOMBUS_III:      shape_link = new Rhombus_III();
                    break;
        case RHOMBUS_IV:       shape_link = new Rhombus_IV();
                    break;
        case HEART:            shape_link = new Heart();
                    break;
        case DIAMOND:          shape_link = new Diamond();
                    break;
        case CLUB:             shape_link = new Club();
                    break;
        case SPADE:            shape_link = new Spade();
                    break;
    }
    return shape_link;
}

std::string name_by_shape_type(Shape_type type) {
    switch (type) {
        case TRIANGLE:         return "Triangle";
        case SQUARE:           return "Square";
        case PENTAGON:         return "Pentagon";
        case HEXAGON:          return "Hexagon";
        case CIRCLE:           return "Circle";
        case CIRCLE_II:        return "Circle II";
        case CIRCLE_III:       return "Circle III";
        case CIRCLE_IV:        return "Circle IV";
        case RHOMBUS:          return "Rhombus";
        case RHOMBUS_II:       return "Rhombus II";
        case RHOMBUS_III:      return "Rhombus III";
        case RHOMBUS_IV:       return "Rhombus IV";
        case HEART:            return "Heart";
        case DIAMOND:          return "Diamond";
        case CLUB:             return "Club";
        case SPADE:            return "Spade";
        default:               return "Unrecognized shape in name_by_shape_type";    // Error
    }
}

std::string file_name_by_shape_type(Shape_type type) {
    switch (type) {
        case TRIANGLE:         return "triangle";
        case SQUARE:           return "square";
        case PENTAGON:         return "pentagon";
        case HEXAGON:          return "hexagon";
        case CIRCLE:           return "circle";
        case CIRCLE_II:        return "circleii";
        case CIRCLE_III:       return "circleiii";
        case CIRCLE_IV:        return "circleiv";
        case RHOMBUS:          return "rhombus";
        case RHOMBUS_II:       return "rhombusii";
        case RHOMBUS_III:      return "rhombusiii";
        case RHOMBUS_IV:       return "rhombusiv";
        case HEART:            return "heart";
        case DIAMOND:          return "diamond";
        case CLUB:             return "club";
        case SPADE:            return "spade";
        default:               return "Unrecognized shape in file_name_by_shape_type";    // Error
    }
}
