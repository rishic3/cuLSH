#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <iostream>

int main() {
    RandomProjectionLSH lsh(16, 4);

    MatrixXd X(4, 10);
    X << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    std::cout << X << std::endl;

    auto model = lsh.fit(X);

    model.save("../models");

    auto new_model = RandomProjectionLSHModel::load("../models");

    cout << new_model.get_n_projections() << endl;
}
