#include <iostream>
#include <vector>
#include <chrono>

#include <Eigen/Eigen> 
#include <Eigen/Core>   // for Solver

#include <x86intrin.h>

double flops(const double &secs, size_t length)
{
    double len_ = static_cast<double>(length);
    double ret = (2*len_*len_*len_)/secs;
    return ret;
}

double gflops(const double &secs, size_t length)
{
    double len_ = static_cast<double>(length);
    double ret = (2*len_*len_*len_)/(1000*1000*1000)/secs;
    return ret;
}

void print256d(__m256d x) {
  printf("%f %f %f %f\n", x[3], x[2], x[1], x[0]);
}

template <typename T>
class Matrix {
public:
    Matrix(const size_t &rows, const size_t &cols):
        rows_(rows), cols_(cols)
    {
        size_t elems = rows_ * cols_;
        container_ = std::vector<T>(elems, T(0));
    }

    static
    Matrix<T> Identity(const size_t &rows_cols)
    {
        Matrix<T> ret = Matrix<T>(rows_cols, rows_cols);
        for(auto i = 0; i < rows_cols; i++) {
            ret(i,i) = T(1.0);
        }
        return ret;
    }

    inline
    const T &operator()(const size_t &row, const size_t &col) const 
    {
        size_t index = row * this->cols_ + col;
        return container_[index];
    }
    
    inline
    T &operator()(const size_t &row, const size_t &col) 
    {
        size_t index = row * this->cols_ + col;
        return container_[index];
    }


    template <typename N>
    Matrix operator*(const N &times) const
    {
        Matrix<T> ret(this->row(), this->col() );
        for(auto i = 0; i < this->container_.size(); i++) {
            ret.container_[i] = this->container_[i] * times;
        }
        return ret;
    }

    Matrix dot(const Matrix &rhs) const
    {
        if (this->col() != rhs.row()) { throw;  }

        Matrix ret(this->row() , rhs.col() );
        for(auto i = 0; i < this->row(); i++) {
            for(auto j = 0; j < rhs.col(); j++) {
                for(auto k = 0; k < this->col(); k++) {
                    ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return ret;
    }

    std::vector<T> column(const size_t &col) const
    {
        std::vector<T> ret(this->row());
        for(auto i = 0; i < this->row(); i++) {
            ret[i] = (*this)(i, col);
        }
        return ret;
    }

    Matrix dot2(const Matrix &rhs) const
    {
        // ijk indicess exchange
        if (this->col() != rhs.row()) { throw;  }

        Matrix ret(this->row() , rhs.col() );
        for(auto i = 0; i < this->row(); i++) {
            for(auto k = 0; k < this->col(); k++) {
                for(auto j = 0; j < rhs.col(); j++) {
                    ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return ret;
    }

    Matrix dot3(const Matrix &rhs) const
    {
        // ijk indicess exchange
        if (this->col() != rhs.row()) { throw;  }

        const size_t unroll_length = 8;
        size_t j_times = this->col() / unroll_length;

        Matrix ret(this->row() , rhs.col() );
        for(auto i = 0; i < this->row(); i++) {
            for(auto k = 0; k < this->col(); k++) {
                size_t j_unroll = 0;
                for( ; j_unroll < j_times; j_unroll++) {
                    auto j = j_unroll * unroll_length;
                    ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                    ret(i,j+1) +=  (*this)(i,k) * rhs(k,j+1);
                    ret(i,j+2) +=  (*this)(i,k) * rhs(k,j+2);
                    ret(i,j+3) +=  (*this)(i,k) * rhs(k,j+3);
                    ret(i,j+4) +=  (*this)(i,k) * rhs(k,j+4);
                    ret(i,j+5) +=  (*this)(i,k) * rhs(k,j+5);
                    ret(i,j+6) +=  (*this)(i,k) * rhs(k,j+6);
                    ret(i,j+7) +=  (*this)(i,k) * rhs(k,j+7);
                }
                auto j = j_unroll * unroll_length;
                for( ; j < rhs.col(); j++) {
                    ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return ret;
    }

    Matrix dot4(const Matrix &rhs) const
    {
        // ijk indicess exchange
        if (this->col() != rhs.row()) { throw;  }

        Matrix ret(this->row() , rhs.col() );
        for(auto i = 0; i < this->row(); i++) {
            for(auto k = 0; k < this->col(); k++) {
                #pragma unroll
                for(auto j = 0; j < rhs.col(); j++) {
                    ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return ret;
    }

    Matrix dot5(const Matrix &rhs) const
    {
        // ijk indicess exchange
        if (this->col() != rhs.row()) { throw;  }

        Matrix ret(this->row() , rhs.col() );
        for(auto i = 0; i < this->row(); i++) {
            for(auto k = 0; k < this->col(); k++) {
                double a_ik = (*this)(i,k);
                __m256d v1 = _mm256_set_pd(a_ik, a_ik, a_ik, a_ik); 
                for(auto j = 0; j < rhs.col(); j+=4) {
                    size_t index_kj = k * this->cols_ + j;
                    __m256d v2 = _mm256_load_pd( rhs.container_.data() + index_kj );

                    size_t index_ij = i * this->cols_ + j;
                    __m256d v3 = _mm256_load_pd( ret.container_.data() + index_ij );
                    v3 += v1 * v2;
                    _mm256_store_pd(ret.container_.data() + index_ij, v3);
                    //ret(i,j) +=  (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return ret;

    }

    bool operator==(const Matrix &rhs) const
    {
        bool ret = true;
        if ( this->col() != rhs.col() || this->row() != rhs.row() ) {
            return false;
        }
        ret = (this->container_ == rhs.container_);
        return ret;
    }

    size_t size() const
    {
        return this->container_.size();
    }

    size_t row() const
    {
        return this->rows_;
    }

    size_t col() const
    {
        return this->cols_;
    }

    void print() const
    {
        for(auto i = 0; i != rows_; i++) {
            for(auto j = 0; j != cols_; j++) {
                std::cout << (*this)(i,j) << " ";
            }
            std::cout << std::endl;
        }
    }
private:
    size_t rows_, cols_;
    std::vector<T> container_;
};

int main(void)
{

    std::cout << "sizeof(double): " << sizeof(double) << std::endl;
    const size_t len = 2048;

    Matrix<double> a = Matrix<double>::Identity(len);
    std::cout << "start: length: " << len << std::endl;
#if 0
    {
        std::cout << "====================" << "simple" << "====================" <<std::endl;
        auto start = std::chrono::system_clock::now();
        auto a2 = a.dot(a);
        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << elapsed << " milliseconds" << std::endl;
        std::cout << gflops(elapsed/1000, len) << "Gflops" << std::endl;
        std::cout << "a2(0,0) = " << a2(0,0) << std::endl;
    }
#endif

    {
        std::cout << "====================" << "ijk-exchange" << "====================" <<std::endl;
        auto start = std::chrono::system_clock::now();
        auto a2 = a.dot2(a);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() ;
        std::cout << elapsed << " milliseconds" << std::endl;
        std::cout << gflops(elapsed/1000, len) << "Gflops" << std::endl;
        std::cout << "a2(0,0) = " << a2(0,0) << std::endl;
    }

    {
        std::cout << "====================" << "ijk-exchange + unroll" << "====================" <<std::endl;
        auto start = std::chrono::system_clock::now();
        auto a2 = a.dot4(a);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() ;
        std::cout << elapsed << " milliseconds" << std::endl;
        std::cout << gflops(elapsed/1000, len) << "Gflops" << std::endl;
        std::cout << "a2(0,0) = " << a2(0,0) << std::endl;
    }

    {
        std::cout << "====================" << "ijk-exchange + simd" << "====================" <<std::endl;
        auto start = std::chrono::system_clock::now();
        auto a2 = a.dot5(a);
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() ;
        std::cout << elapsed << " milliseconds" << std::endl;
        std::cout << gflops(elapsed/1000, len) << "Gflops" << std::endl;
        std::cout << "a2(0,0) = " << a2(0,0) << std::endl;
    }

    {
        Eigen::MatrixXd eigen_1 = Eigen::MatrixXd::Identity(len, len);
        std::cout << "====================" << "Eigen" << "====================" <<std::endl;
        auto start = std::chrono::system_clock::now();
        auto eigen_2 = eigen_1 * eigen_1;
        std::cout << "eigen_2(0,0) = " << eigen_2(0,0) << std::endl;
        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout << elapsed << " milliseconds" << std::endl;
        std::cout << gflops(elapsed/1000, len) << "Gflops" << std::endl;

    }

    return 0;
}
