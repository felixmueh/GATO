#pragma once
// Portable CPU setup from the public GLASS negative-Schur row strips.
// No private TIAGo headers, CUDA dependency, or global dense N*D matrix.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace upstream_precond {
class BlockLdltFactors {
 public:
  using Matrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
  void setup(const float* negative_s,std::size_t count,int knots,int block_size) {
    lower_.clear();inverse_.clear();knots_=0;block_size_=0;
    if(!negative_s||knots<1||block_size<1||count!=3ULL*knots*block_size*block_size)
      throw std::invalid_argument("Expected public [N,D,3D] negative-Schur strips");
    const std::size_t block=static_cast<std::size_t>(block_size)*block_size;
    std::vector<float> lower(static_cast<std::size_t>(knots)*block,0.0f);
    std::vector<float> inverse(lower.size(),0.0f);
    Matrix previous_inverse;
    for(int k=0;k<knots;++k){
      Matrix diagonal(block_size,block_size),edge(block_size,block_size);
      for(int row=0;row<block_size;++row)
        for(int col=0;col<block_size;++col){
          const auto base=(static_cast<std::size_t>(k)*block_size+row)*3*block_size;
          diagonal(row,col)=-negative_s[base+block_size+col];
          edge(row,col)=k? -negative_s[base+col]:0.0f;
        }
      if(!diagonal.allFinite()||!edge.allFinite())
        throw std::runtime_error("Nonfinite block LDLT mathematical input");
      const float scale=std::max(1.0f,diagonal.cwiseAbs().maxCoeff());
      const float tolerance=64.0f*std::numeric_limits<float>::epsilon()*scale;
      if((diagonal-diagonal.transpose()).cwiseAbs().maxCoeff()>tolerance)
        throw std::runtime_error("Nonsymmetric Schur diagonal block");
      if(k){
        for(int row=0;row<block_size;++row)
          for(int col=0;col<block_size;++col){
            const auto previous_base=(static_cast<std::size_t>(k-1)*block_size+col)*3*block_size;
            const float transpose_entry=-negative_s[previous_base+2*block_size+row];
            if(!std::isfinite(transpose_entry)||std::abs(edge(row,col)-transpose_entry)>tolerance)
              throw std::runtime_error("Nonsymmetric Schur adjacent blocks");
          }
        const Matrix L=edge*previous_inverse;
        Eigen::Map<Matrix>(lower.data()+static_cast<std::size_t>(k)*block,block_size,block_size)=L;
        diagonal.noalias()-=L*edge.transpose();
      }
      // Eigen's symmetric LDLT consumes the lower triangle. No shift, floor,
      // thresholded sparsity, rescue, or alteration of the device S is applied.
      Eigen::LDLT<Matrix> factor(diagonal);
      if(factor.info()!=Eigen::Success||!factor.vectorD().allFinite()||
         (factor.vectorD().array()<=0.0f).any())
        throw std::runtime_error("Block LDLT factor is not positive definite");
      previous_inverse=factor.solve(Matrix::Identity(block_size,block_size));
      if(factor.info()!=Eigen::Success||!previous_inverse.allFinite())
        throw std::runtime_error("Block LDLT inverse is nonfinite");
      Eigen::Map<Matrix>(inverse.data()+static_cast<std::size_t>(k)*block,block_size,block_size)=previous_inverse;
    }
    lower_=std::move(lower);inverse_=std::move(inverse);knots_=knots;block_size_=block_size;
  }
  const std::vector<float>& lower()const{return lower_;}
  const std::vector<float>& diagonal_inverse()const{return inverse_;}
  int knots()const{return knots_;}
  int block_size()const{return block_size_;}
 private:
  int knots_=0,block_size_=0;
  std::vector<float> lower_,inverse_;
};
} // namespace upstream_precond
