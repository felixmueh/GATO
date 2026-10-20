#include "block_ldlt_factors.hpp"
#include <iostream>

int main(){
  using Factors=upstream_precond::BlockLdltFactors;
  using Matrix=Factors::Matrix;
  try{
    for(int n:{1,2,7})for(int d:{1,3,12}){
      const int size=n*d;
      Matrix H=Matrix::Zero(size,size);
      for(int i=0;i<size;++i)for(int j=0;j<i;++j)if(i/d-j/d<=1)
        H(i,j)=H(j,i)=.015f*static_cast<float>((i*7+j*3)%9-4);
      for(int i=0;i<size;++i)H(i,i)=1.0f+H.row(i).cwiseAbs().sum();
      // Poison unused off-edge storage to prove setup consumes mathematical
      // blocks only, matching public GLASS's padded-vector interpretation.
      std::vector<float> S(3*n*d*d,std::numeric_limits<float>::quiet_NaN());
      for(int k=0;k<n;++k)for(int i=0;i<d;++i)for(int off=-1;off<=1;++off)
        if(k+off>=0&&k+off<n)for(int j=0;j<d;++j)
          S[(k*d+i)*3*d+(off+1)*d+j]=-H(k*d+i,(k+off)*d+j);
      Factors factors;factors.setup(S.data(),S.size(),n,d);
      Matrix L=Matrix::Identity(size,size),D=Matrix::Zero(size,size);
      for(int k=0;k<n;++k){
        const Eigen::Map<const Matrix> inverse(factors.diagonal_inverse().data()+k*d*d,d,d);
        D.block(k*d,k*d,d,d)=inverse.inverse();
        if(k)L.block(k*d,(k-1)*d,d,d)=Eigen::Map<const Matrix>(factors.lower().data()+k*d*d,d,d);
      }
      const Matrix reconstructed=L*D*L.transpose();
      if((H-reconstructed).norm()>3e-5f*H.norm())throw std::runtime_error("Block LDLT reconstruction disagrees with independent SPD matrix");
      S[d]=1.0f; // force a negative first H diagonal
      bool rejected=false;
      try{factors.setup(S.data(),S.size(),n,d);}catch(const std::runtime_error&){rejected=true;}
      if(!rejected||!factors.lower().empty())throw std::runtime_error("Non-SPD setup must fail and discard old factors");
    }
    std::cout<<"Direct strip block LDLT factor reconstruction checks passed\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
