#include "cpu_strip_methods.hpp"
#include <iostream>

using upstream_cpu::Matrix;
using upstream_cpu::Vector;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void close(const Matrix& a, const Matrix& b, const std::string& message) {
  require((a-b).norm() < 1.0e-4f * std::max(1.0f,b.norm()),message);
}
void check(int knots, int width, const std::vector<std::string>& methods, bool poisson=false) {
  const int size=knots*width;
  Matrix H=Matrix::Zero(size,size);
  for(int i=0;i<size;++i)
    for(int j=0;j<i;++j)
      if(i/width-j/width<=1) H(i,j)=H(j,i)=0.03f*((i*7+j*3)%11-5);
  for(int i=0;i<size;++i) H(i,i)=1.0f+H.row(i).cwiseAbs().sum();
  if(poisson){H.setZero();H.diagonal().setConstant(2.0f);for(int i=1;i<size;++i)H(i,i-1)=H(i-1,i)=-1.0f;}
  std::vector<float> strip(static_cast<size_t>(knots)*3*width*width,0.0f);
  for(int k=0;k<knots;++k)
    for(int row=0;row<width;++row)
      for(int slot=0;slot<3;++slot){
        const int column=k+slot-1;if(column<0||column>=knots)continue;
        for(int col=0;col<width;++col)
          strip[(k*width+row)*3*width+slot*width+col]=-H(k*width+row,column*width+col);
      }
  upstream_cpu::Blocks blocks;blocks.setup(strip.data(),knots,width);
  close(Matrix(blocks.sparse()),H,"Strip conversion lost coupling/sign/layout");
  const Matrix identity=Matrix::Identity(size,size);
  for(const std::string& name:methods){
    require(upstream_cpu::available(name),"Required CPU method unavailable: "+name);
    auto method=upstream_cpu::make(name);method->setup(blocks);
    Matrix inverse(size,size);
    for(int i=0;i<size;++i)inverse.col(i)=method->apply(Vector::Unit(size,i));
    close(inverse,inverse.transpose(),name+" inverse symmetry");
    Eigen::SelfAdjointEigenSolver<Matrix> eig(inverse);
    require(eig.info()==Eigen::Success&&eig.eigenvalues().minCoeff()>0,name+" inverse SPD");
    const Vector rhs=Vector::LinSpaced(size,-0.5f,0.7f);
    close(method->apply(rhs),inverse*rhs,name+" fixed linear map");
    if(name=="incomplete_cholesky")require(method->configured_initial_shift()==1.0e-3f&&!method->actual_shift_known(),"IC shift metadata");
    if(name=="block_sgs"||name=="block_jacobi"){
      Matrix D=Matrix::Zero(size,size),L=Matrix::Zero(size,size);
      for(int k=0;k<knots;++k){
        D.block(k*width,k*width,width,width)=H.block(k*width,k*width,width,width);
        if(k)L.block(k*width,(k-1)*width,width,width)=H.block(k*width,(k-1)*width,width,width);
      }
      Matrix M=D;
      if(name=="block_sgs"){const Matrix forward=D+L;M=forward*D.llt().solve(Matrix(forward.transpose()));}
      close(inverse,M.llt().solve(identity),name+" mathematical formula");
    }
    if(name=="schwarz"){
      Matrix expected=Matrix::Zero(size,size);
      for(int start=0;start<knots;start+=3){
        const int count=std::min(4,knots-start),n=count*width;
        const Matrix local=H.block(start*width,start*width,n,n);
        expected.block(start*width,start*width,n,n)+=local.llt().solve(Matrix::Identity(n,n));
        if(start+count==knots)break;
      }
      close(inverse,expected,"Additive Schwarz restriction/prolongation oracle");
    }
    if(name=="amg"&&size<=32)close(inverse,H.llt().solve(identity),"AMG coarse LLT oracle");
  }
}
int main(int argc,char** argv){
  try{
    const bool require_amg=argc==2&&std::string(argv[1])=="--require-amg";
    require(argc==1||require_amg,"Only optional argument is --require-amg");
    std::vector<std::string> methods={"incomplete_cholesky","block_sgs","schwarz","block_jacobi"};
    if(require_amg)methods.push_back("amg");
    for(int knots:{1,2,4,5,8})check(knots,3,methods);
    check(6,12,methods);
    if(require_amg)check(97,1,{"amg"},true);
    std::cout<<"Public CPU strip math checks passed for IC/SGS/Schwarz/Jacobi; AMG "
             <<(require_amg?"required and checked":"not requested or checked")<<'\n';return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
