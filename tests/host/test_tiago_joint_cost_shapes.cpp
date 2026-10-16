// Focused scalar check: compile and run without CUDA or a GPU.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#define __host__
#define __device__
#include "gato/dynamics/tiago_right/tiago_right_limits.cuh"
#include "gato/dynamics/tiago_right/tiago_right_barrier.cuh"
using namespace gato::plant;
static int checks=0;
static void check(bool pass, const char* label) {
    ++checks;
    if (!pass) { throw std::runtime_error(label); }
}
static void close(double a, double b, double abs_tol, double rel_tol, const char* label) {
    check(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=abs_tol+rel_tol*std::max(std::abs(a),std::abs(b)),label);
}
template<class T> void baseline_identity() {
    for(int j=0;j<7;++j) {
        const T lower=JOINT_LIMITS<T>()[j][0]+T(.2), upper=JOINT_LIMITS<T>()[j][1]-T(.2);
        for(int n=0;n<=500;++n) {
            const T q=lower-T(.5)+(upper-lower+T(1))*T(n)/T(500);
            const T actual[]={tiagoJointShapeValue(q,lower,upper,T(.2),0,T(0)),
                              tiagoJointShapeGradient(q,lower,upper,T(.2),0,T(0)),
                              tiagoJointShapeHessian(q,lower,upper,T(.2),0,T(0))};
            const T original[]={jointBarrier(q,lower,upper),jointBarrierGradient(q,lower,upper),jointBarrierHessian(q,lower,upper)};
            check(std::memcmp(actual,original,sizeof(actual))==0,"mode0 scalar identity");
        }
    }
}
int main() {
    baseline_identity<float>(); baseline_identity<double>();
    for(int j=0;j<7;++j) {
        const double lower=JOINT_LIMITS<double>()[j][0]+.2,upper=JOINT_LIMITS<double>()[j][1]-.2;
        const double delta=jointBarrierRelaxationDistance(lower,upper),margin=.2;
        check(upper>lower,"soft bounds nonempty");
        for(double outward: {0.,.02,.1,.199,.2,.25,.5}) {
            const auto p=tiagoFlatBarrierSide(outward,margin,delta);
            check(p.value>=0 && p.gradient>=0 && p.hessian>=0,"flat convex side");
            if(outward>=margin) {
                close(p.gradient,-relaxedLogBarrierGradient(-outward,delta),1e-10,1e-12,"raw/outside restoring slope");
                close(p.hessian,relaxedLogBarrierHessian(-outward,delta),1e-10,1e-12,"raw/outside curvature");
            }
        }
        for(double join: {0.,margin}) {
            const auto left=tiagoFlatBarrierSide(join-1e-8,margin,delta);
            const auto at=tiagoFlatBarrierSide(join,margin,delta);
            const auto right=tiagoFlatBarrierSide(join+1e-8,margin,delta);
            for(auto p:{left,right}) {
                close(p.value,at.value,1e-5,1e-6,"C2 join value");
                close(p.gradient,at.gradient,1e-4,1e-6,"C2 join gradient");
                close(p.hessian,at.hessian,1e-3,1e-6,"C2 join Hessian");
            }
        }
        for(int n=0;n<=500;++n) {
            const double q=lower-.5+(upper-lower+1)*n/500.;
            const auto flat=tiagoFlatJointBarrier(q,lower,upper,margin);
            check(flat.value>=0 && flat.hessian>=0,"flat value and curvature");
            if(q>=lower && q<=upper) {
                check(flat.value==0 && flat.gradient==0 && flat.hessian==0,"exact flat interior");
            }
            for(double ref:{(lower+upper)/2,lower-.1,upper+.1}) {
                for(int mode=0;mode<=2;++mode) {
                    auto value=[&](double x){return tiagoJointShapeValue(x,lower,upper,margin,mode,ref);};
                    auto gradient=[&](double x){return tiagoJointShapeGradient(x,lower,upper,margin,mode,ref);};
                    const double h=1e-6;
                    const double v=value(q),g=gradient(q);
                    const double H=tiagoJointShapeHessian(q,lower,upper,margin,mode,ref);
                    check(std::isfinite(v) && std::isfinite(g) && std::isfinite(H),"finite all modes");
                    close((value(q+h)-value(q-h))/(2*h),g,2e-6,2e-6,"gradient finite difference");
                    close((gradient(q+h)-gradient(q-h))/(2*h),H,1e-5,2e-5,"Hessian finite difference");
                    if(mode==2) {
                        check(v>=-1e-9,"recenter convex nonnegative");
                        close(H,jointBarrierHessian(q,lower,upper),0,0,"recenter unchanged Hessian");
                        check(value(ref)==0 && gradient(ref)==0,"zero reference value and slope");
                    }
                }
            }
        }
    }
    std::cout << "PASS " << checks << " scalar checks; seven bounds; float/double baseline identity; C2, convexity, finite differences, recenter\n";
}
