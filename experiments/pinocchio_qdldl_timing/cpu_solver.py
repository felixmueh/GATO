"""Experimental Python-orchestrated CPU SQP reference, not native GATO.

Pinocchio and NumPy use float64; the unchanged historical QDLDL bridge casts
Schur rows/RHS to float32 and retains its full-row residual acceptance check.
The returned whole-batch time includes Python, Pinocchio, NumPy and QDLDL.
Only the frozen one-SQP, position-only, zero-wrench timing protocol is supported.
"""
import time
import numpy as np
import pinocchio as pin


def barrier(value, lower, upper):
    """Historical value/gradient/Hessian, including outside-limit mismatch."""
    a, b = value-lower, upper-value
    def signed_floor(x):
        return np.where(x >= 0, np.maximum(x, 1e-6), np.minimum(x, -1e-6))
    cost = -np.log(np.maximum(a, 1e-10))-np.log(np.maximum(b, 1e-10))
    grad = -1/signed_floor(a)+1/signed_floor(b)
    hess = 1/np.maximum(np.abs(a), 1e-6)**2+1/np.maximum(np.abs(b), 1e-6)**2
    return cost, grad, hess


class CPUSolver:
    def __init__(self, model, n, batch, parameters, bridge):
        if model.nq != 7 or model.nv != 7 or parameters['max_sqp_iters'] != 1:
            raise ValueError('Only TIAGo seven-joint one-SQP timing is supported')
        if parameters['ee_orient_cost'] or parameters['ee_orient_N_cost']:
            raise ValueError('Frozen position-only tracking required')
        self.model, self.N, self.batch_size = model, n, batch
        self.params, self.bridge, self.dt = dict(parameters), bridge, .03
        self.data = model.createData()
        self.tool = model.getFrameId('arm_right_tool_link')
        self.torso = model.getFrameId('torso_lift_link')
        if model.frames[self.torso].parentJoint != 0:
            raise ValueError('The tracking torso frame must be fixed to universe')
        self.stats = {}
        self.reset_dual()
        self.reset_rho()

    def reset_dual(self):
        # QDLDL direct solution never uses a dual initial iterate.
        self.dual = np.zeros((self.batch_size, self.N, 14))

    def reset_rho(self):
        self.rho = np.full(self.batch_size, self.params['rho'])
        self.drho = np.ones(self.batch_size)

    def set_f_ext_B(self, fext):
        if np.any(np.asarray(fext) != 0):
            raise ValueError('Only zero external wrench is supported')

    def get_stats(self):
        return self.stats

    def position(self, q, jacobian=False):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        torso, tool = self.data.oMf[self.torso], self.data.oMf[self.tool]
        position = (torso.inverse()*tool).translation.copy()
        if not jacobian:
            return position
        j = pin.computeFrameJacobian(self.model, self.data, q, self.tool,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3].copy()
        return position, torso.rotation.T@j

    def dynamics(self, x, u, derivatives=False):
        q, v = x[:7], x[7:]
        if derivatives:
            aq, av, au = [a.copy() for a in pin.computeABADerivatives(
                self.model, self.data, q, v, u)]
            acceleration = self.data.ddq.copy()
        else:
            acceleration = pin.aba(self.model, self.data, q, v, u).copy()
        dt = self.dt
        result = np.r_[q+dt*v+.5*dt*dt*acceleration, v+dt*acceleration]
        if not derivatives:
            return result
        a = np.eye(14)
        a[:7, :7] += .5*dt*dt*aq
        a[:7, 7:] = dt*np.eye(7)+.5*dt*dt*av
        a[7:, :7] = dt*aq
        a[7:, 7:] += dt*av
        b = np.r_[.5*dt*dt*au, dt*au]
        return result, a, b

    def unpack(self, plan):
        plan = np.asarray(plan, dtype=float)
        x = np.empty((self.N, 14))
        u = np.empty((self.N-1, 7))
        x[:-1] = plan[:-14].reshape(-1, 21)[:, :14]
        x[-1] = plan[-14:]
        u[:] = plan[:-14].reshape(-1, 21)[:, 14:]
        return x, u

    def pack(self, x, u):
        return np.r_[np.c_[x[:-1], u].ravel(), x[-1]]

    def knot_cost(self, x, u, target, terminal=False, derivatives=False):
        p = self.params
        q, v = x[:7], x[7:]
        if derivatives:
            pos, j = self.position(q, True)
        else:
            pos = self.position(q)
        error = pos-target[:3]
        weight = p['N_cost'] if terminal else p['q_cost']
        bcq, bgq, bhq = barrier(q, self.model.lowerPositionLimit,
                                self.model.upperPositionLimit)
        bcv, bgv, bhv = barrier(v, -self.model.velocityLimit, self.model.velocityLimit)
        cost = (.5*weight*(error@error)+.5*p['qd_cost']*(v@v)
                +p['q_lim_cost']*bcq.sum()+p['vel_lim_cost']*bcv.sum())
        if derivatives:
            gx = np.r_[weight*j.T@error+p['q_lim_cost']*bgq,
                       p['qd_cost']*v+p['vel_lim_cost']*bgv]
            h = np.zeros((14, 14))
            h[:7, :7] = weight*j.T@j+np.diag(p['q_lim_cost']*bhq)
            h[7:, 7:] = np.diag(p['qd_cost']+p['vel_lim_cost']*bhv)
        if terminal:
            return (cost, gx, h, None, None) if derivatives else cost
        bcu, bgu, bhu = barrier(u, -self.model.effortLimit, self.model.effortLimit)
        cost += .5*p['u_cost']*(u@u)+p['ctrl_lim_cost']*bcu.sum()
        if not derivatives:
            return cost
        gu = p['u_cost']*u+p['ctrl_lim_cost']*bgu
        r = np.diag(p['u_cost']+p['ctrl_lim_cost']*bhu)
        return cost, gx, h, gu, r

    def merit(self, x, u, start, reference):
        cost = 0.
        defect = np.abs(x[0]-start).sum()
        for k in range(self.N):
            cost += self.knot_cost(x[k], None if k == self.N-1 else u[k],
                                   reference[k], terminal=k == self.N-1)
            if k < self.N-1:
                defect += np.abs(x[k+1]-self.dynamics(x[k], u[k])).sum()
        return float(cost+self.params['mu']*defect)

    def linearize(self, x, u, start, reference, rho):
        n = self.N
        qmat, rmat, gx, gu = [], [], [], []
        a, b, defect = [], [], [x[0]-start]
        for k in range(n):
            _, g, h, r, h_u = self.knot_cost(x[k], None if k == n-1 else u[k],
                reference[k], terminal=k == n-1, derivatives=True)
            qmat.append(h+rho*np.eye(14))
            gx.append(g)
            if k < n-1:
                rmat.append(h_u)
                gu.append(r)
                predicted, ak, bk = self.dynamics(x[k], u[k], True)
                a.append(ak); b.append(bk); defect.append(x[k+1]-predicted)
        qi = np.linalg.inv(np.asarray(qmat))
        ri = np.linalg.inv(np.asarray(rmat))
        gx, gu, a, b, defect = map(np.asarray, (gx, gu, a, b, defect))
        # Native negative Schur S=-J H^-1 J^T; J=[I; -A,-B,I].
        s = np.zeros((n, 14, 42))
        gamma = np.zeros((n+2, 14))
        s[0, :, 14:28] = -qi[0]
        gamma[1] = defect[0]-qi[0]@gx[0]
        for k in range(n-1):
            aq, br = a[k]@qi[k], b[k]@ri[k]
            s[k, :, 28:] = aq.T
            s[k+1, :, :14] = aq
            s[k+1, :, 14:28] = -(aq@a[k].T+br@b[k].T+qi[k+1])
            gamma[k+2] = defect[k+1]+aq@gx[k]+br@gu[k]-qi[k+1]@gx[k+1]
        return s, gamma.ravel(), (qi, ri, gx, gu, a, b, defect)

    def direction(self, terms, multiplier):
        qi, ri, gx, gu, a, b, defect = terms
        lam = multiplier.reshape(self.N+2, 14)[1:-1].astype(float)
        sx = gx-lam
        sx[:-1] += np.einsum('kji,kj->ki', a, lam[1:])
        su = gu+np.einsum('kji,kj->ki', b, lam[1:])
        dx = -np.einsum('kij,kj->ki', qi, sx)
        du = -np.einsum('kij,kj->ki', ri, su)
        # Preserve the existing optional internal shortcut; never a study gate.
        tol = self.params['kkt_tol']
        flag = all(np.all(np.isfinite(v)) and np.all(np.abs(v) < tol)
                   for v in (sx, su, defect))
        return dx, du, bool(flag)

    def solve(self, starts, references, plans):
        # Binding array validation/copy is outside the timed solve boundary,
        # matching the scope exclusion of native GATO binding transfers.
        starts = np.array(starts, dtype=float, copy=True)
        references = np.array(references, dtype=float, copy=True).reshape(self.batch_size, self.N, 6)
        plans = np.array(plans, dtype=float, copy=True)
        if starts.shape != (self.batch_size, 14) or plans.shape != (self.batch_size, self.N*21-7):
            raise ValueError('Batch input shapes do not match solver')
        if not all(np.isfinite(v).all() for v in (starts, references, plans)):
            raise FloatingPointError('Nonfinite CPU solve input')
        begin = time.perf_counter_ns()
        lanes, flags = [], []
        initial, final, steps = [], [], []
        for lane in range(self.batch_size):
            x, u = self.unpack(plans[lane])
            merit = self.merit(x, u, starts[lane], references[lane])
            s, rhs, terms = self.linearize(x, u, starts[lane], references[lane], self.rho[lane])
            lam = self.bridge.solve(self.N, s, rhs, self.params['pcg_tol'])
            dx, du, flag = self.direction(terms, lam)
            self.dual[lane] = np.asarray(lam).reshape(self.N+2, 14)[1:-1]
            lanes.append((x, u, dx, du, merit)); flags.append(flag)
            initial.append(merit)
        all_stop = all(flags)
        for lane, (x, u, dx, du, initial_merit) in enumerate(lanes):
            if all_stop:
                final.append(initial_merit); steps.append(0.)
                continue
            alphas = 2.**(-np.arange(8))
            merits = [self.merit(x+alpha*dx, u+alpha*du,
                                 starts[lane], references[lane]) for alpha in alphas]
            # Native strict comparisons ignore NaN/non-improving infinity
            # candidates; they do not let NaN hide a finite better candidate.
            ranked = np.where(np.isfinite(merits), merits, np.inf)
            index = int(np.argmin(ranked))
            success = bool(ranked[index] < min(initial_merit, 1e38))
            self.drho[lane] = min(self.drho[lane]/1.2, 1/1.2) if success else max(self.drho[lane]*1.2, 1.2)
            self.rho[lane] = np.clip(self.rho[lane]*self.drho[lane], 1e-8, 10.)
            if success:
                plans[lane] = self.pack(x+alphas[index]*dx, u+alphas[index]*du)
            final.append(merits[index] if success else initial_merit)
            steps.append(float(alphas[index]) if success else -1.)
        # Native recomputes final merit even after its line search.
        for lane in range(self.batch_size):
            x, u = self.unpack(plans[lane])
            final[lane] = self.merit(x, u, starts[lane], references[lane])
        elapsed_us = (time.perf_counter_ns()-begin)/1000.
        self.drho[:] = 1.
        self.stats = dict(sqp_iters=[1]*self.batch_size,
            kkt_converged=flags if all_stop else [False]*self.batch_size,
            direct_solve_work=[1]*self.batch_size, initial_merit=initial,
            final_merit=final, step_size=steps,
            timing_scope='Whole-batch CPU wall including Python/Pinocchio/NumPy/QDLDL; no CUDA')
        if not np.isfinite(plans).all():
            raise FloatingPointError('Nonfinite CPU returned plan')
        return plans, elapsed_us
