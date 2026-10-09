"""Host-only checks of the actual production pybind telemetry conversion."""
import hashlib
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def extract_conversion(source):
    start = source.index('                // Per-iteration stats: shape them as (iters, BatchSize)')
    end = source.index('                std::vector<float> ls_min_merit;', start)
    return source[start:end]


def build_conversion(source, output, name):
    """Compile the verbatim production block with host record transport only."""
    output.mkdir(parents=True, exist_ok=False)
    block = extract_conversion(source)
    (output / 'conversion.txt').write_text(block)
    cpp = r'''
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace py = pybind11;
struct Record { double solve_time_us; std::vector<int> num_iterations; };
struct Stats { std::vector<Record> pcg_stats; std::vector<int> line_search_stats; };
template<uint32_t BatchSize> py::dict convert(std::vector<std::vector<int>> rows,
                                              std::vector<double> times, size_t ls_rows) {
    if(rows.size()!=times.size()) throw std::invalid_argument("time rows");
    Stats stats; stats.line_search_stats.resize(ls_rows);
    for(size_t i=0;i<rows.size();++i) {
        if(rows[i].size()!=BatchSize) throw std::invalid_argument("batch width");
        stats.pcg_stats.push_back({times[i],rows[i]});
    }
    py::dict result;
'''+block+r'''
    return result;
}
PYBIND11_MODULE('''+name+r''',m) {
    m.def("convert1",&convert<1>);m.def("convert4",&convert<4>);
}
'''
    path = output / 'conversion.cpp'; path.write_text(cpp)
    binary = output / (name + sysconfig.get_config_var('EXT_SUFFIX'))
    includes = shlex.split(subprocess.check_output([sys.executable, '-m', 'pybind11', '--includes'], text=True))
    command = ['c++', '-std=c++17', '-O2', '-shared', '-fPIC', *includes, str(path), '-o', str(binary)]
    with (output / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    spec = importlib.util.spec_from_file_location(name, binary)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    receipt = dict(command=command, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                   extracted_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
                   cpp_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    return module, receipt


def wrapper_result(converted, batch):
    """Feed recorded transport through the unchanged public Python method."""
    from bsqp.interface import BSQP
    result = dict(converted, XU=np.zeros((batch, 665), np.float32), sqp_time_us=0,
                  sqp_iters=np.full(batch, converted['pcg_iters'].shape[0], np.int32),
                  kkt_converged=np.zeros(batch, np.int32), final_merit=np.zeros(batch, np.float32),
                  ls_min_merit=np.zeros((converted['ls_num_iters'], batch), np.float32),
                  ls_step_size=np.zeros((converted['ls_num_iters'], batch), np.float32))
    instance = BSQP.__new__(BSQP)
    instance.batch_size=batch; instance.nx=14; instance.dt=.03; instance.stats={}
    instance.XU_B=result['XU'].copy()
    instance.solver=SimpleNamespace(solve=lambda *args:result)
    instance.solve(np.zeros((batch,14),np.float32),np.zeros((batch,192),np.float32))
    return instance.stats


@pytest.fixture(scope='module')
def conversion(tmp_path_factory):
    return build_conversion((ROOT/'python/bindings.cu').read_text(),
                            tmp_path_factory.mktemp('pcg_parent')/'candidate', 'pcg_conversion_test')[0]


@pytest.mark.parametrize('batch,rows,times,ls_rows', [
    (1, [[22],[35],[1],[0]], [0.,0.,0.,0.], 3),
    (1, [[0]], [7.25], 0),
    (1, [], [], 0),
    (1, [[3],[120]], [2.5,5.25], 2),
    (4, [[4,3,2,1],[0,3,120,4]], [1.25,2.5], 1),
    (4, [], [], 0),
])
def test_pcg_rows_and_times_independent_of_line_search(conversion,batch,rows,times,ls_rows):
    result=getattr(conversion,f'convert{batch}')(rows,times,ls_rows)
    expected=np.asarray(rows,np.int32).reshape(len(rows),batch)
    np.testing.assert_array_equal(result['pcg_iters'],expected)
    np.testing.assert_array_equal(result['pcg_times_us'],np.asarray(times,np.float32))
    assert result['pcg_iters'].dtype==np.int32
    assert result['pcg_times_us'].dtype==np.float32
    assert result['ls_num_iters']==ls_rows
    # Also proves Python normalization retains native independent 2D shapes.
    stats=wrapper_result(result,batch)
    np.testing.assert_array_equal(stats['pcg_iters'],expected)
    np.testing.assert_array_equal(stats['pcg_times_us'],np.asarray(times,np.float32))
    assert stats['min_merit'].shape==(ls_rows,batch)
    assert stats['step_size'].shape==(ls_rows,batch)
