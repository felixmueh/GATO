#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace PMP = CGAL::Polygon_mesh_processing;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = K::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;

int main(int argc, char** argv) {
  if (argc < 3 || argc > 7) {
    std::cerr << "usage: " << argv[0]
              << " INPUT_MESH OUTPUT_MESH [relative_alpha=20] [relative_offset=600]"
              << " [max_offset=inf] [max_alpha=inf]\n";
    return EXIT_FAILURE;
  }

  const std::string input_path = argv[1];
  const std::string output_path = argv[2];
  const double relative_alpha = argc > 3 ? std::stod(argv[3]) : 20.0;
  const double relative_offset = argc > 4 ? std::stod(argv[4]) : 600.0;
  const double max_offset = argc > 5 ? std::stod(argv[5]) : HUGE_VAL;
  const double max_alpha = argc > 6 ? std::stod(argv[6]) : HUGE_VAL;

  Mesh input;
  if (!PMP::IO::read_polygon_mesh(input_path, input) || CGAL::is_empty(input) ||
      !CGAL::is_triangle_mesh(input)) {
    std::cerr << "invalid triangle mesh: " << input_path << "\n";
    return EXIT_FAILURE;
  }

  const CGAL::Bbox_3 bbox = PMP::bbox(input);
  const double diag = std::sqrt(
      CGAL::square(bbox.xmax() - bbox.xmin()) +
      CGAL::square(bbox.ymax() - bbox.ymin()) +
      CGAL::square(bbox.zmax() - bbox.zmin()));
  const double relative_alpha_value = diag / relative_alpha;
  const double alpha = std::min(relative_alpha_value, max_alpha);
  const double relative_offset_value = diag / relative_offset;
  const double offset = std::min(relative_offset_value, max_offset);

  Mesh wrapped;
  CGAL::alpha_wrap_3(input, alpha, offset, wrapped);

  if (!CGAL::IO::write_polygon_mesh(
          output_path, wrapped, CGAL::parameters::stream_precision(17))) {
    std::cerr << "failed to write: " << output_path << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "input_vertices " << num_vertices(input) << "\n";
  std::cout << "input_faces " << num_faces(input) << "\n";
  std::cout << "output_vertices " << num_vertices(wrapped) << "\n";
  std::cout << "output_faces " << num_faces(wrapped) << "\n";
  std::cout << "diag " << diag << "\n";
  std::cout << "relative_alpha_value " << relative_alpha_value << "\n";
  std::cout << "max_alpha " << max_alpha << "\n";
  std::cout << "alpha " << alpha << "\n";
  std::cout << "relative_offset_value " << relative_offset_value << "\n";
  std::cout << "max_offset " << max_offset << "\n";
  std::cout << "offset " << offset << "\n";
  std::cout << "output " << output_path << "\n";

  return EXIT_SUCCESS;
}
