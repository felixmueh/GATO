# PAL controller interfaces

Unchanged `.msg`, `.srv`, and `package.xml` files from
`pal-alum-controller-manager-msgs` **4.32.2-1jammy.20250627.080505**, installed in
the university simulator image `development-tiago-pro-12:alum-25.01` on c3po.
Captured 2026-09-23 from `/opt/pal/alum/share/controller_manager_msgs`.
Package repository: `https://repo.pal-robotics.com/clients/alum/25.01/pal`.
The package declares the BSD license; original authors/maintainers are preserved
in `package.xml`. `SHA256SUMS` pins all captured source files.

The physical TIAGo reports the same package version and inspected definitions.
Standard Humble 2.54.x has incompatible controller-management message layouts.
The simulator also reproduces malformed controller listings with that client.

The local `CMakeLists.txt` builds only interface type support against Humble.
Docker builds it natively for desktop amd64 or Jetson arm64. Do not copy compiled
PAL libraries between architectures or replace the robot's controller manager.
Generated `.idl`, request/response `.msg`, and compiled files are deliberately
excluded; ROS generates them from these source definitions.

To update: obtain the complete package's definitions from the target PAL release,
review dependencies and API changes, replace the snapshot and hashes, then check
both the generated Python API and read-only service responses against PAL.
Do not infer compatibility from `ROS_DISTRO` or the package version alone.
