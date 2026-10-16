# Client schema transcribed from TIAGo Pro's installed interface, 2026-10-16.
# position: motor-side angle; absolute_position: output joint angle [rad].
# expected_gravity_torque is a model quantity [Nm], not measured effort.
std_msgs/Header header
string[] name
float64[] position
float64[] absolute_position
float64[] expected_gravity_torque
