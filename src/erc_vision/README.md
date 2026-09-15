# Emirates Robotics Competition 2026 — Team Tuttle

## Solution: Autonomous Book Retrieval with TIAGo Pro

### Pipeline
The solution runs as a 4-stage pipeline orchestrated by `solution.launch.py`:

1. **`tuck_arms_once`** — Folds both arms and lowers the torso before any motion (protects against the nearby table).
2. **`set_initial_pose`** — Seeds AMCL from the known spawn pose so navigation has a valid localization.
3. **`column_detector`** — Rotates to find the shelf, reads all five overhead numeral markers via template matching, fits a line through them to establish the shelf face, and publishes the target column.
4. **`approach_column`** — Uses Nav2 to navigate near the target column, then closed-loop aligns to 2 cm and 0.5° precision.
5. **`book_color_detector`** — Sweeps head tilt across the four book rows, detects the target colour via HSV masking, determines the row from the measured 3D height, and publishes the row.

### How to Run
```bash
# Terminal 1 — launch the simulation
ros2 launch erc_bringup simulation.launch.py

# Terminal 2 — launch the solution
ros2 launch erc_vision solution.launch.py shelf_column_number:=2 book_colour:=red
