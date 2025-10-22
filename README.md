# CompRobo (FA25)

This repository is my submission for the Olin College's "A Computational Introduction to Robotics" (CompRobo), Fall 2025. Below are overviews for the main projects in this course.

Robot Behaviors
------------------
**Purpose**
: Build and integrate multiple robot behaviors and switch between them using a finite state machine (FSM). Emphasis on clean hand-offs, debugging with ROS tools, and visualization.

**What It Contains**
:- Behavior nodes: `teleop`, `drive_square`, `wall_follower`, `person_follower`.
 - `fsm_manager.py`: subscribes to each behavior's private velocity topic, selects one, republishes it to `/cmd_vel`, and publishes the current mode on `fsm/state`.
 - Launch wrapper: `robobehaviors_fsm/launch/fsm_launch.py` to start the stack.
 - Recorded bag files in `robobehaviors_fsm/bags/` for offline playback and testing.

**Highlights from the Report**
:- **Teleop**: a keyboard driver that is robust when launched or run interactively. Keys map to velocity commands and the node publishes a safe stop when deactivated.
 - **Drive Square**: sequences forward and turn actions in a thread so the robot drives repeated squares while the square behavior is active.
 - **Wall Follower**: keeps the robot parallel to nearby walls by comparing side-range readings and applying proportional corrections; includes a small recovery routine after bump events.
 - **Person Follower**: looks for the closest object in a forward-facing sector of the scan and follows it while keeping a safety buffer; tuning scan selection improves reliability.

Robot Localization
------------------
**Purpose**
: Implement a particle filter to localize the Neato on a known map using lidar scans and odometry.

**What It Contains**
:- Particle filter implementation with the standard predict–update–resample loop.
 - Example bag files for evaluation in `robot_localization/bags/` and demo media in the project assets.

**Method and Implementation Details**
:- **Initialization**: the filter begins with a cloud of particles spread around the initial pose estimate.
 - **Motion Update**: particles are updated to reflect odometry-based motion while maintaining diversity so the cloud can recover if needed.
 - **Sensor Update**: lidar scans are compared to the map from each particle's perspective to evaluate how well each particle explains the observations.
 - **Pose Estimate**: the algorithm selects the most likely particle as the reported pose to avoid poor averaging when multiple hypotheses exist.
 - **Resampling**: particles are resampled based on their relative weights to prepare for the next iteration; updates are performed only when movement warrants it to save computation.

**Results and Challenges**
:- The filter converges well on the provided Gauntlet and MAC recordings (tested in simulation and by replaying bags). A key bug early on was applying world-frame odometry directly to particles; the fix was to transform motions into the body frame and then apply them per-particle.

Notes
-----
- Read each project's `README.md` for exact parameters, commands, and reproduction steps — the subproject READMEs contain the step-by-step report-style instructions used to produce the submitted results.
- Bag files under `*/bags/` are included for offline testing when simulator or hardware are not available.

