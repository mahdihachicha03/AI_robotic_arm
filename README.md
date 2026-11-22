# UR5e Vision + RL Pick-and-Place (MuJoCo)

A simulated UR5e robot arm that sees objects, plans motions and sorts them into bins. The project compares **classical control** (custom FK/IK, PID, computed torque) against **learning-based control** (SAC reinforcement learning), adds **YOLO-based 3D object localization**, and finishes with an **LLM agent** that takes natural-language commands.

> Status: in progress. Results sections below are placeholders. Fill them with measured numbers only.

---

## Goals

- Implement forward and inverse kinematics from scratch and validate them against Pinocchio and MuJoCo.
- Implement joint-space PID and computed-torque control, and show where each one breaks down (for example with an unmodeled payload).
- Detect objects with a camera, localize them in 3D and convert them to the robot base frame.
- Train an RL policy for reaching and benchmark it against classical IK.
- Control the whole system through an LLM that calls tools such as `detect_objects`, `pick` and `place`.

## Tech stack (CPU-friendly, no local GPU required)

| Area | Tools |
|---|---|
| Simulator | MuJoCo + UR5e model from MuJoCo Menagerie |
| Kinematics / dynamics reference | Pinocchio (`pip install pin`) |
| Vision | OpenCV baseline, then YOLOv8n |
| RL | Stable-Baselines3 (SAC / PPO), Gymnasium |
| Agent | LLM API with tool calling, or a small local model via Ollama |
| Heavy training | Google Colab / Kaggle GPU (YOLO fine-tuning only) |

## Repository structure

```
ur5-vision-rl/
├── kinematics/    # FK, Jacobian, IK, validation vs Pinocchio
├── control/       # PID, computed torque, (optional) operational space
├── vision/        # synthetic data generation, detector, 3D localization
├── rl/            # Gymnasium env, training, evaluation
├── agent/         # LLM tool layer
├── experiments/   # scripts that generate the plots in this README
├── notebooks/     # thin Colab notebooks that call the repo modules
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install mujoco numpy matplotlib pin gymnasium stable-baselines3[extra] opencv-python ultralytics
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

For headless rendering (Colab / servers): `export MUJOCO_GL=egl`.

---

## Roadmap (about 5 weeks, 8-10 h/week)

### Phase 0: Setup (2-3 days)

- [ ] Install dependencies and clone MuJoCo Menagerie
- [ ] Create the repo structure above, commit daily
- [ ] Load the UR5e, step the sim, read joint angles and the end-effector pose

**Learn from**
- [MuJoCo-Tutorial notebooks (Colab-friendly)](https://github.com/tayalmanan28/MuJoCo-Tutorial)
- [MuJoCoPy Bootcamp](https://pab47.github.io/mujocopy.html) and its [video playlist](https://www.youtube.com/playlist?list=PLc7bpbeTIk758Ad3fkSywdxHWpBh9PM0G)

### Phase 1: Kinematics (week 1)

- [ ] FK from DH parameters, validated against MuJoCo's end-effector pose
- [ ] Geometric Jacobian, checked with finite differences
- [ ] Numerical IK (damped least squares) with singularity handling
- [ ] Straight-line Cartesian trajectory respecting joint limits

**Deliverable:** end-effector error plot (own FK vs MuJoCo vs Pinocchio) and an IK success-rate table.

**Learn from**
- [Modern Robotics video series, Class Central playlist](https://www.classcentral.com/classroom/youtube-modern-robotics-all-videos-48187) (Kevin Lynch): FK, Jacobians, singularities, manipulability
- [Chapter 6: Inverse Kinematics](https://modernrobotics.northwestern.edu/chapters/chapter6)
- [MujocoTutorials `robot_arms.ipynb`](https://gitee.com/intheposition/MujocoTutorials): model-based control on UR5e, Panda and Kuka (read after writing your own IK)

### Phase 2: Dynamics and control (week 2)

- [ ] Joint-space PD / PID on a smooth trajectory
- [ ] Computed-torque control using `mj_fullM` (mass matrix) and `qfrc_bias`
- [ ] Payload experiment: add mass at the end-effector, compare PID vs computed torque with and without an accurate model
- [ ] Optional: task-space (operational space) control

**Deliverable:** tracking-error and torque-effort plots for PID vs computed torque, plus the payload result.

**Learn from**
- [Modern Robotics 11.4: computed torque](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-4-motion-control-with-torque-or-force-inputs-part-3-of-3/)
- [Chapter 8: Dynamics of open chains](https://modernrobotics.northwestern.edu/chapters/chapter8)
- [Chapter 11: Robot control](https://modernrobotics.northwestern.edu/chapters/chapter11)

### Phase 3: Vision and pick-and-place (week 3)

- [ ] Add a camera (RGB + depth) to the scene
- [ ] Generate a synthetic dataset: randomize cube colors, positions, lighting and camera pose; auto-label from MuJoCo's segmentation render
- [ ] OpenCV color-threshold baseline, then fine-tune YOLOv8n on Colab
- [ ] 3D localization: pixel + depth -> camera frame (using intrinsics) -> robot base frame
- [ ] Scripted grasp: approach, descend, close, lift, move to bin, release

**Deliverable:** video of the arm sorting cubes, a detection precision/recall table, and the pick success rate.

**Learn from**
- [Ultralytics: training YOLOv8 on a custom dataset in Colab](https://www.ultralytics.com/blog/training-custom-datasets-with-ultralytics-yolov8-in-google-colab) (includes the video)
- [Camera projection lecture slides (Yu Xiang)](https://yuxng.github.io/Courses/CS4391Spring2025/lecture_11_camera_projection.pdf): back-projecting a pixel with known depth to a 3D point

### Phase 4: Reinforcement learning (week 4)

- [ ] Gymnasium env: observation = joint positions, velocities, target position; action = joint velocity or position deltas; reward = negative distance to target + small action penalty + success bonus
- [ ] Run `check_env`, train SAC (add HER if the reward is sparse), use parallel envs
- [ ] Evaluate on 100+ random targets and compare to the IK controller: success rate, final error, path smoothness, training cost

**Deliverable:** learning curves and a classical-vs-RL comparison table.

**Learn from**
- [SB3 + Gymnasium quickstart tutorial (Antonin Raffin)](https://araffin.github.io/slides/sb3-gym-quickstart)
- [RL tips and tricks (video)](https://www.youtube.com/watch?v=Ikngt0_DXJg) before designing rewards
- [Recent advances in RL for continuous control (video)](https://www.youtube.com/watch?v=Sb0dgmxbGZY) before choosing SAC vs PPO
- [SB3 custom environment guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [TorchRL tutorial with a Menagerie UR5e + Robotiq gripper](https://docs.pytorch.org/rl/stable/tutorials/mujoco_cube_bowl_macros.html)

### Phase 5: LLM agent and polish (week 5)

- [ ] Wrap `detect_objects()`, `pick(object_id)`, `place(bin)`, `go_home()` as tools with JSON schemas
- [ ] LLM loop: parse "put the red cube in the blue bin", call tools, read results, retry on failure
- [ ] Guardrails: the agent never sends raw joint commands, and every tool validates its inputs
- [ ] Record the demo video, add an architecture diagram, results tables and a limitations section

**Learn from**
- [Robot Arm Control Lesson](https://shepai.github.io/tutorials/robotarm.html): connecting a language model (Ollama) to a MuJoCo arm
- [AI Agent Tutorial 3: Tool Calling](https://medium.com/@xzw0005/ai-agent-tutorial-3-tool-calling-b46d02d63eb6) (article)

---

## Working with a free Colab / Kaggle GPU

- Develop and debug locally, then clone the repo into Colab for heavy runs.
- Use the GPU for **YOLO fine-tuning**. State-based SAC/PPO is bottlenecked by CPU physics, so a GPU barely helps.
- Save checkpoints to Google Drive regularly (SB3 `CheckpointCallback`), since sessions can disconnect.
- Keep logic in repo modules and call it from thin notebooks, so no code lives only in a notebook.
- Download trained weights (YOLO, RL policy) and run the final demo on a laptop CPU.

## Results (to be filled with measured numbers)

| Experiment | Metric | Result |
|---|---|---|
| FK validation | Max position error vs MuJoCo | TBD |
| IK | Success rate over N random targets | TBD |
| PID vs computed torque | RMS tracking error (no payload / payload) | TBD |
| Detection | Precision / recall (synthetic test set) | TBD |
| Pick-and-place | Success rate over N trials | TBD |
| RL vs classical IK | Success rate, final error, smoothness | TBD |

## Limitations (to update)

- Simulation only; no sim-to-real transfer yet.
- Grasping uses a scripted primitive, not a learned grasp.
- RL is state-based (not pixel-based) to stay CPU-friendly.

## Next steps / ideas

- Pixel-based or domain-randomized RL policy trained on a GPU
- Learned grasping
- Sim-to-real on a low-cost arm
- ROS 2 / MoveIt 2 integration

## References

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control* (Cambridge University Press, 2017)
- [MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Ultralytics YOLO](https://www.ultralytics.com/)

## License

MIT (or your choice)
