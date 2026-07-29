# ROS 2 Jazzy + Gazebo Harmonic + Gymnasium/SB3.
#
# Jazzy's official binaries target Gazebo Harmonic, so `ros-jazzy-ros-gz` pulls
# the correct gz-sim8 stack from the ROS repos directly. Do NOT add the OSRF
# packages.osrfoundation.org repo on top of this: mixing it with the ROS repo
# gives you two copies of gz-transport and a link-time failure that is
# genuinely miserable to diagnose.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# --- system + ROS/Gazebo packages -------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      python3-pip \
      python3-venv \
      python3-colcon-common-extensions \
      python3-vcstool \
      procps \
      psmisc \
      ros-jazzy-ros-gz \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-ros-gz-interfaces \
      ros-jazzy-teleop-twist-keyboard \
      ros-jazzy-rmw-cyclonedds-cpp \
      gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# --- python RL stack ---------------------------------------------------------
# Ubuntu 24.04 ships PEP 668 "externally managed" python. We are in a container
# and there is exactly one python; --break-system-packages is the right call
# here and avoids a venv that then has to be re-plumbed into ament.
#
# CPU-only torch: the dev machine has no NVIDIA GPU, and an MLP over a 26-dim
# observation is simulator-bound anyway. The CUDA wheels are ~2.5 GB of nothing.
RUN pip install --no-cache-dir --break-system-packages \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.4.*" \
 && pip install --no-cache-dir --break-system-packages \
      "gymnasium>=0.29,<2.0" \
      "stable-baselines3>=2.3,<3.0" \
      "numpy<2.0" \
      "tensorboard>=2.16" \
      "pyyaml" \
      "pytest>=8.0" \
      "pytest-timeout" \
      "ruff>=0.5"

# CycloneDDS: FastDDS shared-memory transport is unreliable under Docker
# Desktop's WSL2 backing store and produces intermittent discovery failures
# that look exactly like a bridge misconfiguration. Rule it out up front.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Headless by default. Set GZ_GUI=1 and a DISPLAY to override.
ENV QT_QPA_PLATFORM=offscreen
ENV GZ_SIM_RESOURCE_PATH=/ws/src/robot_rl_env/models:/ws/src/robot_rl_env/worlds

WORKDIR /ws

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
