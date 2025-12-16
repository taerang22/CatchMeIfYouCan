#!/bin/bash
# Source the ROS2 workspace and fix shebangs for conda compatibility

# Activate conda environment if not already active
if [[ "$CONDA_DEFAULT_ENV" != "kinova" ]]; then
    source /home/iconlab/miniconda3/etc/profile.d/conda.sh
    conda activate kinova
fi

# Source the workspace
source /home/iconlab/C106/install/setup.bash

# Fix shebangs to use conda Python
for script in /home/iconlab/C106/install/mpc/lib/mpc/mpc_node \
              /home/iconlab/C106/install/perception/lib/perception/perception_node \
              /home/iconlab/C106/install/control/lib/control/control_node \
              /home/iconlab/C106/install/control/lib/control/mpc_controller \
              /home/iconlab/C106/install/control/lib/control/state_publisher \
              /home/iconlab/C106/install/prediction/lib/prediction/prediction_node; do
    [ -f "$script" ] && sed -i '1s|.*|#!/home/iconlab/miniconda3/envs/kinova/bin/python|' "$script"
done

echo "Kinova workspace ready!"
