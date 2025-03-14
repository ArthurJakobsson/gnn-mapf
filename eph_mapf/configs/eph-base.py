import datetime

from common import *

# Wandb setting
use_wandb = True
project = "FinalTest"
name = "eph_base"
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Network setting
model_target = "eph_mapf.src.models.eph.Network"


# For saving model
save_path = "./eph_mapf/saved_models/eph_net"

##### Note #####
# If you want to override some parameters from the common/ folder, just
# override down here. For examples:
# active_agent_radius = 1

use_wandb_test = False
test_env_settings = (
    (40, 4, 0.3),
    (40, 8, 0.3),
    (40, 16, 0.3),
    (40, 32, 0.3),
    (40, 64, 0.3),
    (80, 4, 0.3),
    (80, 8, 0.3),
    (80, 16, 0.3),
    (80, 32, 0.3),
    (80, 64, 0.3),
    (80, 128, 0.3),
)  # map length, number of agents, density
