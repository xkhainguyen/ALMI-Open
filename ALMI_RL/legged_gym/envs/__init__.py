from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry




from legged_gym.envs.h1_2.h1_2_wb_curriculum_config import H1_2_WholeBodyCfg, H1_2_WholeBodyCfgPPO
from legged_gym.envs.h1_2.h1_2_wb_curriculum_env import H1_2_WholeBody
task_registry.register( "h1_2_wb_curriculum", H1_2_WholeBody, H1_2_WholeBodyCfg(), H1_2_WholeBodyCfgPPO())

from legged_gym.envs.h1_2_upper.h1_2_upper_config import H1_2_WholeBodyCfg, H1_2_WholeBodyCfgPPO
from legged_gym.envs.h1_2_upper.h1_2_upper_env import H1_2_WholeBody
task_registry.register( "h1_2_upper", H1_2_WholeBody, H1_2_WholeBodyCfg(), H1_2_WholeBodyCfgPPO())


from legged_gym.envs.h1_2_lower.h1_2_lower_config import H1_2_WholeBodyCfg, H1_2_WholeBodyCfgPPO
from legged_gym.envs.h1_2_lower.h1_2_lower_env import H1_2_WholeBody
task_registry.register( "h1_2_lower", H1_2_WholeBody, H1_2_WholeBodyCfg(), H1_2_WholeBodyCfgPPO())

