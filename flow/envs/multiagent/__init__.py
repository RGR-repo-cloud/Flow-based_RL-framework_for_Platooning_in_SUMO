"""Empty init file to ensure documentation for multi-agent envs is created."""

from flow.envs.multiagent.base import MultiEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiWaveAttenuationPOEnv
from flow.envs.multiagent.ring.wave_attenuation import \
    MultiAgentWaveAttenuationPOEnv
from flow.envs.multiagent.ring.accel import AdversarialAccelEnv
from flow.envs.multiagent.ring.accel import MultiAgentAccelPOEnv
from flow.envs.multiagent.traffic_light_grid import MultiTrafficLightGridPOEnv
from flow.envs.multiagent.highway import MultiAgentHighwayPOEnv
from flow.envs.multiagent.merge import MultiAgentMergePOEnv
from flow.envs.multiagent.i210 import I210MultiEnv
from flow.envs.multiagent.platoon import PlatoonEnv, UnilateralPlatoonEnv, BilateralPlatoonEnv, FlatbedEnv, PloegEnv
from flow.envs.multiagent.constant_speed import ConstantSpeedEnv2V, ConstantSpeedEnv5V, ConstantSpeedEnv10V
from flow.envs.multiagent.custom_accel import CustomAccelEnv


__all__ = [
    'MultiEnv',
    'AdversarialAccelEnv',
    'MultiWaveAttenuationPOEnv',
    'MultiTrafficLightGridPOEnv',
    'MultiAgentHighwayPOEnv',
    'MultiAgentAccelPOEnv',
    'MultiAgentWaveAttenuationPOEnv',
    'MultiAgentMergePOEnv',
    'I210MultiEnv',
    'PlatoonEnv', 
    'ConstantSpeedEnv2V',
    'ConstantSpeedEnv5V',
    'ConstantSpeedEnv10V',
    'CustomAccelEnv',
    'UnilateralPlatoonEnv',
    'BilateralPlatoonEnv',
    'FlatbedEnv',
    'PloegEnv'

]
