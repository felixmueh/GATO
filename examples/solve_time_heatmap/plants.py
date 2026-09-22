"""Model-specific inputs; importing metadata does not load CUDA or simulation libraries."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PLANTS = ('tiago_right', 'indy7')
DEFAULT_PLANT = 'tiago_right'
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tiago_src'))


def plant_metadata(plant):
    if plant == 'tiago_right':
        return dict(plant=plant, plant_label='TIAGo right arm',
                    urdf='gato/dynamics/tiago_right/tiago_right_arm.urdf',
                    nq=7, nv=7, start_pose='comfortable_high_clearance',
                    reference='tiago_horizontal', tracking_frame='arm_right_tool_link in torso_lift_link')
    if plant == 'indy7':
        return dict(plant=plant, plant_label='Indy7',
                    urdf='examples/indy7_description/indy7.urdf',
                    nq=6, nv=6, start_pose='ready', reference='historical_indy7',
                    tracking_frame='joint-6 origin in world')
    raise ValueError(f'Unknown plant: {plant}')


def end_effector_position(plant, model, data, q):
    import pinocchio as pin
    pin.forwardKinematics(model, data, q)
    if plant == 'tiago_right':
        pin.updateFramePlacements(model, data)
        torso = model.getFrameId('torso_lift_link')
        tool = model.getFrameId('arm_right_tool_link')
        return (data.oMf[torso].inverse() * data.oMf[tool]).translation.copy()
    plant_metadata(plant)  # Reject accidental unsupported models.
    return data.oMi[6].translation.copy()


def plant_inputs(plant, protocol='wall-time'):
    import numpy as np
    import pinocchio as pin
    meta = plant_metadata(plant)
    urdf = ROOT / meta['urdf']
    if plant == 'tiago_right':
        if protocol != 'wall-time':
            raise ValueError('Historical reference protocol requires --plant indy7')
        from gato_tiago.config import TIAGO_RIGHT_START_CONFIGS
        from gato_tiago.references import tiago_horizontal_figure8
        q = np.asarray(TIAGO_RIGHT_START_CONFIGS[meta['start_pose']], dtype=float)
        model = pin.buildModelFromUrdf(str(urdf))
        position = end_effector_position(plant, model, model.createData(), q)
        reference, _ = tiago_horizontal_figure8(.01, position, cycles=1)
    else:
        from bsqp.common import figure8
        from bsqp.config import INDY7_START_CONFIGS, FIG8_DEFAULT_PARAMS
        q = np.asarray(INDY7_START_CONFIGS['ready'], dtype=float)
        params = FIG8_DEFAULT_PARAMS | ({'cycles': 1} if protocol == 'wall-time' else {})
        reference = figure8(.01, **params)
    return urdf, np.r_[q, np.zeros(meta['nv'])], reference
