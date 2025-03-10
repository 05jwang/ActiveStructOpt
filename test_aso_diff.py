from activestructopt.sampler.diffusion import Diffusion
from activestructopt.common.materialsproject import get_structure
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY", None)

if api_key is None:
    raise ValueError("API_KEY environment variable not set.")
initial_structure = get_structure("mp-54", api_key)
"""
self.model_path = '/home/hice1/adaftardar3/DiffCSP-Forked'
self.save_path = '/home/hice1/adaftardar3/DiffCSP-Forked'
self.formula = initial_structure.formula
self.num_evals = 10
self.batch_size = 500
self.step_lr = 1e-5
"""
pretrained_checkpoints = os.getenv(
    "MODEL_PATH", None
)  # Should look something like /home/hice1/jwang3450/DiffCSP_kt/pretrained_checkpoints
save_path = os.getenv(
    "SAVE_PATH", None
)  # Should look something like /home/hice1/jwang3450/DiffCSP_kt/sample_output
if pretrained_checkpoints is None:
    raise ValueError("MODEL_PATH environment variable not set.")
if save_path is None:
    raise ValueError("SAVE_PATH environment variable not set.")
diff_model = Diffusion(
    initial_structure,
    pretrained_checkpoints,
    save_path,
    10,
    500,
    1e-5,
)
new_structure = diff_model.sample()
print(new_structure)
