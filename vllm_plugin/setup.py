import shutil
from setuptools import setup

shutil.copy("../bridge/configuration_wbl.py", "./wbl/")
setup(
    name='wbl',
    version='1.2.0',
    packages=['wbl'],
    entry_points={
        'vllm.general_plugins': [
            "wbl_model = wbl:register",
        ],
    },
)
