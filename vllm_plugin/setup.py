import shutil
from setuptools import setup

shutil.copy("../bridge/configuration_Vaetki.py", "./Vaetki/")
setup(
    name='Vaetki',
    version='1.2.0',
    packages=['Vaetki'],
    entry_points={
        'vllm.general_plugins': [
            "Vaetki_model = Vaetki:register",
        ],
    },
)
