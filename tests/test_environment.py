import os

import dotenv


def test_env_vars():
    dotenv.load_dotenv()

    data_dir_str = os.getenv('DATA_DIR')
    assert data_dir_str, (
        'DATA_DIR environment variable is not set, did you remember to configure the .env file?'
    )
