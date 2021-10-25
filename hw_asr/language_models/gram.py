import gzip
import shutil
import wget

from pathlib import Path


def pretrained_language_model(language_model_path: str = '3-gram.pruned.1e-7.arpa.gz') -> str:
    language_model_path = Path(language_model_path)
    if not language_model_path.exists():
        print('Downloading pretrained 3-gram language model.')
        lm_url = 'https://www.openslr.org/resources/11/' + str(anguage_model_path)
        language_model_path = wget.download(lm_url)
        print('Downloaded pretrained 3-gram language model.')

    upper_language_model_path = language_model_path.with_suffix('')
    if not upper_language_model_path.exists():
        with gzip.open(language_model_path, 'rb') as f_zipped:
            with open(upper_language_model_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')

    lower_language_model_path = Path('lowercase_' + str(language_model_path))
    if not lower_language_model_path.exists():
        with upper_language_model_path.open('r') as f_upper:
            with lower_language_model_path.open('w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
        print('Converted language model file to lowercase.')

    return str(lower_language_model_path)
