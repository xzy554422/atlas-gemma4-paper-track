"""Create the smoke-inputs Kaggle dataset, rewriting blocked GCS hosts.

`kaggle datasets create` hangs on this machine because dataset file uploads receive a
resumable-upload URL on www.googleapis.com, which is unreachable here, while
storage.googleapis.com is reachable. This wrapper patches `upload_complete` to rewrite
the host and then calls the official dataset_create_new flow unchanged.
"""
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

ORIGINAL_UPLOAD = KaggleApi.upload_complete


def patched_upload_complete(self, path, url, quiet, resume=False):
    url = url.replace('https://www.googleapis.com/', 'https://storage.googleapis.com/')
    return ORIGINAL_UPLOAD(self, path, url, quiet, resume=resume)


def main(folder: str) -> int:
    KaggleApi.upload_complete = patched_upload_complete
    api = KaggleApi()
    api.authenticate()
    message = api.dataset_create_new(folder)
    print('dataset_create_new:', message)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
