# %% [markdown]
# ## Setup Environment

# %%
!python3 -c "import monai" || pip install -q "monai-weekly[nibabel]"
!python3 -c "import matplotlib" || pip install -q matplotlib
%matplotlib inline

# %%
!pip install lmdb
!pip install gdown
!pip install pytorch-ignite
!pip install psutil
!pip install psutil
!pip install einops
!pip install transformers
!pip install mlflow
!pip install pynrrd
!pip install ITK 
!pip install clearml
!pip install Tensorboard
!pip install boto3
!pip install sagemaker

# %% [markdown]
# ## Import Modules

# %%
import os
import json
import boto3
import shutil
import tarfile
import tempfile
import time
import glob
import uuid

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import sagemaker
from monai.visualize import blend_images
from botocore.config import Config
from monai.config import print_config



# %% [markdown]
# ## Utils

# %%

def upload_file(input_location, s3_bucket, model_name, sagemaker_session):
    prefix = f"{model_name}/input"
    return sagemaker_session.upload_data(
        input_location,
        bucket=s3_bucket,
        key_prefix=prefix,
        extra_args={"ContentType": "application/json"},
    )
    
    
def get_output(output_location, s3_client):
    output_url = urllib.parse.urlparse(output_location)
    bucket = output_url.netloc
    key = output_url.path[1:]

    print(f'output_url = {output_url}')
    print(f'bucket = {bucket}')
    print(f'key = {key}')
    
    # wait 4 mins for the output to be arrive
    i=0
    while True:
        if i > 12: raise Exception('Model Timeout Error')
        try:

            return s3_client.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print(f"waiting for output. time = {time.time()}")
                i+=1

                time.sleep(20)
                continue


# %%
model_bucket_name   = "dev-ccipd-swin-unetr-models"
case_study_dir           = "/home/bratsdata/casestudy/"
output_tarfile_dir  = "/home/bratsdata/tarfiles/"

# %%


# %% [markdown]
# ## Setup S3 Client

# %%
os.environ['AWS_PROFILE'] = "model"
os.environ['AWS_DEFAULT_REGION'] = "us-east-2"
aws_access_key_id = os.environ['ACCESS_KEY_ID']
aws_secret_access_key = os.environ['SECRET_ACCESS_KEY']

boto_session = boto3.session.Session(profile_name='model', region_name='us-east-2')
sm_runtime = boto3.client('sagemaker-runtime')
sm_session = sagemaker.session.Session(boto_session=boto_session)

config = Config(
    region_name='us-east-2',
    signature_version='s3v4'
)

s3_client = boto3.client(
    's3',
    config=config,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# %% [markdown]
# ## Load Brats Sample

# %%
sample_num = "00006"

flair_path = case_study_dir + f'BraTS2021_{sample_num}/BraTS2021_{sample_num}_flair.nii.gz'
t1ce_path  = case_study_dir + f'BraTS2021_{sample_num}/BraTS2021_{sample_num}_t1ce.nii.gz'
t1_path    = case_study_dir + f'BraTS2021_{sample_num}/BraTS2021_{sample_num}_t1.nii.gz'
t2_path    = case_study_dir + f'BraTS2021_{sample_num}/BraTS2021_{sample_num}_t2.nii.gz'
seg_path   = case_study_dir + f'BraTS2021_{sample_num}/BraTS2021_{sample_num}_seg.nii.gz'

flair = np.rot90(nib.load(flair_path).get_fdata())
t1ce  = np.rot90(nib.load(t1ce_path).get_fdata())
t1    = np.rot90(nib.load(t1_path).get_fdata())
t2    = np.rot90(nib.load(t2_path).get_fdata())
seg   = np.rot90(nib.load(seg_path).get_fdata())

# %% [markdown]
# ## Visualize Sample

# %%
im_slice = 82
print(f"image shape: {flair.shape}, label shape: {seg.shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(flair[:, :, im_slice], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(seg[:, :, im_slice])
plt.show()

# %% [markdown]
# ## Blend Label and Image

# %%
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(flair[:, :, im_slice], cmap="bone")

# Blend Image
blend = blend_images(
    np.expand_dims(flair[:,:, im_slice], axis=0),
    np.expand_dims(seg[:,:, im_slice], axis=0)
)
plt.subplot(1, 2, 2)
plt.title("Blend Image")
plt.imshow(np.moveaxis(blend, 0, -1), cmap='bone')
plt.show()

# %% [markdown]
# ## Utils

# %%
datalist_dir = output_tarfile_dir

def rename_data_file(old_fn, new_fn):
    try:
        os.rename(old_fn, new_fn)
        print(f"File '{old_fn}' has been renamed to '{new_fn}'.")
    except FileNotFoundError:
        print(f"File '{old_fn}' not found.")
    except FileExistsError:
        print(f"File '{new_fn}' already exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
def clean_dataset_dir(directory):
    files = glob.glob(os.path.join(directory, '*'))
    for file_path in files:
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
                
def tar_datalist(directory, datalist_dict):
    files = os.listdir(directory)
    key = uuid.uuid4()
    tar_name = f"{key}.tar.gz"
    datalist_name = f"{key}.json"
    files.append(datalist_name)
    try:
        
        # Create a datalist.json
        with open(os.path.join(directory, datalist_name), "w") as json_file:
            json.dump(datalist_dict, json_file, indent=4)
        
        # Create a tar.gz archive
        with tarfile.open(os.path.join(directory, tar_name), "w:gz") as tar:
            for file_name in files:
                file_path = os.path.join(directory, file_name)
                if os.path.exists(file_path):
                    tar.add(file_path, arcname=file_name)
                    print(f"Added {file_name} to {tar_name}")
                else:
                    print(f"File {file_name} not found in {directory}")

        print(f"Created {tar_name}")
        
        return tar_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# %% [markdown]
# ## Rename data files

# %%
postfix = f"{uuid.uuid4()}.tar.gz"
rename_data_file(flair_path, f"{datalist_dir}/flair-{postfix}.nii.gz")
rename_data_file(t1ce_path, f"{datalist_dir}/t1ce-{postfix}.nii.gz")
rename_data_file(t1_path, f"{datalist_dir}/t1-{postfix}.nii.gz")
rename_data_file(t2_path, f"{datalist_dir}/t2-{postfix}.nii.gz")

datalist_dict = {
    "flair": f"flair-{postfix}.nii.gz",
    "t1ce": f"t1ce-{postfix}.nii.gz",
    "t1": f"t1-{postfix}.nii.gz",
    "t2": f"t2-{postfix}.nii.gz"
}
    

# %%
input_nii_path = tar_datalist(datalist_dir, datalist_dict)

# %% [markdown]
# ## Upload datalist tar file

# %%
s3_data_bucket='dev-ccipd-from-dicom-server-nnunet'
s3_inference_bucket = 'dev-ccipd-sagemaker-async-inference'

# upload input datalist tar file to s3
s3_client.upload_file(
    os.path.join(datalist_dir, input_nii_path),
    s3_data_bucket, 
    input_nii_path
)

# %% [markdown]
# ## Prepare inference payload

# %%
model_name='swin-UNETR-3D-Brain-4modality'
endpoint_name = 'swin-UNETR-3D-Brain-4modality-endpoint'

## save payload json
payload={
    's3_path': {
        'bucket': s3_data_bucket,
        'key': input_nii_path
    }
}

with tempfile.NamedTemporaryFile() as tmp_json:
    fn = tmp_json.name 
    with open(fn, 'w') as f:
        json.dump(payload, f)
    input_s3_location = upload_file(fn, s3_inference_bucket, model_name, sm_session)
    
print(input_s3_location)

# %% [markdown]
# ## Invoke Model Endpoint

# %%
import urllib, time
from botocore.exceptions import ClientError
### ASYNC INVOKE
response = sm_runtime.invoke_endpoint_async(
    EndpointName=endpoint_name, 
    InputLocation=input_s3_location,
    ContentType='application/json'
)
output_location = response["OutputLocation"]
print(output_location)

output = get_output(output_location, s3_client)
output_file = json.loads(output)['outputFile']
print(output_file)

# %% [markdown]
# ## Visualize Prediction

# %%
with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
    fn = tmp.name
    s3_client.download_file(
        output_file.split('/')[-2],
        output_file.split('/')[-1],
        fn
    )
    pred = np.load(fn)
</

# %%


# %%
pred   = np.rot90(pred)
plt.figure("image", (18, 6))
plt.subplot(1, 3, 1)
plt.title("image")
plt.imshow(flair[:, :, im_slice], cmap="bone")
plt.subplot(1, 3, 2)
plt.title("label")
plt.imshow(seg[:, :, im_slice])
plt.subplot(1, 3, 3)
plt.title("Prediction")
plt.imshow(pred[:, :, im_slice])
plt.show()

# %%


# %%
clean_dataset_dir(output_tarfile_dir)

# %%



